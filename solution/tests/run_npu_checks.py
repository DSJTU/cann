#!/usr/bin/env python3
"""Target-only regression with independent goldens and capture ownership checks.

Run after sourcing the CANN SDK. Compile tracing and shared-library targets first.
Reports retain source hashes and stop at the first failed or incomplete check.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from check_launch_profile import dispatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, default=Path('build-npu'))
    parser.add_argument('--runs', type=Path, default=Path('runs/hybrid'))
    parser.add_argument('--large-prefix', type=Path)
    parser.add_argument('--suites', nargs='+', default=['correctness','extended','performance','short-cube','long-cube'])
    parser.add_argument('--modes', nargs='+', choices=('ordinary','capture-cold','capture-chain','capture-streams'),
                        default=['ordinary','capture-cold','capture-chain','capture-streams'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build, runs = args.build.resolve(), args.runs.resolve()
    runs.mkdir(parents=True, exist_ok=True)
    data = runs / 'data'
    data.mkdir(exist_ok=True)
    result = dict(kernel_sha256=hashlib.sha256((root/'kernel.asc').read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256((root/'tests/npu_runner.asc').read_bytes()).hexdigest(),
                  steps=[], reports={}, resources={})

    def save():
        (runs/'results.json').write_text(json.dumps(result, indent=2)+'\n')

    def run(name, command, env=None):
        with (runs/(name+'.log')).open('w') as log:
            completed = subprocess.run(list(map(str, command)), cwd=root, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        result['steps'].append(dict(name=name, exit_code=completed.returncode))
        save()
        print(name, 'exit=', completed.returncode, flush=True)
        if completed.returncode:
            raise RuntimeError(name+' failed; see '+str(runs/(name+'.log')))

    def check(name, prefix, mode=None, shared=False):
        output = data/(name+'.out.bin')
        executable = build/('bmmms_shared_runner' if shared else 'bmmms_npu_runner')
        env = dict(os.environ, BMMMS_SHARED_LIBRARY=str(build/'libbmmms_kernel.so'))
        command = ['timeout', '600', executable, prefix.with_suffix('.bin'), output]
        if mode:
            command.append(mode)
        run(name, command, env)
        if not shared:
            log = (runs/(name+'.log')).read_text()
            summaries = re.findall(r'INTERNAL_SUMMARY allocations=(\d+) frees=(\d+) error=(\d+)',log)
            if len(summaries) != 1:
                raise RuntimeError(name+' needs BMMMS_TRACE_RUNTIME=ON')
            allocated, freed, error = map(int,summaries[0])
            metadata = json.loads(prefix.with_suffix('.json').read_text())
            cubes = sum(dispatch(v['shape']) == 'fused_kernel' for v in metadata)
            registrations = len(re.findall(r'^INTERNAL_REGISTER ret=0$', log, re.M))
            release_syncs = len(re.findall(r'^INTERNAL_RELEASE_SYNC ret=0$', log, re.M))
            if freed != allocated or error or registrations or release_syncs != allocated:
                raise RuntimeError(name+' scratch cache was not fully released or an internal ACL call failed')
            if (cubes > 0) != (allocated > 0):
                raise RuntimeError(name+' unexpected scratch-cache allocation')
            result['resources'][name] = dict(cube_cases=cubes, allocations=allocated, frees=freed,
                                            registrations=registrations, release_synchronizations=release_syncs,
                                            error=error)
        run(name+'-verify',['python3',root/'tests/npu_data.py','--prefix',prefix,'--verify',output])
        report = json.loads(output.with_suffix('.report.json').read_text())
        count = len(json.loads(prefix.with_suffix('.json').read_text()))
        if report['cases'] != count or report['passed'] != count or report['failures']:
            raise RuntimeError(name+' incomplete independent verification')
        result['reports'][name] = report
        save()

    try:
        for suite in args.suites:
            prefix = data/suite
            run('generate-'+suite,['python3',root/'tests/npu_data.py','--suite',suite,'--prefix',prefix])
            for selected_mode in args.modes:
                mode = None if selected_mode == 'ordinary' else '--'+selected_mode
                check(suite+'-'+(mode or 'ordinary').removeprefix('--'),prefix,mode)
            check(suite+'-cold-shared',prefix,'--capture-cold',True)
        if args.large_prefix:
            prefix = args.large_prefix.resolve()
            result['large_input_sha256'] = hashlib.sha256(prefix.with_suffix('.bin').read_bytes()).hexdigest()
            for selected_mode in args.modes:
                mode = None if selected_mode == 'ordinary' else '--'+selected_mode
                check('large-'+(mode or 'ordinary').removeprefix('--'),prefix,mode)
            check('large-cold-shared',prefix,'--capture-cold',True)
            check('large-benchmark',prefix,'--benchmark')
        # Numerical checks do not prove the contest's exactly-one-launch rule.
        prefix = data/'launch-rule'
        run('generate-launch-rule',['python3',root/'tests/npu_data.py','--suite','launch-rule','--prefix',prefix])
        output = data/'launch-rule.out.bin'
        directory = runs/'launch-rule-profile'
        application = ' '.join(map(str,[build/'bmmms_npu_runner',prefix.with_suffix('.bin'),output,'--profile-five']))
        run('launch-rule-profile',['timeout','600','msprof','--output='+str(directory),'--application='+application])
        run('launch-rule-verify',['python3',root/'tests/npu_data.py','--prefix',prefix,'--verify',output])
        from check_launch_profile import verify as verify_launches
        files = list(directory.rglob('op_summary*.csv'))
        if len(files) != 1:
            raise RuntimeError('ambiguous or missing launch profile')
        metadata = json.loads(prefix.with_suffix('.json').read_text())
        rows = verify_launches(files[0],metadata,5)
        result['launch_rule'] = dict(expected=len(metadata)*5,observed=len(rows),passed=True)
        result['workflow_passed'] = True
        save();print('HYBRID_NPU_CHECKS_PASS',flush=True)
    except Exception as error:
        result['workflow_passed'] = False
        result['error'] = str(error)
        save();print('HYBRID_NPU_CHECKS_FAIL',str(error),flush=True)
        raise


if __name__ == '__main__':
    main()
