#!/usr/bin/env python3
"""Target-only regression with independent goldens and capture ownership checks.

Run after sourcing the CANN SDK. Compile tracing and shared-library targets first.
Reports retain source hashes and stop at the first failed or incomplete check.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import sys
sys.dont_write_bytecode = True

from npu_data import SUITES, sha256
from run_benchmark import build_identity


def main():
    root = Path(__file__).resolve().parents[1]
    work = root.parent / '.private/runtime/npu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, default=work / 'build')
    parser.add_argument('--runs', type=Path, default=work / 'runs')
    parser.add_argument('--large-prefix', type=Path)
    parser.add_argument('--suites', nargs='+', choices=SUITES, default=['native'])
    parser.add_argument('--modes', nargs='+', choices=('ordinary','capture-cold','capture-chain','capture-streams'),
                        default=['ordinary','capture-cold'])
    parser.add_argument('--shared', action='store_true')
    args = parser.parse_args()
    build, runs = args.build.resolve(), args.runs.resolve()
    # Never overwrite provenance or append profiles to an earlier execution.
    runs.mkdir(parents=True, exist_ok=False)
    data = runs / 'data'
    data.mkdir(exist_ok=True)
    result = dict(kernel_sha256=sha256(root/'kernel.asc'),
                  runner_sha256=sha256(root/'tests/npu_runner.asc'),
                  steps=[], reports={}, resources={})

    def save():
        (runs/'results.json').write_text(json.dumps(result, indent=2)+'\n')

    def run(name, command, env=None):
        with (runs/(name+'.log')).open('w') as log:
            completed = subprocess.run(list(map(str, command)), cwd=runs, env=env,
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
            release_syncs = len(re.findall(r'^INTERNAL_RELEASE_SYNC ret=0$', log, re.M))
            if freed != allocated or error or (allocated > 0 and release_syncs == 0):
                raise RuntimeError(name+' scratch cache was not fully released or an internal ACL call failed')
            result['resources'][name] = dict(allocations=allocated, frees=freed,
                                            release_synchronizations=release_syncs,
                                            error=error)
        run(name+'-verify',['python3',root/'tests/npu_data.py','--prefix',prefix,'--verify',output])
        report = json.loads(output.with_suffix('.report.json').read_text())
        count = len(json.loads(prefix.with_suffix('.json').read_text()))
        if report['cases'] != count or report['passed'] != count or report['failures']:
            raise RuntimeError(name+' incomplete independent verification')
        result['reports'][name] = report
        save()

    try:
        identity = build_identity(build/'bmmms_npu_runner')
        if any(identity[key] != result[key] for key in ('kernel_sha256', 'runner_sha256')):
            raise RuntimeError('runner build does not match current source; rebuild before testing')
        result['build'] = identity
        if args.shared:
            for filename in ('libbmmms_kernel.so', 'bmmms_shared_runner'):
                shared_identity = build_identity(build/filename)
                if any(shared_identity[key] != result[key] for key in ('kernel_sha256', 'runner_sha256')):
                    raise RuntimeError('shared build does not match current source: '+filename)
                result.setdefault('shared_builds', {})[filename] = shared_identity
        save()
        for suite in args.suites:
            prefix = data/suite
            run('generate-'+suite,['python3',root/'tests/npu_data.py','--suite',suite,'--prefix',prefix])
            result.setdefault('data', {})[suite] = json.loads(prefix.with_suffix('.manifest.json').read_text())
            save()
            for selected_mode in args.modes:
                mode = None if selected_mode == 'ordinary' else '--'+selected_mode
                check(suite+'-'+(mode or 'ordinary').removeprefix('--'),prefix,mode)
            if args.shared:
                check(suite+'-cold-shared',prefix,'--capture-cold',True)
        if args.large_prefix:
            prefix = args.large_prefix.resolve()
            result['large_input_sha256'] = sha256(prefix.with_suffix('.bin'))
            for selected_mode in args.modes:
                mode = None if selected_mode == 'ordinary' else '--'+selected_mode
                check('large-'+(mode or 'ordinary').removeprefix('--'),prefix,mode)
            if args.shared:
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
        observed_cores = dict((int(case), int(cores)) for case, cores in
            re.findall(r'CASE (\d+) repeat=\d+[^\n]* cores=(\d+)',
                       (runs/'launch-rule-profile.log').read_text()))
        if len(observed_cores) != len(metadata):
            raise RuntimeError('launch profile lacks effective per-case core counts')
        for spec in metadata:
            spec['cores'] = observed_cores[spec['index']]
        rows = verify_launches(files[0],metadata,5)
        result['launch_rule'] = dict(expected=len(metadata)*5,observed=len(rows),passed=True,
                                    dispatches=[rows[i*5]['Op Name'] for i in range(len(metadata))])
        result['workflow_passed'] = True
        save();print('NPU_CHECKS_PASS',flush=True)
    except Exception as error:
        result['workflow_passed'] = False
        result['error'] = str(error)
        save();print('NPU_CHECKS_FAIL',str(error),flush=True)
        raise


if __name__ == '__main__':
    main()
