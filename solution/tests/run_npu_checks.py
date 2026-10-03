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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, default=Path('build-npu'))
    parser.add_argument('--runs', type=Path, default=Path('runs/hybrid'))
    parser.add_argument('--large-prefix', type=Path)
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
            cubes = sum(1 for v in metadata if v['shape'][3] <= 128 and
                        v['shape'][1] >= 256 and v['shape'][2] >= 256 and
                        v['shape'][1]*v['shape'][2]*v['shape'][3] >= 1 << 24)
            calls = {'--capture-cold': 1, '--capture-chain': 4,
                     '--capture-streams': 2, '--streams': 4, '--benchmark': 12}.get(mode, 2)
            registrations = len(re.findall(r'^INTERNAL_REGISTER ret=0$',log,re.M))
            syncs = len(re.findall(r'^INTERNAL_SYNC ret=0$',log,re.M))
            expected_registers = cubes*(2 if mode == '--capture-streams' else 1) if mode and mode.startswith('--capture') else 0
            expected_syncs = cubes if mode == '--capture-chain' else 0 if mode and mode.startswith('--capture') else cubes*calls
            if (allocated != cubes*calls*2 or freed != allocated or error or
                    registrations != expected_registers or syncs != expected_syncs):
                raise RuntimeError(name+' resource counts or runtime status differ from expected ownership')
            result['resources'][name] = dict(cube_cases=cubes,allocations=allocated,frees=freed,
                                            registrations=registrations,ordinary_synchronizations=syncs,error=error)
        run(name+'-verify',['python3',root/'tests/npu_data.py','--prefix',prefix,'--verify',output])
        report = json.loads(output.with_suffix('.report.json').read_text())
        count = len(json.loads(prefix.with_suffix('.json').read_text()))
        if report['cases'] != count or report['passed'] != count or report['failures']:
            raise RuntimeError(name+' incomplete independent verification')
        result['reports'][name] = report
        save()

    try:
        for suite in ('correctness','extended','performance'):
            prefix = data/suite
            run('generate-'+suite,['python3',root/'tests/npu_data.py','--suite',suite,'--prefix',prefix])
            for mode in (None,'--capture-cold','--capture-chain','--capture-streams'):
                check(suite+'-'+(mode or 'ordinary').removeprefix('--'),prefix,mode)
            check(suite+'-cold-shared',prefix,'--capture-cold',True)
        if args.large_prefix:
            prefix = args.large_prefix.resolve()
            result['large_input_sha256'] = hashlib.sha256(prefix.with_suffix('.bin').read_bytes()).hexdigest()
            for mode in (None,'--capture-cold','--capture-chain','--capture-streams'):
                check('large-'+(mode or 'ordinary').removeprefix('--'),prefix,mode)
            check('large-cold-shared',prefix,'--capture-cold',True)
            check('large-benchmark',prefix,'--benchmark')
        result['workflow_passed'] = True
        save();print('HYBRID_NPU_CHECKS_PASS',flush=True)
    except Exception as error:
        result['workflow_passed'] = False
        result['error'] = str(error)
        save();print('HYBRID_NPU_CHECKS_FAIL',str(error),flush=True)
        raise


if __name__ == '__main__':
    main()
