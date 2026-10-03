#!/usr/bin/env python3
"""Validate the integrated candidate and profile against a pinned baseline.

Run from an isolated stage containing candidate/solution, baseline/solution,
and candidate-manifest.json. Input files remain on the authorized NPU lab.
"""
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
RUNS.mkdir(exist_ok=True)
CANDIDATE = ROOT / 'candidate/solution'
BASELINE = ROOT / 'baseline/solution'
LONG = Path('/mnt/workspace/bmmms/long-k-grid-20261003/runs/benchmark')
SHORT = Path('/mnt/workspace/bmmms/perf-20261002/cache-block/solution/lab-data/large-short-k')
result = dict(manifest=json.loads((ROOT / 'candidate-manifest.json').read_text()), steps=[], reports={}, resources={})


def save():
    (RUNS / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


def run(name, command, cwd=CANDIDATE):
    with (RUNS / (name + '.log')).open('w') as log:
        process = subprocess.run(list(map(str, command)), cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    result['steps'].append(dict(name=name, exit_code=process.returncode))
    save()
    print(name, 'exit=', process.returncode, flush=True)
    if process.returncode:
        raise RuntimeError(name + ' failed; inspect its log')


def verify(name, prefix, output):
    run(name + '-verify', ['python3', CANDIDATE / 'tests/npu_data.py', '--prefix', prefix, '--verify', output])
    report = json.loads(output.with_suffix('.report.json').read_text())
    assert report['passed'] == report['cases'] == len(json.loads(prefix.with_suffix('.json').read_text()))
    assert not report['failures']
    result['reports'][name] = report
    save()


def main():
    for name, source in (('candidate', CANDIDATE), ('baseline', BASELINE)):
        actual = hashlib.sha256((source / 'kernel.asc').read_bytes()).hexdigest()
        assert actual == result['manifest'][name + '_sha256']
        run(name + '-configure', ['cmake', '-S', 'tests', '-B', 'build-npu', '-DNPU_ARCH=dav-2201',
                                  '-DBMMMS_TRACE_RUNTIME=ON', '-DBMMMS_BUILD_SHARED_TEST=' +
                                  ('ON' if name == 'candidate' else 'OFF')], source)
        run(name + '-build', ['cmake', '--build', 'build-npu', '-j4'], source)
    for suite in ('long-cube', 'cube-finish', 'smoke'):
        run('generate-' + suite, ['python3', 'tests/npu_data.py', '--suite', suite, '--prefix', RUNS / suite])
    result['input_sha256'] = {suite: hashlib.sha256((RUNS / (suite + '.bin')).read_bytes()).hexdigest()
                              for suite in ('long-cube', 'cube-finish', 'smoke')}
    for name, suite, mode, shared in (
            ('ordinary', 'long-cube', None, False), ('cold', 'long-cube', '--capture-cold', False),
            ('chain', 'long-cube', '--capture-chain', False), ('streams', 'long-cube', '--capture-streams', False),
            ('shared-cold', 'long-cube', '--capture-cold', True),
            ('short', 'cube-finish', None, False), ('small', 'smoke', None, False)):
        prefix = RUNS / suite
        output = RUNS / (name + '.out.bin')
        executable = 'bmmms_shared_runner' if shared else 'bmmms_npu_runner'
        command = ['timeout', '300', 'env', 'BMMMS_SHARED_LIBRARY=' + str(CANDIDATE / 'build-npu/libbmmms_kernel.so'),
                   CANDIDATE / 'build-npu' / executable, prefix.with_suffix('.bin'), output]
        if mode:
            command.append(mode)
        run(name, command)
        verify(name, prefix, output)
        if not shared:
            text = (RUNS / (name + '.log')).read_text()
            counts = re.findall(r'INTERNAL_SUMMARY allocations=(\d+) frees=(\d+) error=(\d+)', text)
            assert len(counts) == 1
            allocated, freed, error = map(int, counts[0])
            assert allocated == freed and error == 0
            assert not re.search(r'^INTERNAL_.*ret=[1-9]', text, re.M)
            if name in ('cold', 'streams'):
                assert 'INTERNAL_SYNC' not in text
            result['resources'][name] = dict(allocations=allocated, frees=freed, error=error)
            save()
    sys.path.insert(0, str(CANDIDATE / 'tests'))
    from compare_cube_profile import samples, vector_samples
    result['profiles'] = {}
    result['shape_summary'] = []
    for suite, prefix, expected in (
            ('long', LONG, '143cb57afe9d915c5a7eef121b13dbe348ae17c68d41ff853fdc4520676048b5'),
            ('short', SHORT, '55d9cca489f2f6ae9b1a59c3ed94a7c826a43a04fbe6573e7798d17bb1cac8ca')):
        assert hashlib.sha256(prefix.with_suffix('.bin').read_bytes()).hexdigest() == expected
        result['input_sha256'][suite + '-benchmark'] = expected
        metadata = json.loads(prefix.with_suffix('.json').read_text())
        profiles = {}
        for name, source in (('baseline', BASELINE), ('candidate', CANDIDATE)):
            label = suite + '-' + name
            output = RUNS / (label + '.out.bin')
            profile = RUNS / (label + '-profile')
            application = str(source / 'build-npu/bmmms_npu_runner') + ' ' + str(prefix.with_suffix('.bin')) + ' ' + str(output) + ' --benchmark'
            run(label + '-profile', ['timeout', '600', 'msprof', '--output=' + str(profile), '--application=' + application], source)
            verify(label, prefix, output)
            files = list(profile.rglob('op_summary*.csv'))
            assert len(files) == 1
            if suite == 'long' and name == 'baseline':
                profiles[name] = [dict(interval_us=v, scores_us=v, finish_us=0, gap_us=0)
                                  for v in vector_samples(files[0], len(metadata))]
            else:
                profiles[name], _ = samples(files[0], len(metadata))
        rows = [dict(spec=s, baseline=b, candidate=c, speedup=b['interval_us'] / c['interval_us'])
                for s, b, c in zip(metadata, profiles['baseline'], profiles['candidate'])]
        result['profiles'][suite] = rows
        for shape in dict.fromkeys(tuple(s['shape']) for s in metadata):
            selected = [r for r in rows if tuple(r['spec']['shape']) == shape]
            result['shape_summary'].append(dict(suite=suite, shape=shape,
                baseline_us=statistics.median(r['baseline']['interval_us'] for r in selected),
                candidate_us=statistics.median(r['candidate']['interval_us'] for r in selected),
                median_speedup=statistics.median(r['speedup'] for r in selected),
                minimum_speedup=min(r['speedup'] for r in selected)))
        save()
    result['workflow_passed'] = True
    save()
    print(json.dumps(result['shape_summary'], indent=2), flush=True)
    print('LONG_K_CANDIDATE_CHECKPOINT_PASS', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        result['workflow_passed'] = False
        result['error'] = str(error)
        save()
        print('LONG_K_CANDIDATE_CHECKPOINT_FAIL', str(error), flush=True)
        raise
