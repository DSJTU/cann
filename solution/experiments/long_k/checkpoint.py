#!/usr/bin/env python3
"""Target-SDK design checkpoint; numeric failures are retained as evidence."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
RUNS.mkdir(exist_ok=True)
result = dict(manifest=json.loads((ROOT / 'manifest.json').read_text()), steps=[], reports={}, profiles={})


def save():
    (RUNS / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


def run(name, command, cwd=ROOT):
    with (RUNS / (name + '.log')).open('w') as log:
        process = subprocess.run(list(map(str, command)), cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    result['steps'].append(dict(name=name, exit_code=process.returncode))
    save()
    print(name, 'exit=', process.returncode, flush=True)
    return process.returncode == 0


def verify(name, prefix, output):
    ok = run(name + '-verify', ['python3', ROOT / 'baseline/solution/tests/npu_data.py',
                               '--prefix', prefix, '--verify', output])
    report = output.with_suffix('.report.json')
    if report.exists():
        result['reports'][name] = json.loads(report.read_text())
        save()
    return ok


def main():
    for suite in ('precision', 'benchmark'):
        if not run('generate-' + suite, ['python3', ROOT / 'data.py', RUNS / suite, '--suite', suite]):
            raise RuntimeError('input generation failed')
    result['input_sha256'] = {suite: hashlib.sha256((RUNS / (suite + '.bin')).read_bytes()).hexdigest()
                              for suite in ('precision', 'benchmark')}
    built = []
    for name in result['manifest']['variants']:
        source = ROOT / name / 'solution'
        if not run(name + '-configure', ['cmake', '-S', 'tests', '-B', 'build-npu',
                                         '-DNPU_ARCH=dav-2201', '-DBMMMS_TRACE_RUNTIME=ON'], source):
            continue
        if run(name + '-build', ['cmake', '--build', 'build-npu', '-j4'], source):
            built.append(name)
    result['built'] = built
    save()
    for name in built:
        source = ROOT / name / 'solution'
        prefix = RUNS / 'precision'
        output = RUNS / (name + '-precision.out.bin')
        if run(name + '-precision', ['timeout', '300', source / 'build-npu/bmmms_npu_runner',
                                    prefix.with_suffix('.bin'), output]):
            verify(name + '-precision', prefix, output)
    sys.path.insert(0, str(ROOT / 'baseline/solution/tests'))
    from compare_cube_profile import samples, vector_samples
    metadata = json.loads((RUNS / 'benchmark.json').read_text())
    for name in built:
        source = ROOT / name / 'solution'
        output = RUNS / (name + '-benchmark.out.bin')
        profile = RUNS / (name + '-profile')
        application = (str(source / 'build-npu/bmmms_npu_runner') + ' ' +
                       str(RUNS / 'benchmark.bin') + ' ' + str(output) + ' --benchmark')
        if not run(name + '-profile', ['timeout', '600', 'msprof', '--output=' + str(profile),
                                      '--application=' + application], source):
            continue
        verify(name + '-benchmark', RUNS / 'benchmark', output)
        files = list(profile.rglob('op_summary*.csv'))
        if len(files) != 1:
            raise RuntimeError('ambiguous profile for ' + name)
        if name == 'baseline':
            durations = vector_samples(files[0], len(metadata))
            values = [dict(interval_us=v, scores_us=v, finish_us=0, gap_us=0) for v in durations]
        else:
            values, _ = samples(files[0], len(metadata))
        result['profiles'][name] = [dict(spec=s, **v) for s, v in zip(metadata, values)]
        save()
    result['shape_summary'] = []
    for shape in dict.fromkeys(tuple(s['shape']) for s in metadata):
        summary = dict(shape=shape, variants={})
        for name, values in result['profiles'].items():
            selected = [v for v in values if tuple(v['spec']['shape']) == shape]
            summary['variants'][name] = {key: statistics.median(v[key] for v in selected)
                                         for key in ('interval_us', 'scores_us', 'finish_us', 'gap_us')}
        result['shape_summary'].append(summary)
    result['design_checkpoint_completed'] = True
    save()
    print(json.dumps(result['shape_summary'], indent=2), flush=True)
    print('LONG_K_DESIGN_CHECKPOINT_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        result['error'] = str(error)
        save()
        print('LONG_K_DESIGN_CHECKPOINT_ERROR', str(error), flush=True)
        raise
