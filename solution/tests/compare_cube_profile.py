"""Compare a sequential two-kernel probe with the checked Vector baseline.

Reject incomplete or ambiguous profiler traces rather than double-counting
mixed-kernel physical task records. Inspect the actual CSV before use.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

def samples(csv_path, count):
    with Path(csv_path).open() as source:
        raw = list(csv.DictReader(source))
    selected = []
    for row in raw:
        name = row['Op Name']
        phase = 'scores' if 'Scores' in name else 'finish' if 'Finish' in name else None
        if phase:
            start, duration = float(row['Task Start Time(us)']), float(row['Task Duration(us)'])
            if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
                raise ValueError('invalid kernel sample')
            selected.append(dict(phase=phase,start=start,duration=duration,task_type=row['Task Type']))
    selected.sort(key=lambda row: row['start'])
    if len(selected) != count*12*2:
        raise ValueError('unexpected task count; inspect physical mixed-task records')
    values=[]
    for i in range(0,len(selected),2):
        scores,finish = selected[i:i+2]
        if scores['phase'] != 'scores' or finish['phase'] != 'finish':
            raise ValueError('kernel order is not sequential Scores/Finish')
        gap = finish['start']-scores['start']-scores['duration']
        if gap < 0:
            raise ValueError('overlapping task intervals; inspect profiler semantics')
        values.append(dict(scores_us=scores['duration'],finish_us=finish['duration'],gap_us=gap,
                           duration_sum_us=scores['duration']+finish['duration'],
                           interval_us=finish['start']+finish['duration']-scores['start']))
    return [dict((key,statistics.median(r[key] for r in values[ci*12+2:(ci+1)*12]))
                 for key in values[0]) for ci in range(count)], sorted({r['task_type'] for r in selected})

def vector_samples(path, count):
    with Path(path).open() as source:
        raw = list(csv.DictReader(source))
    rows = [r for r in raw if 'Baseline' in r['Op Name']]
    rows.sort(key=lambda r: float(r['Task Start Time(us)']))
    if len(rows) != count * 12 or any(r['Task Type'] != 'AI_VECTOR_CORE' for r in rows):
        raise ValueError('incomplete or ambiguous Vector profile')
    if any(not math.isfinite(float(r['Task Start Time(us)'])) for r in rows):
        raise ValueError('invalid Vector start time')
    durations = [float(r['Task Duration(us)']) for r in rows]
    if any(not math.isfinite(v) or v <= 0 for v in durations):
        raise ValueError('invalid Vector duration')
    return [statistics.median(durations[i*12+2:(i+1)*12]) for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', help='shared input/golden/metadata prefix')
    parser.add_argument('vector_csv')
    parser.add_argument('cube_csv')
    parser.add_argument('regression_json', help='passed run_npu_checks.py report')
    parser.add_argument('output')
    args = parser.parse_args()
    import hashlib
    prefix = Path(args.prefix)
    metadata = json.loads(prefix.with_suffix('.json').read_text())
    regression = json.loads(Path(args.regression_json).read_text())
    digest = hashlib.sha256(prefix.with_suffix('.bin').read_bytes()).hexdigest()
    if not regression.get('workflow_passed') or digest != regression['large_input_sha256']:
        raise ValueError('candidate regression failed or input bytes differ')
    for name in ('large-ordinary','large-capture-cold','large-capture-chain',
                 'large-capture-streams','large-cold-shared','large-benchmark'):
        report = regression['reports'][name]
        if report['cases'] != len(metadata) or report['passed'] != len(metadata) or report['failures']:
            raise ValueError('incomplete large validation')
    before = vector_samples(args.vector_csv, len(metadata))
    after, task_types = samples(args.cube_csv, len(metadata))
    rows = [dict(spec=spec, vector_kernel_us=old, **new,
                 cube_over_vector=new['interval_us']/old)
            for spec, old, new in zip(metadata, before, after)]
    result = dict(kernel_sha256=regression['kernel_sha256'], input_sha256=digest,
                  metric='median Scores-start to Finish-end interval after 2 of 12 warmups',
                  task_types=task_types, cases=len(rows), rows=rows,
                  median_cube_over_vector=statistics.median(r['cube_over_vector'] for r in rows))
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    for shape in dict.fromkeys(tuple(r['spec']['shape']) for r in rows):
        selected = [r for r in rows if tuple(r['spec']['shape']) == shape]
        print(shape, {k: statistics.median(r[k] for r in selected) for k in
                      ('vector_kernel_us','scores_us','finish_us','gap_us','interval_us','cube_over_vector')})


if __name__ == '__main__':
    main()
