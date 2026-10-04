#!/usr/bin/env python3
"""Compare checked stream-event intervals, excluding two warmup calls.

These intervals include device queue/submission effects; they are not the
judge's pure kernel timing. Both runs must use identical generated input data.
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

from check_launch_profile import records


def timings(path, count):
    samples = {}
    for case, repeat, value in re.findall(
        r'^EVENT case=(\d+) repeat=(\d+) event_us=([\d.]+)$',
        Path(path).read_text(), re.MULTILINE,
    ):
        case, repeat, value = int(case), int(repeat), float(value)
        if case >= count or repeat not in range(2, 12) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'invalid timing sample in {path}')
        if repeat in samples.setdefault(case, {}):
            raise ValueError(f'duplicate timing sample in {path}')
        samples[case][repeat] = value
    if set(samples) != set(range(count)) or any(len(v) != 10 for v in samples.values()):
        raise ValueError(f'incomplete timing samples in {path}')
    return {case: statistics.median(v.values()) for case, v in samples.items()}


def profile_timings(directory, count):
    files = list(Path(directory).rglob('op_summary*.csv'))
    if len(files) != 1:
        raise ValueError(f'expected one operator summary in {directory}')
    rows = records(files[0], allow_historical=True)
    if len(rows) != count * 12:
        raise ValueError(f'expected {count * 12} single-kernel launches in {directory}, got {len(rows)}')
    # Both sides may use different dispatch implementations, but each case
    # must execute one consistent kernel for its two warmups and ten samples.
    for i in range(count):
        names = {row['Op Name'] for row in rows[i * 12:(i + 1) * 12]}
        if len(names) != 1 or not any(kind in next(iter(names)) for kind in ('Baseline', 'Fused', 'fused_kernel', 'bmmms_small_kernel', 'bmmms_dot_kernel')):
            raise ValueError(f'unexpected kernel dispatch for case {i} in {directory}')
    values = [float(r['Task Duration(us)']) for r in rows]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f'invalid kernel duration in {directory}')
    return {i: statistics.median(values[i * 12 + 2:(i + 1) * 12]) for i in range(count)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata')
    parser.add_argument('baseline_log')
    parser.add_argument('candidate_log')
    parser.add_argument('output')
    parser.add_argument('--baseline-profile')
    parser.add_argument('--candidate-profile')
    args = parser.parse_args()
    specs = json.loads(Path(args.metadata).read_text())
    baseline = timings(args.baseline_log, len(specs))
    candidate = timings(args.candidate_log, len(specs))
    rows = [dict(spec=spec, baseline_event_us=baseline[i], candidate_event_us=candidate[i],
                 candidate_over_baseline=candidate[i] / baseline[i])
            for i, spec in enumerate(specs)]
    report = dict(metric='median of 10 stream-event intervals after 2 warmups', cases=len(rows),
                  median_candidate_over_baseline=statistics.median(r['candidate_over_baseline'] for r in rows),
                  rows=rows)
    if bool(args.baseline_profile) != bool(args.candidate_profile):
        parser.error('supply both profile directories')
    if args.baseline_profile:
        before = profile_timings(args.baseline_profile, len(specs))
        after = profile_timings(args.candidate_profile, len(specs))
        for i, row in enumerate(rows):
            row.update(baseline_kernel_us=before[i], candidate_kernel_us=after[i],
                       kernel_candidate_over_baseline=after[i] / before[i])
        report['median_kernel_candidate_over_baseline'] = statistics.median(after[i] / before[i] for i in before)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    for shape in dict.fromkeys(tuple(s['shape']) for s in specs):
        selected = [r for r in rows if tuple(r['spec']['shape']) == shape]
        print(f'{shape}: median candidate/baseline={statistics.median(r["candidate_over_baseline"] for r in selected):.3f}')
        if args.baseline_profile:
            print(f'  kernel ratio={statistics.median(r["kernel_candidate_over_baseline"] for r in selected):.3f}')
    print(f'Compared {len(rows)} cases; overall median ratio={report["median_candidate_over_baseline"]:.3f}')


if __name__ == '__main__':
    main()
