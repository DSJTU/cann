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
import sys
import random

sys.dont_write_bytecode = True

from check_launch_profile import records


def timings(path, count):
    samples = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith('EVENT '):
            continue
        match = re.fullmatch(r'EVENT case=(\d+) repeat=(\d+) event_us=([\d.]+)', line)
        if not match:
            raise ValueError(f'malformed timing sample in {path}: {line}')
        case, repeat, value = match.groups()
        case, repeat, value = int(case), int(repeat), float(value)
        if case >= count or repeat not in range(2, 12) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'invalid timing sample in {path}')
        if repeat in samples.setdefault(case, {}):
            raise ValueError(f'duplicate timing sample in {path}')
        samples[case][repeat] = value
    if set(samples) != set(range(count)) or any(len(v) != 10 for v in samples.values()):
        raise ValueError(f'incomplete timing samples in {path}')
    return {case: statistics.median(v.values()) for case, v in samples.items()}


def profile_samples(directory, count):
    files = list(Path(directory).rglob('op_summary*.csv'))
    if len(files) != 1:
        raise ValueError(f'expected one operator summary in {directory}')
    rows = records(files[0])
    if len(rows) != count * 12:
        raise ValueError(f'expected {count * 12} single-kernel launches in {directory}, got {len(rows)}')
    # Both sides may use different dispatch implementations, but each case
    # must execute one consistent kernel for its two warmups and ten samples.
    for i in range(count):
        names = {row['Op Name'] for row in rows[i * 12:(i + 1) * 12]}
        if len(names) != 1 or not any(kind in next(iter(names)) for kind in (
            'fused_kernel', 'bmmms_small_kernel', 'bmmms_dot_kernel',
            'bmmms_static_dot_kernel',
        )):
            raise ValueError(f'unexpected kernel dispatch for case {i} in {directory}')
    values = [float(r['Task Duration(us)']) for r in rows]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f'invalid kernel duration in {directory}')
    return {i: values[i * 12 + 2:(i + 1) * 12] for i in range(count)}


def profile_timings(directory, count):
    return {i: statistics.median(v) for i, v in profile_samples(directory, count).items()}


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def balanced_ratio(specs, ratios):
    # Eight dtype/layout variants do not count as eight different geometries.
    families = {}
    for i, spec in enumerate(specs):
        family = spec.get('family', 'diagnostic')
        families.setdefault(family, {}).setdefault(tuple(spec['shape']), []).append(ratios[i])
    return geomean(geomean(geomean(v) for v in shapes.values()) for shapes in families.values())


def compare_rounds(specs, rounds):
    """Paired whole-process rounds, not ten correlated repeats as n=10.

    rounds contains (baseline kernel medians, candidate kernel medians).
    Bootstrap describes run noise on this fixed corpus, not unknown shapes.
    """
    if not specs or not rounds:
        raise ValueError('empty comparison')
    ratios = []
    for before, after in rounds:
        if set(before) != set(range(len(specs))) or set(after) != set(before):
            raise ValueError('incomplete paired round')
        if any(not math.isfinite(v) or v <= 0 for v in list(before.values()) + list(after.values())):
            raise ValueError('invalid paired round duration')
        ratios.append({i: after[i] / before[i] for i in before})
    rows = []
    for i, spec in enumerate(specs):
        rows.append(dict(spec=spec,
                         baseline_kernel_us=statistics.median(r[0][i] for r in rounds),
                         candidate_kernel_us=statistics.median(r[1][i] for r in rounds),
                         kernel_candidate_over_baseline=geomean(r[i] for r in ratios),
                         paired_round_ratios=[r[i] for r in ratios]))
    aggregate = [balanced_ratio(specs, r) for r in ratios]
    ratio = geomean(aggregate)
    interval = None
    if len(rounds) >= 4:
        rng = random.Random(20261005)
        samples = sorted(geomean(rng.choices(aggregate, k=len(aggregate))) for _ in range(2000))
        interval = [samples[49], samples[1949]]
    groups = {}
    for label, key in (('family', lambda s: s.get('family', 'diagnostic')),
                       ('dtype', lambda s: str(s['dtype'])),
                       ('layout', lambda s: f"ta={int(s['ta'])},tb={int(s['tb'])}")):
        for value in dict.fromkeys(key(s) for s in specs):
            selected = [i for i, s in enumerate(specs) if key(s) == value]
            groups[f'{label}:{value}'] = dict(
                cases=len(selected), geometries=len({tuple(specs[i]['shape']) for i in selected}),
                candidate_over_baseline=geomean(rows[i]['kernel_candidate_over_baseline'] for i in selected))
    worst = sorted(rows, key=lambda r: r['kernel_candidate_over_baseline'], reverse=True)
    geometries = []
    for shape in dict.fromkeys(tuple(s['shape']) for s in specs):
        selected = [r for r in rows if tuple(r['spec']['shape']) == shape]
        geometries.append(dict(shape=shape, family=selected[0]['spec'].get('family', 'diagnostic'),
                               candidate_over_baseline=geomean(r['kernel_candidate_over_baseline'] for r in selected),
                               worst_variant_ratio=max(r['kernel_candidate_over_baseline'] for r in selected)))
    family_regressions = [name for name, g in groups.items()
                          if name.startswith('family:') and g['candidate_over_baseline'] > 1.03]
    status = 'insufficient_rounds'
    if interval:
        if interval[0] <= 1 <= interval[1]:
            status = 'inconclusive_noise'
        elif interval[0] > 1:
            status = 'slower_on_this_corpus'
        elif family_regressions or worst[0]['kernel_candidate_over_baseline'] > 1.10:
            status = 'mixed_tradeoff'
        else:
            status = 'promising_on_this_corpus'
    return dict(metric='msprof kernel duration; 2 warmups, median of 10 samples per paired round',
                cases=len(specs), geometries=len({tuple(s['shape']) for s in specs}), rounds=len(rounds),
                weighting='equal families / equal geometries / equal dtype-layout variants',
                balanced_candidate_over_baseline=ratio, balanced_speedup=1/ratio,
                balanced_ratio_bootstrap_95pct=interval,
                paired_round_balanced_ratios=aggregate, groups=groups,
                improved_cases=sum(r['kernel_candidate_over_baseline'] < 0.97 for r in rows),
                regressed_cases=sum(r['kernel_candidate_over_baseline'] > 1.03 for r in rows),
                family_regressions=family_regressions, worst_regressions=worst[:10],
                geometry_comparison=sorted(geometries, key=lambda r: r['candidate_over_baseline'], reverse=True),
                screening_status=status, rows=rows,
                limitation='Synthetic screening only; interval measures round noise, not hidden-case uncertainty. Contest score decides best version.')


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
                  performance_evidence='unverified single-round diagnostic; use run_benchmark.py for checked paired rounds',
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
        report['kernel_screening'] = compare_rounds(specs, [(before, after)])
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    for shape in dict.fromkeys(tuple(s['shape']) for s in specs):
        selected = [r for r in rows if tuple(r['spec']['shape']) == shape]
        print(f'{shape}: median candidate/baseline={statistics.median(r["candidate_over_baseline"] for r in selected):.3f}')
        if args.baseline_profile:
            print(f'  kernel ratio={statistics.median(r["kernel_candidate_over_baseline"] for r in selected):.3f}')
    print(f'Compared {len(rows)} cases; overall median ratio={report["median_candidate_over_baseline"]:.3f}')


if __name__ == '__main__':
    main()
