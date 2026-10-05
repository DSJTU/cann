#!/usr/bin/env python3
"""Checked, alternating paired NPU profiling of two successfully built runners."""
import argparse
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True

from case_catalog import BENCHMARK_SUITES
from npu_data import generate, sha256, verify
from summarize_benchmark import compare_rounds, profile_samples, timings


def build_identity(binary):
    identity = json.loads(Path(str(binary) + '.build.json').read_text())
    if sha256(binary) != identity['binary_sha256']:
        raise ValueError(f'binary does not match successful build record: {binary}')
    for key in ('kernel_sha256', 'runner_sha256', 'build_spec_sha256', 'compiler_sha256'):
        if not re.fullmatch(r'[a-f0-9]{64}', identity[key]):
            raise ValueError('invalid build identity: ' + key)
    return identity


def checked_log(path, specs):
    text = Path(path).read_text()
    devices = re.findall(r'^Device: (\S+), availableCoreNum=(\d+)$', text, re.M)
    if len(devices) != 1:
        raise ValueError('missing or ambiguous device identity')
    device, cores = devices[0]
    observed = re.findall(r'^CASE (\d+) repeat=(\d+) B=(\d+) M=(\d+) N=(\d+) K=(\d+) '
                          r'dtype=(\d+) ta=(\d+) tb=(\d+) cores=(\d+) wall_us=[\d.]+$', text, re.M)
    expected = [(str(i), str(repeat), *map(str, spec['shape']), str(spec['dtype']),
                 str(int(spec['ta'])), str(int(spec['tb'])),
                 str(min(spec.get('cores', 0) or int(cores), int(cores))))
                for i, spec in enumerate(specs) for repeat in range(12)]
    if observed != expected:
        raise ValueError('runner did not execute the expected cases/repeats/core counts')
    timings(path, len(specs))  # Event completeness, not the screening metric.
    return dict(soc=device, available_cores=int(cores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--suite', choices=BENCHMARK_SUITES, default='benchmark-tune')
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--runs', type=Path, required=True, help='new directory, never reuse old profiles')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive; at least four for a screening decision')
    binaries = dict(baseline=args.baseline.resolve(), candidate=args.candidate.resolve())
    identities = {side: build_identity(binary) for side, binary in binaries.items()}
    for key in ('runner_sha256', 'build_spec_sha256', 'compiler_sha256', 'arch', 'flags'):
        if identities['baseline'][key] != identities['candidate'][key]:
            parser.error('different build environment/protocol: ' + key)
    if identities['baseline']['runner_sha256'] != sha256(Path(__file__).with_name('npu_runner.asc')):
        parser.error('runner record differs from the current benchmark protocol')
    if identities['baseline']['build_spec_sha256'] != sha256(Path(__file__).with_name('CMakeLists.txt')):
        parser.error('build configuration differs from the current benchmark protocol')
    runs = args.runs.resolve()
    runs.mkdir(parents=True, exist_ok=False)
    result = dict(workflow_passed=False, suite=args.suite, seed=args.seed,
                  builds=identities, executions=[], device=None)

    def save():
        (runs/'results.json').write_text(json.dumps(result, indent=2) + '\n')

    save()
    try:
        prefix = runs/'input'
        generate(args.suite, prefix, args.seed)
        specs = json.loads(prefix.with_suffix('.json').read_text())
        result['data'] = json.loads(prefix.with_suffix('.manifest.json').read_text())
        paired = []
        for round_index in range(args.rounds):
            pair = {}
            # AB, BA, AB, BA counterbalances process-level thermal/clock drift.
            order = ('baseline', 'candidate') if round_index % 2 == 0 else ('candidate', 'baseline')
            for side in order:
                directory = runs/f'round-{round_index}-{side}'
                directory.mkdir()
                output, log = directory/'output.bin', directory/'runner.log'
                if sha256(binaries[side]) != identities[side]['binary_sha256']:
                    raise ValueError('runner changed during comparison')
                command = ['msprof', '--output='+str(directory/'profile'), '--application='+shlex.join(
                    [str(binaries[side]), str(prefix.with_suffix('.bin')), str(output), '--benchmark'])]
                with log.open('w') as stream:
                    completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
                execution = dict(round=round_index, side=side, exit_code=completed.returncode,
                                 directory=str(directory.relative_to(runs)))
                result['executions'].append(execution)
                save()
                if completed.returncode:
                    raise RuntimeError(f'{side} round {round_index} failed; see {log}')
                device = checked_log(log, specs)
                if result['device'] is not None and device != result['device']:
                    raise ValueError('device/core count changed between runs')
                result['device'] = device
                verification = verify(prefix, output, quiet=True)
                samples = profile_samples(directory/'profile', len(specs))
                pair[side] = {i: statistics.median(v) for i, v in samples.items()}
                execution.update(verification=verification, profile_launches=len(specs)*12,
                                 log_sha256=sha256(log), kernel_samples_us=samples)
                save()
                print(f'round {round_index} {side}: numerical, repeatability, guards and single-kernel profile PASS', flush=True)
            paired.append((pair['baseline'], pair['candidate']))
        result['comparison'] = compare_rounds(specs, paired)
        result['workflow_passed'] = True
        save()
        comparison = result['comparison']
        print(f"{comparison['screening_status']}: balanced candidate/baseline="
              f"{comparison['balanced_candidate_over_baseline']:.4f}; "
              f"95% round-noise interval={comparison['balanced_ratio_bootstrap_95pct']}")
        print(f"regressed cases (>3%): {comparison['regressed_cases']}; "
              f"regressed families: {comparison['family_regressions']}")
        print('Synthetic screening only. Check a completed candidate on benchmark-holdout; contest score decides the winner.')
    except (Exception, SystemExit) as error:
        result['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    main()
