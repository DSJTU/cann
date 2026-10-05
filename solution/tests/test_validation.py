#!/usr/bin/env python3
"""Test the validation tools against misleading, corrupt and incomplete evidence.

No NPU is needed. Synthetic logs below test acceptance/rejection of evidence;
they never establish a device correctness or performance result.
"""
import contextlib
import io
import json
import re
import shlex
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.dont_write_bytecode = True

import numpy as np

from case_catalog import GEOMETRIES, benchmark_specs, input_seed, validate_shape
from check_launch_profile import verify as verify_launches
import npu_data
import run_benchmark
from run_benchmark import build_identity, checked_log
from summarize_benchmark import compare_rounds, profile_samples, timings
from test_cpu import make_inputs, quantize


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_geometry_disjointness_and_coverage(self):
        before, after = [list(benchmark_specs(s)) for s in ('benchmark-tune', 'benchmark-holdout')]
        a, b = [{spec[0] for spec in values} for values in (before, after)]
        self.assertFalse(a & b)
        self.assertEqual((len(a), len(b), len(before), len(after)), (36, 36, 288, 288))
        self.assertEqual(len(GEOMETRIES), 9)
        for spec in before + after:
            validate_shape(spec[0])
        for shape in a | b:
            variants = {(s[1], s[2], s[3]) for s in before + after if s[0] == shape}
            self.assertEqual(len(variants), 8)

    def test_illegal_or_unknown_tests_rejected(self):
        for shape in ((1, 1, 1, 33), (65, 1, 1, 32), (64, 8192, 1, 8192)):
            with self.assertRaises(ValueError):
                validate_shape(shape)
        with self.assertRaises(ValueError):
            list(npu_data.specs('benhcmark'))

    def test_golden_quantization_and_reduction_order(self):
        shape = (3, 3, 2, 32)
        a, b = make_inputs(shape, 'row-winners', np.random.default_rng(1))
        for dtype in (1, 2):
            _, a64 = quantize(a, dtype)
            _, b64 = quantize(b, dtype)
            golden = npu_data.golden_rows(a64, b64)
            np.testing.assert_array_equal(golden, [3, 3, 3])
            # Explicit scalar oracle; column-sum-then-max would incorrectly give 2.
            scalar = [sum(max(sum(a64[bi, m, k]*b64[bi, k, n] for k in range(32))
                              for n in range(2)) for m in range(3)) for bi in range(3)]
            np.testing.assert_array_equal(golden, scalar)

    def test_batch_and_tail_oracles(self):
        for mode in ('batch-distinct', 'last-max'):
            a, b = make_inputs((64, 3, 9, 40), mode, np.random.default_rng(1))
            np.testing.assert_array_equal(npu_data.golden_rows(a, b), -3*np.arange(1, 65))

    def test_stable_inputs_across_order_layout_and_seeds(self):
        definitions = [((1, 3, 7, 40), 1, False, False, 0, 'random'),
                       ((2, 5, 9, 56), 2, True, True, 0, 'random')]
        with patch.object(npu_data, 'specs', return_value=iter(definitions)), contextlib.redirect_stdout(io.StringIO()):
            npu_data.generate('smoke', self.root/'a', 123)
        with patch.object(npu_data, 'specs', return_value=iter(reversed(definitions))), contextlib.redirect_stdout(io.StringIO()):
            npu_data.generate('smoke', self.root/'b', 123)
        aa = np.fromfile(self.root/'a.golden.bin', np.float32)
        bb = np.fromfile(self.root/'b.golden.bin', np.float32)
        np.testing.assert_array_equal(aa, np.concatenate((bb[2:], bb[:2])))
        self.assertEqual(input_seed(definitions[0][0], 'random', 123),
                         json.loads((self.root/'a.json').read_text())[0]['input_seed'])
        self.assertNotEqual(input_seed(definitions[0][0], 'random', 123), input_seed(definitions[0][0], 'random', 124))
        # Full generator checks same logical values under all four storage layouts.
        variants = [(definitions[0][0], 1, ta, tb, 0, 'random') for ta in (False, True) for tb in (False, True)]
        with patch.object(npu_data, 'specs', return_value=iter(variants)), contextlib.redirect_stdout(io.StringIO()):
            npu_data.generate('smoke', self.root/'layouts', 123)
        values = np.fromfile(self.root/'layouts.golden.bin', np.float32)
        np.testing.assert_array_equal(values, np.repeat(values[0], 4))

    def data(self):
        definition = ((1, 3, 2, 32), 1, False, False, 0, 'row-winners')
        with patch.object(npu_data, 'specs', return_value=iter([definition])), contextlib.redirect_stdout(io.StringIO()):
            npu_data.generate('smoke', self.root/'input')
        (self.root/'output.bin').write_bytes((self.root/'input.golden.bin').read_bytes())
        return self.root/'input', self.root/'output.bin'

    def test_data_tamper_rejected(self):
        prefix, output = self.data()
        npu_data.verify(prefix, output, quiet=True)
        (self.root/'input.bin').write_bytes((self.root/'input.bin').read_bytes() + b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            npu_data.verify(prefix, output, quiet=True)

    def test_nan_wrong_and_incomplete_output_rejected(self):
        prefix, output = self.data()
        for value in (np.nan, 2.0):
            np.array([value], np.float32).tofile(output)
            with self.assertRaises(SystemExit):
                npu_data.verify(prefix, output, quiet=True)
        output.write_bytes(b'')
        with self.assertRaisesRegex(RuntimeError, 'incomplete output'):
            npu_data.verify(prefix, output, quiet=True)

    def profile(self, durations):
        path = self.root/'op_summary.csv'
        path.write_text('Op Name,Task Start Time(us),Task Duration(us)\n' + ''.join(
            f'bmmms_dot_kernel,{i*100},{v}\n' for i, v in enumerate(durations)))
        return self.root

    def test_profile_warmup_and_launch_count(self):
        directory = self.profile([999, 999] + [2]*10)
        self.assertEqual(profile_samples(directory, 1)[0], [2]*10)
        self.profile([2]*13)
        with self.assertRaisesRegex(ValueError, 'single-kernel launches'):
            profile_samples(directory, 1)

    def test_legal_dispatch_change_does_not_freeze_reference_threshold(self):
        self.profile([2]*5)
        # This tiny geometry normally uses static dot; choosing another single
        # entry is legal when independently verified numerical output passes.
        path = self.root/'op_summary.csv'
        path.write_text(path.read_text().replace('bmmms_dot_kernel', 'fused_kernel'))
        rows = verify_launches(path, [dict(shape=[1, 1, 1, 32])], 5)
        self.assertEqual(len(rows), 5)

    def test_profile_duplicate_csv_and_nonfinite_rejected(self):
        directory = self.profile([2]*12)
        (directory/'op_summary_old.csv').write_text((directory/'op_summary.csv').read_text())
        with self.assertRaises(ValueError):
            profile_samples(directory, 1)
        (directory/'op_summary_old.csv').unlink()
        self.profile([float('nan')]*12)
        with self.assertRaises(ValueError):
            profile_samples(directory, 1)

    def test_incomplete_or_duplicate_event_samples_rejected(self):
        log = self.root/'log'
        lines = [f'EVENT case=0 repeat={i} event_us=2.0\n' for i in range(2, 12)]
        log.write_text(''.join(lines))
        self.assertEqual(timings(log, 1), {0: 2.0})
        for bad in (lines[:-1], lines + lines[:1]):
            log.write_text(''.join(bad))
            with self.assertRaises(ValueError):
                timings(log, 1)

    def test_median_hides_regression_but_report_exposes_it(self):
        specs = [dict(shape=[1, 17+i, 19, 40], dtype=1, ta=False, tb=False,
                      family='small' if i < 3 else 'large') for i in range(4)]
        before, after = dict.fromkeys(range(4), 1), {0: 0.9, 1: 0.9, 2: 0.9, 3: 2}
        report = compare_rounds(specs, [(before, after)]*4)
        self.assertGreater(report['balanced_candidate_over_baseline'], 1)
        self.assertEqual(report['regressed_cases'], 1)
        self.assertIn('family:large', report['family_regressions'])
        self.assertEqual(report['worst_regressions'][0]['kernel_candidate_over_baseline'], 2)

    def test_insufficient_rounds_and_noise_are_not_improvements(self):
        specs = [dict(shape=[1, 17, 19, 40], dtype=1, ta=False, tb=False, family='small')]
        self.assertEqual(compare_rounds(specs, [({0: 1}, {0: .8})])['screening_status'], 'insufficient_rounds')
        report = compare_rounds(specs, [({0: 1}, {0: v}) for v in (.8, 1.2, .8, 1.2)])
        self.assertEqual(report['screening_status'], 'inconclusive_noise')
        self.assertEqual(compare_rounds(specs, [({0: 1}, {0: .8})]*4)['screening_status'], 'promising_on_this_corpus')

    def test_build_record_binds_binary_and_escapes_flags(self):
        binary, compiler = self.root/'binary', self.root/'compiler'
        binary.write_bytes(b'new binary')
        compiler.write_bytes(b'compiler')
        subprocess.run(['cmake', f'-DKERNEL={binary}', f'-DRUNNER={binary}', f'-DBINARY={binary}',
                        f'-DCOMPILER={compiler}', '-DARCH=dav-2201', '-DFLAGS=-DNAME="test"', '-P',
                        str(Path(__file__).with_name('write_build_manifest.cmake'))], check=True)
        self.assertEqual(build_identity(binary)['flags'], '-DNAME="test"')
        binary.write_bytes(b'old binary accidentally copied in')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            build_identity(binary)

    def test_post_link_record_and_failed_rebuild(self):
        project, build = self.root/'project', self.root/'build'
        project.mkdir()
        (self.root/'kernel.asc').write_text('// fixture kernel\n')
        (project/'npu_runner.asc').write_text('// fixture runner\n')
        helper = Path(__file__).with_name('write_build_manifest.cmake')
        (project/helper.name).write_bytes(helper.read_bytes())
        source = Path(__file__).with_name('CMakeLists.txt').read_text()
        function = re.search(r'function\(record_validation_build target\).*?endfunction\(\)', source, re.S).group()
        (project/'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.16)\nproject(record_check LANGUAGES CXX)\n'
            'set(CMAKE_ASC_COMPILER "${CMAKE_CXX_COMPILER}")\n'
            'set(CMAKE_ASC_FLAGS "a|b\\\"c")\nset(NPU_ARCH dav-2201)\n'
            'add_executable(fixture main.cpp)\n' + function + '\nrecord_validation_build(fixture)\n')
        (project/'main.cpp').write_text('int main() { return 0; }\n')
        for command in (['cmake', '-S', str(project), '-B', str(build)], ['cmake', '--build', str(build)]):
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        binary = build/'fixture'
        original = Path(str(binary)+'.build.json').read_bytes()
        self.assertIn('a|b"c', build_identity(binary)['flags'])
        # A failed rebuild must not label the earlier binary as the new source.
        (project/'main.cpp').write_text('#error intentionally failed build\n')
        completed = subprocess.run(['cmake', '--build', str(build)], capture_output=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(Path(str(binary)+'.build.json').read_bytes(), original)

    def test_runner_log_shape_or_repeat_mismatch_rejected(self):
        specs = [dict(shape=[1, 1, 1, 72], dtype=1, ta=False, tb=False, cores=0)]
        content = 'Device: Ascend910_9362, availableCoreNum=20\n' + ''.join(
            f'CASE 0 repeat={i} B=1 M=1 N=1 K=72 dtype=1 ta=0 tb=0 cores=20 wall_us=10.000\n'
            for i in range(12)) + ''.join(f'EVENT case=0 repeat={i} event_us=2.000\n' for i in range(2, 12))
        log = self.root/'log'
        log.write_text(content)
        self.assertEqual(checked_log(log, specs)['available_cores'], 20)
        log.write_text(content.replace('K=72', 'K=80'))
        with self.assertRaisesRegex(ValueError, 'expected cases'):
            checked_log(log, specs)

    def test_benchmark_alternation_and_failure_evidence(self):
        # Exercise the entire coordinator while replacing only the device process.
        before, after = self.root/'baseline', self.root/'candidate'
        for binary in (before, after):
            binary.write_bytes(binary.name.encode())
            identity = dict(binary_sha256=npu_data.sha256(binary), kernel_sha256='a'*64,
                            runner_sha256=npu_data.sha256(Path(__file__).with_name('npu_runner.asc')),
                            compiler_sha256='b'*64, arch='dav-2201', flags='same')
            Path(str(binary)+'.build.json').write_text(json.dumps(identity))
        seen = []

        def tiny_generate(suite, prefix, seed):
            spec = ((1, 1, 1, 72), 1, False, False, 0, 'random')
            with patch.object(npu_data, 'specs', return_value=iter([spec])):
                npu_data.generate(suite, prefix, seed)

        def fake_device(command, stdout, stderr, timeout):
            application = shlex.split(next(c.split('=', 1)[1] for c in command if c.startswith('--application=')))
            binary, source, output, _ = application
            side = Path(binary).name
            seen.append(side)
            Path(output).write_bytes(Path(source).with_suffix('.golden.bin').read_bytes())
            stdout.write('Device: Ascend910_9362, availableCoreNum=20\n' + ''.join(
                f'CASE 0 repeat={i} B=1 M=1 N=1 K=72 dtype=1 ta=0 tb=0 cores=20 wall_us=10.000\n'
                for i in range(12)) + ''.join(f'EVENT case=0 repeat={i} event_us=2.000\n' for i in range(2, 12)))
            directory = Path(next(c.split('=', 1)[1] for c in command if c.startswith('--output=')))
            directory.mkdir()
            value = 5 if side == 'baseline' else 4
            (directory/'op_summary.csv').write_text('Op Name,Task Start Time(us),Task Duration(us)\n' + ''.join(
                f'bmmms_dot_kernel,{i*100},{value}\n' for i in range(12)))
            return subprocess.CompletedProcess(command, 0)

        def execute(directory, device):
            argv = ['run_benchmark.py', '--baseline', str(before), '--candidate', str(after),
                    '--rounds', '4', '--runs', str(directory)]
            with patch.object(sys, 'argv', argv), patch.object(run_benchmark, 'generate', tiny_generate), \
                    patch.object(run_benchmark.subprocess, 'run', side_effect=device), contextlib.redirect_stdout(io.StringIO()):
                run_benchmark.main()

        execute(self.root/'success', fake_device)
        self.assertEqual(seen, ['baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate', 'candidate', 'baseline'])
        report = json.loads((self.root/'success/results.json').read_text())
        self.assertTrue(report['workflow_passed'])
        self.assertAlmostEqual(report['comparison']['balanced_candidate_over_baseline'], .8)
        self.assertEqual(len(report['executions']), 8)

        def corrupt_device(command, **kwargs):
            completed = fake_device(command, **kwargs)
            output = shlex.split(next(c.split('=', 1)[1] for c in command if c.startswith('--application=')))[2]
            np.array([np.nan], np.float32).tofile(output)
            return completed

        with self.assertRaises(SystemExit):
            execute(self.root/'failure', corrupt_device)
        failed = json.loads((self.root/'failure/results.json').read_text())
        self.assertFalse(failed['workflow_passed'])
        self.assertNotIn('comparison', failed)
        self.assertIn('error', failed)


if __name__ == '__main__':
    unittest.main()
