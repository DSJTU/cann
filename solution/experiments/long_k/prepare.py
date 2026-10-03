#!/usr/bin/env python3
"""Materialize isolated full-K / compensated-K probes from the actual kernel.

The generated probes force Cube dispatch, including small precision cases.
They are experiments, not submission kernels. Runtime ownership is preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = 'fc8c8fcbe23ae195e7debb780399c44b4afbc6e4'


def reference(path):
    # Pin both kernel and model interfaces so later production changes do not
    # silently change the experiment or invalidate its recorded source hashes.
    return subprocess.check_output(['git', 'show', f'{BASE_COMMIT}:{path}'], cwd=ROOT.parent)


def replace(source, before, after):
    if source.count(before) != 1:
        raise ValueError('source changed; inspect probe transformation: ' + before[:80])
    return source.replace(before, after)


def kernel(source, rows, chunk):
    source = replace(source, 'constexpr uint32_t rows = 32, columns = 64;',
                     f'constexpr uint32_t rows = {rows}, columns = 64;')
    source = replace(source, 'if (k > 128) throw std::runtime_error("Cube currently requires K <= 128");',
                     'if (k > 8192) throw std::runtime_error("Cube probe K out of range");')
    source = replace(source, 'if (k <= 128 && m >= 256 && n >= 256 && m * n * k >= (1u << 24))',
                     'if (k <= 8192 && m >= 1 && n >= 1) // Probe only: force Cube.')
    if not chunk:
        return source
    source = replace(source, 'api.SetShape(rows, columns, k);',
                     f'api.SetShape(rows, columns, k < {chunk} ? k : {chunk});')
    source = replace(source, 'pipe.InitBuffer(cBuf, rows * columns * sizeof(float));', '''pipe.InitBuffer(cBuf, rows * columns * sizeof(float));
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf, correctionBuf, nextBuf;
    pipe.InitBuffer(sumBuf, rows * columns * sizeof(float));
    pipe.InitBuffer(correctionBuf, rows * columns * sizeof(float));
    pipe.InitBuffer(nextBuf, rows * columns * sizeof(float));
    auto sums = sumBuf.Get<float>(), corrections = correctionBuf.Get<float>();
    auto next = nextBuf.Get<float>();''')
    begin = source.index('            mm.SetOrgShape(m, n, k);', source.index('namespace bmmms_cube'))
    end = source.index('                // Sequential ND output', begin)
    source = source[:begin] + f'''            const uint32_t count = bmmms::Align8(validM * validN);
            // Rounded copies are legal because all padding is initialized.
            AscendC::Duplicate(c, 0.0f, rows * columns);
            AscendC::Duplicate(sums, 0.0f, rows * columns);
            AscendC::Duplicate(corrections, 0.0f, rows * columns);
            AscendC::Duplicate(next, 0.0f, rows * columns);
            AscendC::PipeBarrier<PIPE_V>();
            Fence<AscendC::HardEvent::V_S>(pipe);
            for (uint32_t k0 = 0; k0 < k; k0 += {chunk}) {{
                const uint32_t validK = k - k0 < {chunk} ? k - k0 : {chunk};
                mm.SetOrgShape(m, n, k);
                mm.SetSingleShape(validM, validN, validK);
                mm.SetTensorA(a[bi * m * k + (ta ? k0 * m + m0 : m0 * k + k0)], ta);
                mm.SetTensorB(b[bi * n * k + (tb ? n0 * k + k0 : k0 * n + n0)], tb);
                while (mm.template Iterate<true>()) {{
                    mm.template GetTensorC<true>(c, 0, true);
                    AscendC::Sub(c, c, corrections, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(next, sums, c, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(corrections, next, sums, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(corrections, corrections, c, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::DataCopy(sums, next, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    // The Cube output and all Vector consumers finish before
                    // End or the next request reuses the same output buffer.
                    Fence<AscendC::HardEvent::V_S>(pipe);
                }}
                mm.End();
            }}
            {{
''' + source[end:]
    # Both reduction paths must consume the complete, compensated dot product.
    source = replace(source, 'AscendC::WholeReduceMax(partial, c, int32_t(validN)',
                     'AscendC::WholeReduceMax(partial, sums, int32_t(validN)')
    source = replace(source, 'AscendC::Gather(oneRow, c, offsets, row * validN * sizeof(float), validN);',
                     'AscendC::Gather(oneRow, sums, offsets, row * validN * sizeof(float), validN);')
    source = replace(source, '            mm.End();\n        }\n        Fence<AscendC::HardEvent::V_S>(pipe);',
                     '        }\n        Fence<AscendC::HardEvent::V_S>(pipe);')
    return source


def grid_kernel(source):
    """Each (batch, M tile, N tile) owns its row maxima; Finish merges N."""
    source = replace(source,
        'b.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2), batch * n * k);\n'
        '    maxima.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rowMax), batch * m);',
        'b.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2), batch * n * k);\n'
        '    maxima.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rowMax), batch * m * ((n + columns - 1) / columns));')
    source = replace(source, '    const uint32_t mTiles = (m + rows - 1) / rows;',
                     '    const uint32_t mTiles = (m + rows - 1) / rows;\n    const uint32_t nTiles = (n + columns - 1) / columns;')
    source = replace(source,
        'for (uint32_t task = AscendC::GetBlockIdx(); task < batch * mTiles; task += vectorWorkers) {\n'
        '        const uint32_t bi = task / mTiles, m0 = task % mTiles * rows;',
        'for (uint32_t task = AscendC::GetBlockIdx(); task < batch * mTiles * nTiles; task += vectorWorkers) {\n'
        '        const uint32_t bi = task / (mTiles * nTiles);\n'
        '        const uint32_t m0 = task / nTiles % mTiles * rows, ni = task % nTiles;')
    source = replace(source, 'for (uint32_t n0 = 0; n0 < n; n0 += columns) {',
                     'for (uint32_t n0 = ni * columns; n0 < n && n0 < (ni + 1) * columns; n0 += columns) {')
    source = replace(source, 'if (n0 == 0) AscendC::DataCopy(maximum, partial, validM * 8);\n'
                     '                else AscendC::Max(maximum, maximum, partial, validM * 8);',
                     'AscendC::DataCopy(maximum, partial, validM * 8);')
    source = replace(source, 'AscendC::DataCopyPad(maxima[bi * m + m0], packed, cp);',
                     'AscendC::DataCopyPad(maxima[(bi * nTiles + ni) * m + m0], packed, cp);')
    source = replace(source, 'void Finish(GM_ADDR rowMax, GM_ADDR y, uint32_t batch, uint32_t m) {',
                     'void Finish(GM_ADDR rowMax, GM_ADDR y, uint32_t batch, uint32_t m, uint32_t nTiles) {')
    source = replace(source, 'maxima.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rowMax), batch * m);',
                     'maxima.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rowMax), batch * m * nTiles);')
    source = replace(source,
        'AscendC::TBuf<AscendC::TPosition::VECIN> hiBuf;\n'
        '    AscendC::TBuf<AscendC::TPosition::VECCALC> loBuf, aBuf, bBuf, sBuf, bpBuf, errBuf, tmpBuf, offsetBuf;',
        'AscendC::TBuf<AscendC::TPosition::VECIN> hiBuf, loBuf;\n'
        '    AscendC::TBuf<AscendC::TPosition::VECCALC> aBuf, bBuf, sBuf, bpBuf, errBuf, tmpBuf, offsetBuf;')
    source = replace(source, '        AscendC::Duplicate(lo, 0.0f, padded);\n', '')
    source = replace(source, 'AscendC::DataCopyPad(hi, maxima[bi * m], cp, pad);',
                     'AscendC::DataCopyPad(hi, maxima[bi * nTiles * m], cp, pad);')
    source = replace(source,
        '        // Every level leaves zero padding, so an odd final child pairs with zero.',
        '''        for (uint32_t ni = 1; ni < nTiles; ++ni) {
            // Finish reading lo before the next DMA overwrites it.
            Fence<AscendC::HardEvent::V_S>(pipe);
            Fence<AscendC::HardEvent::S_MTE2>(pipe);
            AscendC::DataCopyPad(lo, maxima[(bi * nTiles + ni) * m], cp, pad);
            Fence<AscendC::HardEvent::MTE2_V>(pipe);
            AscendC::Max(hi, hi, lo, padded);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // Reuse the N-merge input buffer as the summation low component.
        AscendC::Duplicate(lo, 0.0f, padded);
        AscendC::PipeBarrier<PIPE_V>();
        // Every level leaves zero padding, so an odd final child pairs with zero.''')
    source = replace(source, 'size_t(batch) * m * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST',
                     'size_t(batch) * m * ((n + columns - 1) / columns) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST')
    source = replace(source, 'const uint32_t tasks = batch * ((m + rows - 1) / rows);',
                     'const uint32_t tasks = batch * ((m + rows - 1) / rows) * ((n + columns - 1) / columns);')
    source = replace(source, '((GM_ADDR)maxima, y, batch, m);',
                     '((GM_ADDR)maxima, y, batch, m, (n + columns - 1) / columns);')
    return source


def partition_kernel(source):
    """Bound N partitions by the number of useful resident workers."""
    source = replace(source, 'uint32_t batch, uint32_t m, uint32_t n, uint32_t k, bool ta, bool tb) {',
                     'uint32_t batch, uint32_t m, uint32_t n, uint32_t k, uint32_t nGroups, bool ta, bool tb) {')
    source = replace(source, 'batch * m * ((n + columns - 1) / columns));',
                     'batch * m * nGroups);')
    source = replace(source, '    const uint32_t nTiles = (n + columns - 1) / columns;',
                     '    const uint32_t nTiles = nGroups;\n'
                     '    const uint32_t groupTiles = ((n + columns - 1) / columns + nGroups - 1) / nGroups;')
    source = replace(source,
        'for (uint32_t n0 = ni * columns; n0 < n && n0 < (ni + 1) * columns; n0 += columns) {',
        'for (uint32_t n0 = ni * groupTiles * columns; n0 < n && n0 < (ni + 1) * groupTiles * columns; n0 += columns) {')
    source = replace(source, '                AscendC::DataCopy(maximum, partial, validM * 8);',
                     '                if (n0 == ni * groupTiles * columns) AscendC::DataCopy(maximum, partial, validM * 8);\n'
                     '                else AscendC::Max(maximum, maximum, partial, validM * 8);')
    source = replace(source, '    const uint32_t cores = availableCoreNum > 0 ? uint32_t(availableCoreNum) : 20;', '')
    source = replace(source, '    if (k > 8192) throw std::runtime_error("Cube probe K out of range");',
                     '''    const uint32_t cores = availableCoreNum > 0 ? uint32_t(availableCoreNum) : 20;
    const uint32_t mTasks = batch * ((m + rows - 1) / rows);
    const uint32_t columnTiles = (n + columns - 1) / columns;
    uint32_t nGroups = (2 * cores + mTasks - 1) / mTasks;
    if (nGroups > columnTiles) nGroups = columnTiles;
    const uint32_t groupTiles = (columnTiles + nGroups - 1) / nGroups;
    nGroups = (columnTiles + groupTiles - 1) / groupTiles;
    if (k > 8192) throw std::runtime_error("Cube probe K out of range");''')
    source = replace(source, 'size_t(batch) * m * ((n + columns - 1) / columns) * sizeof(float),',
                     'size_t(batch) * m * nGroups * sizeof(float),')
    source = replace(source, 'const uint32_t tasks = batch * ((m + rows - 1) / rows) * ((n + columns - 1) / columns);',
                     'const uint32_t tasks = mTasks * nGroups;')
    # Two typed Scores calls share the exact same arguments.
    before = 'tiling, batch, m, n, k, ta, tb);'
    if source.count(before) != 2:
        raise ValueError('unexpected Scores calls')
    source = source.replace(before, 'tiling, batch, m, n, k, nGroups, ta, tb);')
    source = replace(source, '((GM_ADDR)maxima, y, batch, m, (n + columns - 1) / columns);',
                     '((GM_ADDR)maxima, y, batch, m, nGroups);')
    return source


def prepare(destination, grid=False, partitioned=False):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('use a fresh experiment directory')
    destination.mkdir(parents=True)
    source = reference('solution/kernel.asc').decode()
    manifest = dict(base_commit=BASE_COMMIT, base_kernel_sha256=hashlib.sha256(source.encode()).hexdigest(), variants={})
    test_paths = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', BASE_COMMIT, '--', 'solution/tests'],
        cwd=ROOT.parent, text=True).splitlines()
    for name, rows, chunk in [('baseline', 32, None), ('full', 64, 0),
                              ('k512', 64, 512), ('k128', 64, 128)]:
        target = destination / name / 'solution'
        target.mkdir(parents=True)
        for path in test_paths:
            copied = target / Path(path).relative_to('solution')
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(reference(path))
        text = source if chunk is None else kernel(source, rows, chunk)
        if (grid or partitioned) and chunk is not None:
            text = grid_kernel(text)
        if partitioned and chunk is not None:
            text = partition_kernel(text)
        (target / 'kernel.asc').write_text(text)
        manifest['variants'][name] = dict(rows=rows, columns=64, k_chunk=chunk,
                                          grid=(grid or partitioned) and chunk is not None,
                                          partitioned=partitioned and chunk is not None,
                                          kernel_sha256=hashlib.sha256(text.encode()).hexdigest())
    shutil.copy2(Path(__file__).with_name('data.py'), destination / 'data.py')
    shutil.copy2(Path(__file__).with_name('checkpoint.py'), destination / 'checkpoint.py')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--partitioned', action='store_true')
    args = parser.parse_args()
    print(prepare(args.destination, args.grid, args.partitioned))
