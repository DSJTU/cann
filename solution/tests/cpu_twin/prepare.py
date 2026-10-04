#!/usr/bin/env python3
"""Adapt Host launches for the official C++ CPU Twin library; keep device bodies."""
import hashlib
import json
from pathlib import Path
import re
import sys

source, destination = map(Path, sys.argv[1:])
original = source.read_text()
text = re.sub(r'__schedmode__\(1\)\s*', '', original)
text = re.sub(r'__mix__\([01], [12]\)\s*', '', text)
counts = {'fused_kernel': 0, 'bmmms_small_kernel': 0, 'bmmms_dot_kernel': 0}

def launch(match):
    name, specialization, blocks = match.groups()
    counts[name] += 1
    mode = 'MIX_MODE' if name == 'fused_kernel' else 'AIV_MODE'
    return f'TwinLaunch(KernelMode::{mode}, "{name}", ({name}<{specialization}>), {blocks}, '

text = re.sub(r'(fused_kernel|bmmms_small_kernel|bmmms_dot_kernel)<([^<>]+)><<<([^,]+), nullptr, stream>>>\(', launch, text)
if counts != {'fused_kernel': 1, 'bmmms_small_kernel': 4, 'bmmms_dot_kernel': 2} or '<<<' in text or '__mix__(' in text or '__schedmode__(' in text:
    raise RuntimeError('kernel launch structure changed; review the CPU Twin adapter')
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(f'#line 1 "{source.resolve()}"\n' + text)
destination.with_suffix('.json').write_text(json.dumps({
    'kernel_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'launches_adapted': counts,
    'scope': 'Official CPU Twin device bodies; Host allocations/launches adapted, no graph capture or NPU timing',
}, indent=2) + '\n')
