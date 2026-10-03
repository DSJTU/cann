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
counts = {'Baseline': 0, 'Fused': 0}

def launch(match):
    name, specialization = match.groups()
    counts[name] += 1
    mode = 'AIV_MODE' if name == 'Baseline' else 'MIX_MODE'
    return f'TwinLaunch(KernelMode::{mode}, "{name}", ({name}<{specialization}>), blocks, '

text = re.sub(r'(Baseline|Fused)<([^<>]+)><<<blocks, nullptr, stream>>>\(', launch, text)
if counts != {'Baseline': 8, 'Fused': 2} or '<<<' in text or '__mix__(' in text or '__schedmode__(' in text:
    raise RuntimeError('kernel launch structure changed; review the CPU Twin adapter')
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(f'#line 1 "{source.resolve()}"\n' + text)
destination.with_suffix('.json').write_text(json.dumps({
    'kernel_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'launches_adapted': counts,
    'scope': 'Official CPU Twin device bodies; Host allocations/launches adapted, no graph capture or NPU timing',
}, indent=2) + '\n')
