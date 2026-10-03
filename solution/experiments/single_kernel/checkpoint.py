#!/usr/bin/env python3
"""Single-launch checkpoint; stage has candidate/solution, baseline/solution, manifest.json."""
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
OLD = Path('/mnt/workspace/bmmms/long-k-candidate-20261003')
LONG = Path('/mnt/workspace/bmmms/long-k-grid-20261003/runs/benchmark')
SHORT = Path('/mnt/workspace/bmmms/perf-20261002/cache-block/solution/lab-data/large-short-k')
result = dict(manifest=json.loads((ROOT/'manifest.json').read_text()),steps=[],reports={},resources={},profiles={})


def save():
    (RUNS/'results.json').write_text(json.dumps(result,indent=2)+'\n')


def run(name, command, cwd=CANDIDATE):
    with (RUNS/(name+'.log')).open('w') as log:
        p=subprocess.run(list(map(str,command)),cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    result['steps'].append(dict(name=name,exit_code=p.returncode));save()
    print(name,p.returncode,flush=True)
    if p.returncode: raise RuntimeError(name+' failed; inspect log')


def verify(name,prefix,output):
    run(name+'-verify',['python3',CANDIDATE/'tests/npu_data.py','--prefix',prefix,'--verify',output])
    r=json.loads(output.with_suffix('.report.json').read_text())
    assert r['cases']==r['passed']==len(json.loads(prefix.with_suffix('.json').read_text())) and not r['failures']
    result['reports'][name]=r;save()


def profile(name, source, prefix, mode):
    output=RUNS/(name+'.out.bin');directory=RUNS/(name+'-profile')
    app=' '.join(map(str,[source/'build-npu/bmmms_npu_runner',prefix.with_suffix('.bin'),output,mode]))
    run(name,['timeout','600','msprof','--output='+str(directory),'--application='+app],source)
    verify(name,prefix,output)
    files=list(directory.rglob('op_summary*.csv'));assert len(files)==1
    return files[0]


def main():
    sys.path.insert(0,str(CANDIDATE/'tests'))
    from check_launch_profile import verify as launches,records
    from compare_cube_profile import samples
    for name,source in [('candidate',CANDIDATE),('baseline',BASELINE)]:
        assert hashlib.sha256((source/'kernel.asc').read_bytes()).hexdigest()==result['manifest'][name+'_sha256']
        run(name+'-configure',['cmake','-S','tests','-B','build-npu','-DNPU_ARCH=dav-2201',
            '-DBMMMS_TRACE_RUNTIME=ON','-DBMMMS_BUILD_SHARED_TEST='+('ON' if name=='candidate' else 'OFF')],source)
        run(name+'-build',['cmake','--build','build-npu','-j4'],source)
    for suite in ['long-cube','cube-finish','smoke']:
        prefix=OLD/'runs'/suite
        modes=[('ordinary',None),('cold','--capture-cold'),('chain','--capture-chain'),
               ('streams','--capture-streams'),('shared-cold','--capture-cold')]
        if suite=='smoke':modes=modes[:1]
        for label,mode in modes:
            name=suite+'-'+label;output=RUNS/(name+'.out.bin');shared=label=='shared-cold'
            runner='bmmms_shared_runner' if shared else 'bmmms_npu_runner'
            cmd=['timeout','300','env','BMMMS_SHARED_LIBRARY='+str(CANDIDATE/'build-npu/libbmmms_kernel.so'),
                 CANDIDATE/'build-npu'/runner,prefix.with_suffix('.bin'),output]
            if mode:cmd.append(mode)
            run(name,cmd);verify(name,prefix,output)
            if not shared:
                text=(RUNS/(name+'.log')).read_text()
                counts=re.findall(r'INTERNAL_SUMMARY allocations=(\d+) frees=(\d+) error=(\d+)',text)
                assert len(counts)==1
                a,f,e=map(int,counts[0]);calls={'cold':1,'chain':4,'streams':2,'ordinary':2}[label]
                expected=0 if suite=='smoke' else len(json.loads(prefix.with_suffix('.json').read_text()))*calls*2
                assert a==f==expected and e==0
                assert not re.search(r'^INTERNAL_.*ret=[1-9]',text,re.M)
                if label in ['cold','streams']:assert 'INTERNAL_SYNC' not in text
                result['resources'][name]=dict(allocations=a,frees=f,error=e);save()
    rule=RUNS/'launch-rule'
    run('generate-rule',['python3','tests/npu_data.py','--suite','launch-rule','--prefix',rule])
    metadata=json.loads(rule.with_suffix('.json').read_text());assert len(metadata)==15
    oldcsv=profile('rule-baseline',BASELINE,rule,'--profile-five');oldrows=records(oldcsv)
    assert len(oldrows)==105
    try:
        launches(oldcsv,metadata,5)
    except ValueError as error:
        rejection=str(error)
        assert 'expected 75, got 105' in rejection
    else:
        raise RuntimeError('single-launch check accepted the two-kernel baseline')
    newcsv=profile('rule-candidate',CANDIDATE,rule,'--profile-five');newrows=launches(newcsv,metadata,5)
    result['launch_rule']=dict(cases=15,repeats=5,expected=75,baseline_observed=len(oldrows),candidate_observed=len(newrows),passed=True,
        baseline_rejection=rejection,
        note='Own diagnostic inputs, not contest shapes; per-case dispatch and counts verified')
    save()
    result['shape_summary']=[]
    result['input_sha256']={}
    for suite,prefix,expected in [('long',LONG,'143cb57afe9d915c5a7eef121b13dbe348ae17c68d41ff853fdc4520676048b5'),
                                 ('short',SHORT,'55d9cca489f2f6ae9b1a59c3ed94a7c826a43a04fbe6573e7798d17bb1cac8ca')]:
        digest=hashlib.sha256(prefix.with_suffix('.bin').read_bytes()).hexdigest()
        assert digest==expected
        result['input_sha256'][suite]=digest
        metadata=json.loads(prefix.with_suffix('.json').read_text())
        oldcsv=profile(suite+'-baseline',BASELINE,prefix,'--benchmark')
        before,_=samples(oldcsv,len(metadata))
        newcsv=profile(suite+'-candidate',CANDIDATE,prefix,'--benchmark');newrows=launches(newcsv,metadata,12)
        after=[statistics.median(float(r['Task Duration(us)']) for r in newrows[i*12+2:(i+1)*12]) for i in range(len(metadata))]
        rows=[dict(spec=s,baseline_us=b['interval_us'],candidate_us=c,speedup=b['interval_us']/c) for s,b,c in zip(metadata,before,after)]
        result['profiles'][suite]=rows
        for shape in dict.fromkeys(tuple(s['shape']) for s in metadata):
            chosen=[r for r in rows if tuple(r['spec']['shape'])==shape]
            result['shape_summary'].append(dict(suite=suite,shape=shape,baseline_us=statistics.median(r['baseline_us'] for r in chosen),
                candidate_us=statistics.median(r['candidate_us'] for r in chosen),median_speedup=statistics.median(r['speedup'] for r in chosen)))
        save()
    result['workflow_passed']=True;save()
    print('SINGLE_KERNEL_CHECKPOINT_PASS',json.dumps(result['shape_summary']),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        result.update(workflow_passed=False,error=str(e));save();raise
