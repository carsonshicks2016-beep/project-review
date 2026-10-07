#!/usr/bin/env python3
"""Run the isolated open-disk IBAMR case with project resource guards."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def free_gib(path: Path) -> float: return shutil.disk_usage(path).free/1024**3
def rss_tree_kib(pid: int) -> int:
    p=subprocess.run(['ps','-axo','pid=,ppid=,rss='],text=True,capture_output=True,check=True)
    parents={}; rss={}
    for line in p.stdout.splitlines():
        a=line.split()
        if len(a)==3:
            q,r,s=map(int,a); parents[q]=r; rss[q]=s
    tree={pid}; changed=True
    while changed:
        changed=False
        for q,r in parents.items():
            if r in tree and q not in tree: tree.add(q); changed=True
    return sum(rss.get(q,0) for q in tree)
def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--binary',type=Path,required=True); ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True); ap.add_argument('--mpi-ranks',type=int,default=1)
    ap.add_argument('--rss-cap-gib',type=float,default=8); ap.add_argument('--disk-reserve-gib',type=float,default=40)
    a=ap.parse_args(); binary=a.binary.resolve(); inp=a.input.resolve(); out=a.run_dir.resolve()
    if not binary.is_file() or not inp.is_file(): ap.error('binary and input must exist')
    if not 1<=a.mpi_ranks<=8: ap.error('MPI rank limit is 8')
    if not out.is_dir() or any(out.iterdir()): ap.error('run directory must exist and be empty')
    if free_gib(out)<a.disk_reserve_gib: ap.error('disk reserve preflight failed')
    local_bin=out/'open_disk'; local_in=out/'open_disk.input'; shutil.copy2(binary,local_bin); shutil.copy2(inp,local_in)
    rank_cmd=['mpirun','-n',str(a.mpi_ranks),str(local_bin),str(local_in)] if a.mpi_ranks>1 else [str(local_bin),str(local_in)]
    stdout=out/'stdout.log'; stderr=out/'stderr.log'; rec=out/'run_record.json'
    started=datetime.datetime.now(datetime.timezone.utc); free0=free_gib(out)
    record={'study_id':out.name,'status':'RUNNING','started_utc':started.isoformat(),'command':rank_cmd,
      'working_directory':str(out),'mpi_ranks':a.mpi_ranks,'rss_cap_gib':a.rss_cap_gib,
      'minimum_free_disk_gib':a.disk_reserve_gib,'free_disk_gib_before':free0,
      'binary_sha256':sha(local_bin),'input_sha256':sha(local_in),'stdout_file':stdout.name,'stderr_file':stderr.name}
    rec.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    peak=0; minfree=free0; stop=None
    with stdout.open('wb') as so, stderr.open('wb') as se:
      proc=subprocess.Popen(rank_cmd,cwd=out,stdout=so,stderr=se,start_new_session=True,env=os.environ.copy())
      while proc.poll() is None:
        time.sleep(1); cur=rss_tree_kib(proc.pid); peak=max(peak,cur); free=free_gib(out); minfree=min(minfree,free)
        if cur>a.rss_cap_gib*1024**2: stop=f'RSS cap exceeded: {cur/1024**2:.3f} GiB'
        elif free<a.disk_reserve_gib: stop=f'disk reserve breached: {free:.3f} GiB free'
        if stop:
          os.killpg(proc.pid,signal.SIGTERM)
          try: proc.wait(timeout=15)
          except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
          break
      rc=proc.wait()
    ended=datetime.datetime.now(datetime.timezone.utc)
    record.update({'finished_utc':ended.isoformat(),'elapsed_seconds':(ended-started).total_seconds(),
      'peak_process_tree_rss_gib':peak/1024**2,'minimum_observed_free_disk_gib':minfree,
      'free_disk_gib_after':free_gib(out),'returncode':rc,'stop_reason':stop,
      'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr),
      'status':'STOPPED_RESOURCE_GUARD' if stop else ('COMPLETED' if rc==0 else 'FAILED')})
    rec.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n'); print(json.dumps(record,indent=2,sort_keys=True))
    (out/'resource_record.json').write_text(json.dumps({
      'status':record['status'],'exit_code':rc,'elapsed_seconds':record['elapsed_seconds'],
      'peak_process_tree_rss_gb':record['peak_process_tree_rss_gib'],
      'free_disk_bytes_before':int(record['free_disk_gib_before']*1024**3),
      'free_disk_bytes_after':int(record['free_disk_gib_after']*1024**3),
      'minimum_observed_free_disk_bytes':int(record['minimum_observed_free_disk_gib']*1024**3),
      'mpi_ranks':a.mpi_ranks,'stop_reason':stop,'runner':'open_disk_dg_iim/tools/run_guarded.py'
    },indent=2,sort_keys=True)+'\n')
    (out/'run_manifest.json').write_text(json.dumps({
      'study':out.parent.name,'case':out.name,'status':record['status'],
      'interpretation':'open-sheet IIM feasibility diagnostic only; no validation claim',
      'input_sha256':record['input_sha256'],'executable_sha256':record['binary_sha256'],
      'command':rank_cmd,'working_directory':str(out),'software':'IBAMR 0.19.0 + libMesh 1.7.8'
    },indent=2,sort_keys=True)+'\n')
    return 0 if rc==0 and not stop else 1
if __name__=='__main__': raise SystemExit(main())
