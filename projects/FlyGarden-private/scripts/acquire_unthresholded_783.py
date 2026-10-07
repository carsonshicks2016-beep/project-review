"""Durable, range-verified acquisition of the pinned authors' synapse archive."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/brain-integration/recovery/patchy-compartment-mapping-v2'
SIZE=9492998242
MD5='f8f1b97c9d4b0ea9b4c8b287f6b99091'
URL='https://zenodo.org/records/10676866/files/flywire_synapses_783.feather?download=1'
CHUNK=16*1024**2
RESERVE=2*1024**3


def hashfile(path,kind='sha256'):
    h=hashlib.new(kind)
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()


def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True);source=OUT/'source';source.mkdir(exist_ok=True)
    parts=source/'archive-ranges';parts.mkdir(exist_ok=True)
    final=source/'flywire_synapses_783.feather'
    if final.exists():
        if final.stat().st_size!=SIZE or hashfile(final,'md5')!=MD5:raise RuntimeError('Existing archive identity mismatch')
        # Recover a crash between publishing the archive and its metadata.
        atomic(OUT/'archive-identity.json',{'version':1,'url':URL,'bytes':SIZE,'md5':MD5,
               'sha256':hashfile(final),'published_checksum_verified':True,'ranges_preserved':True,
               'recovered_existing_archive':True})
        atomic(OUT/'acquisition-progress.json',{'status':'complete','archive_bytes':SIZE,'verified_range_bytes':SIZE})
        print('Pinned archive already verified');return
    # Enough room for retained ranges AND final archive; no cap or auto-deletion.
    if shutil.disk_usage(OUT).free < RESERVE+2*SIZE:raise RuntimeError('Disk reserve requires clean pause')
    atomic(OUT/'acquisition-protocol.json',{'version':1,'url':URL,'bytes':SIZE,'expected_md5':MD5,
          'chunk_bytes':CHUNK,'network_workers':8,'reserve_bytes':RESERVE,'preserve_archives_and_checkpoints':True})
    prefix=source/'flywire_synapses_783.feather.partial'
    tasks=[]
    for start in range(0,SIZE,CHUNK):
        end=min(start+CHUNK,SIZE)-1;path=parts/f'{start:012d}-{end:012d}.bin'
        # Reuse completed prefix bytes from the original serial attempt.
        if not path.exists() and prefix.exists() and prefix.stat().st_size>end:
            with prefix.open('rb') as f:f.seek(start);data=f.read(end-start+1)
            path.write_bytes(data);atomic(path.with_suffix('.json'),{'start':start,'end':end,'sha256':hashfile(path),'origin':'saved serial prefix'})
        tasks.append((start,end,path))
    lock=threading.Lock();completed={};began=time.monotonic();downloaded=0
    def progress(status):
        atomic(OUT/'acquisition-progress.json',{'status':status,'completed_ranges':len(completed),'ranges':len(tasks),
              'verified_range_bytes':sum(completed.values()),'archive_bytes':SIZE,'elapsed_seconds':time.monotonic()-began})
    def acquire(task):
        start,end,path=task;meta=path.with_suffix('.json');expected=end-start+1
        if path.exists() and meta.exists() and path.stat().st_size==expected:
            d=json.loads(meta.read_text())
            if d['start']==start and d['end']==end and d['sha256']==hashfile(path):return task
            raise RuntimeError('Cached range integrity mismatch')
        for attempt in range(8):
            if shutil.disk_usage(OUT).free < RESERVE+CHUNK*16:raise RuntimeError('Disk reserve: paused without deletion')
            temporary=path.with_suffix(f'.attempt-{attempt}');headers=path.with_suffix(f'.headers-{attempt}')
            result=subprocess.run(['curl','-fsSL','--connect-timeout','15','--max-time','55','--range',f'{start}-{end}',
                    '-D',str(headers),URL,'-o',str(temporary)],capture_output=True,text=True)
            header=headers.read_text().lower() if headers.exists() else ''
            if (result.returncode==0 and temporary.stat().st_size==expected
                    and f'content-range: bytes {start}-{end}/{SIZE}' in header):
                temporary.replace(path);atomic(meta,{'start':start,'end':end,'sha256':hashfile(path),'origin':'verified HTTP Content-Range'})
                return task
            # A stopped or rate-limited request preserves its partial response.
            time.sleep(min(5*(attempt+1),30))
        raise RuntimeError(f'Range {start}-{end} failed; rerun reuses verified ranges')
    progress('downloading')
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for start,end,path in pool.map(acquire,tasks):
                with lock:
                    completed[start]=end-start+1;progress('downloading')
        progress('assembling')
        if shutil.disk_usage(OUT).free < RESERVE+SIZE:raise RuntimeError('Disk reserve before assembly: source preserved')
        tmp=source/'flywire_synapses_783.feather.assembled'
        with tmp.open('wb') as output:
            for _,_,part in tasks:
                if shutil.disk_usage(OUT).free < RESERVE+part.stat().st_size:raise RuntimeError('Disk reserve during assembly: source preserved')
                with part.open('rb') as inp:shutil.copyfileobj(inp,output,8*1024**2)
            output.flush();os.fsync(output.fileno())
        assert tmp.stat().st_size==SIZE
        digest=hashfile(tmp,'md5')
        if digest!=MD5:raise RuntimeError('Whole archive checksum failed; source preserved')
        fullsha=hashfile(tmp);tmp.replace(final)
        atomic(OUT/'archive-identity.json',{'version':1,'url':URL,'bytes':SIZE,'md5':digest,'sha256':fullsha,
               'published_checksum_verified':True,'ranges_preserved':True,'elapsed_seconds':time.monotonic()-began})
        progress('complete');print('Complete: published MD5 and whole archive SHA256 verified',flush=True)
    except Exception as exc:
        progress('interrupted_or_failed');atomic(OUT/'acquisition-error.json',{'error':str(exc),'preserved':True});raise


if __name__=='__main__':main()
