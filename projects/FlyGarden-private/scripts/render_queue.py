"""Single render worker; waits for simulation/experiment batches to finish."""
import sys,time,json,fcntl,urllib.request,os,traceback,signal
from pathlib import Path
def interrupted(signum,frame):raise KeyboardInterrupt('Render worker interrupted')
signal.signal(signal.SIGTERM,interrupted)
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from flygarden import video_jobs,experiment_jobs
from flygarden.recording import atomic_json
import psutil
lock=open(root/'.runtime/video-worker.lock','a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit(0)

def busy():return video_jobs.simulation_busy()

while True:
    queued=[j for j in video_jobs.statuses() if j['status']=='queued']
    if not queued:break
    if busy():time.sleep(1);continue
    j=sorted(queued,key=lambda j:j['created'])[0];path=video_jobs.job_folder(j['id'])/'job.json'
    j.update(status='rendering',pid=os.getpid(),process_created=psutil.Process().create_time(),started=time.time());atomic_json(path,j)
    try:
        from flygarden.video import render_video
        render_video(j,path)
        j.update(status='complete',progress=1.,finished=time.time(),error=None)
    except Exception as exc:
        traceback.print_exc();j.update(status='failed',error=str(exc),finished=time.time())
    atomic_json(path,j)
