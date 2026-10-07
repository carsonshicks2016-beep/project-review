"""Offline Three.js frames → streamed H.264. Raw recordings remain the source."""
import json,os,time,threading,subprocess,hashlib,math,shutil
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import numpy as np
from .storage import ROOT
from .recording import run_folder,space_check,atomic_json,neurons
from .morphology import skeleton,coverage
VIDEO_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def read_frames(folder):
    frames=[]
    meta=folder/'recording.json';limit=json.loads(meta.read_text()).get('frames') if meta.exists() else None
    for line in (folder/'frames.jsonl').read_text().splitlines():
        if limit is not None and len(frames)>=limit:break
        try:frames.append(json.loads(line))
        except ValueError:break
    return frames

def render_video(job,job_path):
    from playwright.sync_api import sync_playwright
    import imageio_ffmpeg
    folder=run_folder(job['run']);frames=read_frames(folder)
    if not frames:raise ValueError('No complete recorded frames')
    settings=job['settings'];fps=settings['fps'];start=frames[0]['world']['time'];end=frames[-1]['world']['time'];count=max(1,math.ceil((end-start)*fps-1e-9)+1)
    work=Path(job_path).parent;space_check(work,512*1024**2)
    manifest=json.loads((folder/'manifest.json').read_text());meta_path=folder/'recording.json';meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    # Only bounded anatomy is loaded into the renderer. Every neuron retains its raw events.
    selected=neurons(job['run'],population=settings['population'],limit=256);shapes=[];unavailable=[]
    from concurrent.futures import ThreadPoolExecutor
    def load_shape(n):
        try:
            s=skeleton(n['id'],lod='coarse')
            return {**n,'shape':s} if s['status']=='available' else None
        except OSError as exc:
            if 'Storage reserve' in str(exc):raise
            return None
        except Exception:return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number,(n,result) in enumerate(zip(selected['neurons'],pool.map(load_shape,selected['neurons']))):
            if result is not None:shapes.append(result)
            else:unavailable.append(n['id'])
            if number%8==0:
                job.update(phase='Loading recorded neuron shapes',morphology_loaded=len(shapes),morphology_requested=len(selected['neurons']));atomic_json(job_path,job)
    job.update(morphology_loaded=len(shapes),anatomy=dict(visible=len(shapes),modeled=selected.get('modeled_neurons',0),unavailable=unavailable),phase='Preparing recorded anatomy');atomic_json(job_path,job)
    payload=dict(manifest=manifest,geometry=json.loads((folder/'geometry.json').read_text()),neurons=shapes,recording=meta,settings=settings,start=start,end=end)
    payload_path=work/'render-input.json';atomic_json(payload_path,payload)
    source_folder=work/'renderer-source';source_folder.mkdir(exist_ok=True)
    frozen={name:(ROOT/'static'/name).read_bytes() for name in ('render.html','recorded-view.js','recording-camera.js')}
    for name,raw in frozen.items():(source_folder/name).write_bytes(raw)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT),**kwargs)
        def log_message(self,*args):pass
        def do_GET(self):
            from urllib.parse import urlsplit,unquote
            request_path=unquote(urlsplit(self.path).path)
            if request_path=='/render-input':
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(payload_path.read_bytes())
            elif request_path in ('/static/render.html','/static/recorded-view.js','/static/recording-camera.js'):
                raw=frozen[request_path.rsplit('/',1)[1]];self.send_response(200);self.send_header('Content-Type','text/html' if request_path.endswith('.html') else 'application/javascript');self.end_headers();self.wfile.write(raw)
            elif request_path.startswith('/static/') and (ROOT/request_path.lstrip('/')).resolve().is_relative_to(ROOT/'static'):
                super().do_GET()
            else:self.send_error(404)
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    writer=None;begun=time.perf_counter();temporary=work/'video.partial.mp4';spikes_i=np.array([],dtype=np.int32);spikes_t=np.array([],dtype=float);chunk_cursor=0;frame_cursor=0
    try:
        with sync_playwright() as p:
            chrome=os.environ.get('FLYGARDEN_CHROME','/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
            browser=p.chromium.launch(executable_path=chrome if Path(chrome).exists() else None,headless=True,args=['--enable-webgl','--ignore-gpu-blocklist','--use-angle=metal'])
            page=browser.new_page(viewport=dict(width=settings['width'],height=settings['height']),device_scale_factor=1)
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{http.server_port}/static/render.html');page.wait_for_function('window.renderReady === true',timeout=180000)
            writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','image2pipe','-vcodec','png','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],stdin=subprocess.PIPE,stderr=(work/'encoder.log').open('wb'))
            entries=meta.get('chunks',[])
            from .video_jobs import simulation_busy
            for index in range(count):
                while simulation_busy():
                    job['phase']='Waiting for simulation to pause';atomic_json(job_path,job);time.sleep(.25)
                stamp=min(end,start+index/fps)
                while frame_cursor+1<len(frames) and frames[frame_cursor+1]['world']['time']<=stamp+1e-8:frame_cursor+=1
                while chunk_cursor<len(entries) and entries[chunk_cursor]['start']<=stamp+1e-8:
                    entry=entries[chunk_cursor];path=folder/entry['file']
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Spike chunk integrity check failed')
                    with np.load(path,allow_pickle=False) as a:
                        spikes_i=np.concatenate((spikes_i,a['indices']));spikes_t=np.concatenate((spikes_t,a['times']))
                    chunk_cursor+=1
                keep=spikes_t>=stamp-.1-1e-9;spikes_i=spikes_i[keep];spikes_t=spikes_t[keep];now=(spikes_t<=stamp+1e-9)&(spikes_t>stamp-.1+1e-9)
                unique,counts=np.unique(spikes_i[now],return_counts=True);rates={str(int(i)):int(n)*10 for i,n in zip(unique,counts)}
                page.evaluate('(x)=>window.renderRecordedFrame(x)',dict(state=frames[frame_cursor],nextState=frames[frame_cursor+1] if frame_cursor+1<len(frames) else None,time=stamp,rates=rates))
                if errors:raise RuntimeError('Renderer: '+errors[0])
                writer.stdin.write(page.screenshot(type='png'))
                if index%30==0 or index==count-1:
                    space_check(work,64*1024**2);job.update(progress=(index+1)/count,phase='Rendering recorded frames',rendered_frames=index+1,total_frames=count);atomic_json(job_path,job)
            writer.stdin.close();code=writer.wait(timeout=120);browser.close()
            if code:raise RuntimeError('Video encoder failed; inspect encoder.log')
        os.replace(temporary,work/'video.mp4');job.update(phase='Complete',morphology_loaded=len(shapes),video='video.mp4',render_wall_seconds=time.perf_counter()-begun,simulated_duration=end-start,bytes=(work/'video.mp4').stat().st_size,renderer='Three.js offline Chromium / H.264',frame_count=count)
        atomic_json(work/'video-manifest.json',{**job,'renderer_sources':{**{'static/'+name:hashlib.sha256(raw).hexdigest() for name,raw in frozen.items()},'flygarden/video.py':VIDEO_SOURCE_SHA256},'status':'complete','finished':time.time(),'neuron_geometry':[dict(id=n['id'],index=n['index'],source_sha256=n['shape']['sha256'],lod=n['shape'].get('lod','full'),display_units=n['shape']['display_units']) for n in shapes],'recording_manifest_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),'scope':'Anatomical branches share point-neuron activity; no subcellular propagation inferred. Display interpolates rigid body poses; discrete world events retain their recorded timestamps.'})
        payload_path.unlink(missing_ok=True)  # Rebuildable export input; all source recordings and skeletons remain.
    finally:
        if writer is not None and writer.poll() is None:writer.terminate();writer.wait(timeout=10)
        http.shutdown();http.server_close()
