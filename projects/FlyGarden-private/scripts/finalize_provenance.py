import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'reports/provenance.json';r=json.loads(p.read_text());r['licenses']['flygym']='Apache-2.0, pinned 2.1 source';r['alterations'] += ['Dedicated checkpointable Bernoulli spike generator replaces Brian runtime pointer-backed Poisson RNG; probability rate*0.1ms','Physics 100us; supplied gait controller 500us','Physical antennal sites; actual retinal images available for inspection only']
r['alterations']=list(dict.fromkeys(r['alterations']))
r['application_hashes']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ('flygarden','static','scripts') for f in (root/folder).glob('*') if f.is_file() and f.suffix in ('.py','.js','.html','.css')};p.write_text(json.dumps(r,indent=2));(root/'requirements.lock').write_text(subprocess.check_output(['/opt/homebrew/bin/uv','pip','freeze','--python',sys.executable],text=True))
