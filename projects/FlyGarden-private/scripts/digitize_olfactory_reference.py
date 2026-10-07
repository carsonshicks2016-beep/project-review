"""External published-raster comparison, with fixed axes and color rules."""
import sys,json,hashlib
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
OUT=ROOT/'reports/brain-integration/recovery/olfactory-dynamics-reference-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 im=cv2.imread(str(OUT/'figure004.jpg'))[:,:,::-1].astype(float);palettes=[]
 for a,b in ((255,275),(286,307),(320,342),(352,376)):
  patch=im[a:b,395:440].reshape(-1,3);pix=patch[patch.max(axis=1)-patch.min(axis=1)>35];order=np.argsort(pix.sum(axis=1));palettes.append(np.median(pix[order[:max(5,len(pix)//10)]],axis=0))
 palettes=np.array(palettes);distance=np.sqrt(((im[:,:,None,:]-palettes[None,None,:,:])**2).sum(axis=3));nearest=distance.argmin(axis=2);valid=distance.min(axis=2)<45
 valid&=(im.max(axis=2)-im.min(axis=2)>35);valid[:10]=False;valid[420:]=False;valid[:, :123]=False;valid[:,729:]=False;valid[242:430,382:748]=False;valid[25:165,480:700]=False
 curves=[]
 for ch,K in enumerate((400.,133.3,80.,66.7)):
  samples=[];ambiguous=[]
  for t in np.linspace(.2,7.5,80):
   x=int(round(123+t*(728-123)/8));ys,xs=np.nonzero(valid[:,max(123,x-1):min(729,x+2)]&(nearest[:,max(123,x-1):min(729,x+2)]==ch))
   if len(ys)==0:continue
   unique=np.unique(ys);groups=np.split(unique,np.flatnonzero(np.diff(unique)>3)+1)
   if len(groups)!=1:
    ambiguous.append({'time_s':float(t),'pixel_x':x,'components_y':[g.tolist() for g in groups]});continue
   # At a fixed time, median colored pixel provides a line-centre estimate; no model-based search.
   y=float(np.median(ys));rate=(419-y)*175/(419-36);samples.append({'time_s':float(t),'pixel_x':x,'pixel_y':y,'PN_hz':float(rate)})
  assert len(samples)>=30,(K,len(samples))
  with np.load(OUT/f'ramp-{K:g}.npz') as z:pred=np.interp([r['time_s'] for r in samples],z['time'],z['state'][:,0])
  obs=np.array([r['PN_hz'] for r in samples]);err=np.abs(pred-obs)
  curves.append({'K':K,'samples':samples,'points':len(samples),'ambiguous_points_omitted':ambiguous,'mean_absolute_error_hz':float(err.mean()),'maximum_absolute_error_hz':float(err.max()),'passed':bool(err.mean()<=3 and err.max()<=6)})
 result={'status':'passed' if all(c['passed'] for c in curves) else 'failed','figure_sha256':sha(OUT/'figure004.jpg'),'protocol_sha256':sha(OUT/'PROTOCOL.md'),'script_sha256':sha(Path(__file__)),'axes':{'time_0_px':123,'time_8_px':728,'rate_0_px':419,'rate_175_px':36},'palettes_rgb':palettes.tolist(),'curves':curves,'limitations':'Published raster, not author numerical traces or raw electrophysiology. Axis/color/JPEG uncertainty. Ambiguous or missing colors omitted explicitly; no model-assisted pixel selection.'}
 atomic_json(OUT/'figure4-comparison.json',result);print(json.dumps([{k:v for k,v in c.items() if k not in ('samples','ambiguous_points_omitted')} for c in curves],indent=2))
if __name__=='__main__':main()
