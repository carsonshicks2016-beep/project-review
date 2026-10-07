"""Independent published mean-field reference; no garden-controller import/change."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
OUT=ROOT/'reports/brain-integration/recovery/olfactory-dynamics-reference-v1'
P={'tau_e':.055,'tau_p':.3,'tau_d':.1,'tau_f':.05,'rho':.008,'U':.24,'omega_ee_ns':75.,'omega_ie_ns':21.,'k_hz_per_ns':5.}
INITIAL=np.array([0.,0.,1.,1.,0.])
def rhs(t,y,input_fn,public=0.):
 pn,ln,p,x,u=y;r=input_fn(t);up=u+P['U']*(1-u)
 return np.array([-pn/P['tau_e']+P['omega_ee_ns']*P['k_hz_per_ns']*up*x*p*r,
 -ln/P['tau_e']+P['omega_ie_ns']*P['k_hz_per_ns']*(r+public),
 (1/(1+P['rho']*ln)-p)/P['tau_p'],(1-x)/P['tau_d']-x*up*p*r,
 -u/P['tau_f']+P['U']*(1-u)*p*r])
def scipy_solution(input_fn,duration,public=0.,initial=INITIAL,times=None,start=0.):
 if times is None:times=np.linspace(start,duration,round((duration-start)/.001)+1)
 z=solve_ivp(lambda t,y:rhs(t,y,input_fn,public),(start,duration),initial,method='DOP853',t_eval=times,rtol=1e-10,atol=1e-12,max_step=.01)
 assert z.success and np.isfinite(z.y).all();assert np.min(z.y[2:])>=-1e-9 and np.max(z.y[2:])<=1+1e-9
 return z.t,z.y.T

def rk4_independent(input_fn,duration,dt,public=0.):
 # Separate spelling of equations and integration avoids reusing SciPy RHS.
 state=np.array([0.,0.,1.,1.,0.]);rows=[state.copy()];times=[0.];stride=round(.001/dt)
 def derivatives(t,z):
  e,i,inhib,res,fac=z;R=input_fn(t);release=.24+.76*fac
  return np.array([-e/.055+375*release*res*inhib*R,-i/.055+105*(R+public),(-inhib+1/(1+.008*i))/.3,(1-res)/.1-res*release*inhib*R,-fac/.05+.24*(1-fac)*inhib*R])
 for tick in range(round(duration/dt)):
  t=tick*dt;a=derivatives(t,state);b=derivatives(t+dt/2,state+dt*a/2);c=derivatives(t+dt/2,state+dt*b/2);d=derivatives(t+dt,state+dt*c);state=state+dt*(a+2*b+2*c+d)/6
  if (tick+1)%stride==0:times.append((tick+1)*dt);rows.append(state.copy())
 return np.array(times),np.array(rows)

def analytic(R,public=0.):
 # Equation12 directly, separately from dynamic state equations.
 te=.055;w=375.;theta=1+.008*.055*105*(R+public);U=.24;F=.05;D=.1
 pn=te*w*U*R*(theta+F*R)/(theta**2+theta*(F+D)*U*R+D*F*U*R**2)
 ln=te*105*(R+public);p=1/(1+.008*ln);u=U*F*p*R/(1+U*F*p*R);up=u+U*(1-u);x=1/(1+D*up*p*R)
 return np.array([pn,ln,p,x,u])

def triangle(peak_time):
 return lambda t:120*t/peak_time if t<peak_time else max(0.,120*(2-t)/(2-peak_time))

def main():
 started=time.monotonic();ramps={};comparisons=[]
 for K in (400.,133.3,80.,66.7):
  fn=lambda t,K=K:K*t;t,y=scipy_solution(fn,8);t2,z=rk4_independent(fn,8,.0001);assert np.allclose(t,t2,atol=1e-12,rtol=0)
  err=np.max(np.abs(y-z),axis=0);assert np.max(err[:2])<.02 and np.max(err[2:])<2e-5
  _,half=rk4_independent(fn,8,.00005);assert np.max(np.abs(z[:,:2]-half[:,:2]))<.02
  comparisons.append({'K':K,'max_scipy_rk4_absolute_error':err.tolist(),'max_halved_dt_error':float(np.max(np.abs(z-half)))})
  ramps[str(K)]=(t,y);np.savez_compressed(OUT/f'ramp-{K:g}.npz',time=t,state=y,rk4_state=z)
 steady=[]
 for R in (0.,5.,20.,50.,120.):
  for public in (0.,100.,500.):
   _,y=scipy_solution(lambda t,R=R:R,20,public,times=np.array([20.]));expected=analytic(R,public);err=np.abs(y[-1]-expected);assert err[0]<1e-5 and np.max(err[2:])<1e-7
   steady.append({'ORN_hz':R,'public_hz':public,'expected':expected.tolist(),'actual':y[-1].tolist(),'max_absolute_error':float(err.max())})
 effective=1/(.008*.055*105);asymptote=analytic(0)[0] # replace by Eq15, which assumes finite effective input and p-limited drive.
 U=.24;D=.1;F=.05;asymptote=.055*375*U*effective*(1+F*effective)/(1+U*effective*(F+D)+U*D*F*effective**2)
 long=[]
 for K in (400.,133.3,80.,66.7):
  _,z=scipy_solution(lambda t,K=K:K*t,100,times=np.array([100.]));delta=abs(z[-1,0]-asymptote);assert delta<1
  long.append({'K':K,'finite_100s_PN_hz':float(z[-1,0]),'equation15_asymptote_hz':asymptote,'difference_hz':delta})
 triangles=[]
 for peak in (.1,.4,1.1,1.8):
  t,y=scipy_solution(triangle(peak),4);np.savez_compressed(OUT/f'triangle-{peak:g}.npz',time=t,state=y)
  triangles.append({'ORN_peak_time':peak,'PN_global_peak_hz':float(y[:,0].max()),'PN_peak_time':float(t[y[:,0].argmax()]),'PN_at_ORN_peak_hz':float(np.interp(peak,t,y[:,0]))})
 # Piecewise constant inputs are integrated at exact discontinuities rather than treating adaptive steps across them as a reference.
 windows=[(0.,1.,0.),(1.,3.,120.),(3.,4.,0.),(4.,4.5,120.),(4.5,7.,0.)];state=INITIAL.copy();ts=[];ys=[]
 for start,end,rate in windows:
  t,y=scipy_solution(lambda t,rate=rate:rate,end,initial=state,start=start);state=y[-1].copy();ts.extend(t[:-1]);ys.extend(y[:-1])
 ts=np.array(ts+[7.]);ys=np.vstack([ys,state]);assert ys[-1,0]<1e-5 and np.max(np.abs(ys[-1,2:]-INITIAL[2:]))<.001
 held=ys[(ts>=1)&(ts<3),0];assert held.max()>held[-1]
 np.savez_compressed(OUT/'step-repeated-pulse.npz',time=ts,state=ys)
 # State continuation using all five variables on a smooth triangular fixture.
 t,whole=scipy_solution(triangle(1.1),4);_,first=scipy_solution(triangle(1.1),.75,times=np.array([.75]));np.save(OUT/'reference-state.npy',first[-1]);saved=np.load(OUT/'reference-state.npy');tt,continued=scipy_solution(triangle(1.1),4,initial=saved,start=.75);error=float(np.max(np.abs(continued-whole[t>=.75])));assert error<1e-6
 result={'status':'equation_checks_passed','published_curve_validation':'pending','parameters':P,'equations':'Official XML1–5,12,15','units_convention':'Published k5 applied to both conductance-strength coefficients; initial feasibility distinguished this from PN-only conversion. Convention explicit, not author-code equivalence.','integrator_checks':comparisons,'steady_state_checks':steady,'ramp_asymptote_checks':long,'triangle_metrics':triangles,'continuation_max_absolute_error':error,'wall_seconds':time.monotonic()-started,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'full_network_integrated':False,'navigation_validated':False,'learning_validated':False}
 atomic_json(OUT/'equation-results.json',result);print(json.dumps({'status':result['status'],'asymptote':asymptote,'triangle_metrics':triangles,'seconds':result['wall_seconds']},indent=2))
if __name__=='__main__':main()
