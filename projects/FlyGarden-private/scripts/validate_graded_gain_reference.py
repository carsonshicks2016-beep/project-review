"""Reproducible isolated diagnostics; does not acquire or mutate a brain worker."""
from pathlib import Path
import argparse
import hashlib
import json
import resource
import time
import shutil
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flygarden.graded_gain_reference import run_reference, GradedGainReference, PUBLISHED_PARAMETERS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/brain-integration/recovery/graded-compartment-reference-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_worker(path):
    saved = json.loads(path.read_text())
    state = GradedGainReference.restore(saved)
    rows = []
    for i in range(500): rows.append(state.step(.8 if i < 25 else 0)['activity'])
    np.save(path.with_suffix('.continued.npy'), np.asarray(rows))


def main():
    start = time.monotonic()
    if shutil.disk_usage(ROOT).free < 2 * 1024**3 + 50 * 1024**2:
        raise RuntimeError('Preserve the 2 GiB disk reserve')
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT/'literature/published-source/dynamical_model'
    b = list(PUBLISHED_PARAMETERS)
    assert all(format(x, '.14g') in (source/'LN_activity_injection_dynamics.m').read_text()
               for x in [3835])
    protocol = {'version': 1, 'scope': 'isolated published rate mechanism, no full brain',
                'parameters': b, 'parameter_source': 'LN_activity_injection_dynamics.m',
                'dt_ms': 1., 'burn_in_ms': 11000, 'duration_after_burn_in_ms': 6000,
                'pulses_after_burn_in_ms': [[300,800],[3300,3800]],
                'conditions': ['none', 'channel_a', 'channel_b', 'both'],
                'variants': ['published', 'no_post', 'no_pre', 'neither'],
                'synthetic_extension': 'Two independent copies of the published single-glomerulus model; channels are not antenna sides or exact anatomical compartments.',
                'gates': {'finite_nonnegative': True, 'pulse_peak_gain': 10.,
                          'tail_deviation_from_matched_none': 5.,
                          'untouched_channel_max_error': 1e-12,
                          'mirror_max_error': 1e-12,
                          'literal_translation_max_error': 1e-12,
                          'fresh_process_continuation_max_error': 0.,
                          'continuous_limit_error_must_decrease': True},
                'claims_excluded': ['experimental data fit', 'electrical cable compartments',
                                    'graded GABA release kinetics', 'exact FlyWire root mapping',
                                    'full-network recovery', 'navigation', 'learning'],
                'frozen_sources': {str(p.relative_to(ROOT)):digest(p) for p in [
                    ROOT/'flygarden/graded_gain_reference.py', Path(__file__).resolve(),
                    source/'run_AL_model.m', source/'LN_activity_injection_dynamics.m',
                    ROOT/'tests/test_graded_gain_reference.py', ROOT/'flygarden/brain.py']}}
    # Freeze diagnostic thresholds and source identities before computations.
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    t = np.arange(17000)
    pulse = ((t >= 11300)&(t < 11800)) | ((t >= 14300)&(t < 14800))
    traces = {}; metrics = {}; checkpoints = {}
    for variant in protocol['variants']:
        params = b.copy()
        if variant in ['no_pre', 'neither']: params[4:6] = [0.,0.]
        if variant in ['no_post', 'neither']: params[6] = 0.
        quiet, _, _ = run_reference(np.zeros(len(t)), params)
        active, _, state = run_reference(pulse.astype(float), params)
        for condition in protocol['conditions']:
            channels = np.stack([active if condition in ['channel_a','both'] else quiet,
                                 active if condition in ['channel_b','both'] else quiet], axis=1)
            traces[f'{variant}_{condition}'] = channels
        gains = [float(np.max(active[a:c,0]-quiet[a:c,0])) for a,c in [(11300,11800),(14300,14800)]]
        tails = [float(np.max(np.abs(active[a:c,0]-quiet[a:c,0]))) for a,c in [(13500,14000),(16500,17000)]]
        local_error = float(np.max(np.abs(traces[f'{variant}_channel_a'][:,1]-quiet)))
        mirror_error = float(np.max(np.abs(traces[f'{variant}_channel_a'][:,::-1]-traces[f'{variant}_channel_b'])))
        metrics[variant] = {'pulse_peak_gains':gains, 'tail_max_deviations':tails,
                            'untouched_channel_max_error':local_error,'mirror_max_error':mirror_error,
                            'finite_nonnegative':bool(np.isfinite(active).all() and np.min(active)>=0),
                            'pulse_response_passed': bool(min(gains)>=10),
                            'tail_recovery_passed': bool(max(tails)<=5),
                            'note':'Recovery is relative to a time-matched baseline with spontaneous input; resource recovery may be slower than the local activity time constant.'}
        checkpoint_state = GradedGainReference(params)
        for value in pulse[:14775]: checkpoint_state.step(float(value))
        saved = OUT/f'{variant}-checkpoint.json'; saved.write_text(json.dumps(checkpoint_state.checkpoint()))
        expected = np.array([checkpoint_state.step(.8 if i <25 else 0)['activity'] for i in range(500)])
        import subprocess, sys
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--continue-checkpoint',str(saved)],check=True,cwd=ROOT)
        observed = np.load(saved.with_suffix('.continued.npy'))
        checkpoints[variant] = {'samples':500,'exact':bool(np.array_equal(expected,observed)),
                                'max_error':float(np.max(np.abs(expected-observed)))}
        np.save(OUT/f'{variant}-expected-continuation.npy',expected)
    np.savez_compressed(OUT/'traces.npz',time_ms=t,**traces)
    # Independent continuous ODE reference for numerical convergence.
    # Use a zero-background, constant stimulus: no discontinuity in this check.
    def rhs(_, y):
        v,a=y[:4],y[4:]
        s = np.full(2, .4*b[2]+b[3])/(1+v[1]*np.array(b[4:6]))
        u = s*a
        return np.r_[(-v+np.array([u[0]-b[6]*v[2],u[0],v[0],0]))/15,
                     -b[1]*s*a+(1-a)/b[0]]
    y0 = np.r_[0.,1.,0.,0.,.5,.5]
    reference = solve_ivp(rhs,(0,500),y0,rtol=1e-11,atol=1e-12,dense_output=True)
    assert reference.success
    errors=[]
    for dt in [1.,.5,.25]:
        _,_,state=run_reference(np.full(int(500/dt),.4),dt_ms=dt)
        errors.append(float(np.max(np.abs(np.r_[state.v,state.resources]-reference.y[:,-1]))))
    convergence={'dt_ms':[1.,.5,.25],'max_final_state_error':errors,
                 'decreases': bool(errors[2]<errors[1]<errors[0]),
                 'note':'Continuous ODE convergence, not native MATLAB execution or experimental validation.'}
    assert all(c['exact'] for c in checkpoints.values())
    assert convergence['decreases']
    assert all(v['finite_nonnegative'] and v['mirror_max_error']<=1e-12 and v['untouched_channel_max_error']<=1e-12 for v in metrics.values())
    fig, axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
    colors={'published':'#6c3bd1','no_post':'#dd762b','no_pre':'#299a81','neither':'#666666'}
    for variant in protocol['variants']:
        v=traces[f'{variant}_channel_a'][11000:]
        q=traces[f'{variant}_none'][11000:]
        for row,pop in enumerate([0,2,1]):
            for col in range(2):
                axes[row,col].plot(t[11000:]/1000-11,v[:,col,pop]-q[:,col,pop],color=colors[variant],label=variant,linewidth=1.2)
    for row,label in enumerate(['PN response above matched baseline','Local LN continuous activity above baseline','Presynaptic LN activity above baseline']):
        axes[row,0].set_ylabel(label)
        for col in range(2):
            axes[row,col].axvspan(.3,.8,color='#ccc',alpha=.4)
            axes[row,col].axvspan(3.3,3.8,color='#ccc',alpha=.4)
            axes[row,col].grid(alpha=.2)
    axes[0,0].set_title('Stimulated channel A');axes[0,1].set_title('Unstimulated independent channel B')
    axes[0,0].legend(fontsize=9);axes[2,0].set_xlabel('Time after 11-second burn-in (seconds)');axes[2,1].set_xlabel('Seconds')
    fig.suptitle('Published rate-model port: synthetic local channels, not a full-brain or cable model')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    results={'version':1,'metrics':metrics,'fresh_process_checkpoints':checkpoints,
             'convergence':convergence,'variants':4,'synthetic_conditions':4,
             'application_changed':False,'promoted':False,'full_brain_run':False,
             'experimental_validation':False,'wall_seconds':time.monotonic()-start,
             'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'versions':{'numpy':np.__version__,'scipy':scipy.__version__},
             'frozen_sources_reverified':all(digest(ROOT/p)==h for p,h in protocol['frozen_sources'].items())}
    assert results['frozen_sources_reverified']
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--continue-checkpoint',type=Path)
    args=parser.parse_args()
    if args.continue_checkpoint: checkpoint_worker(args.continue_checkpoint)
    else: main()
