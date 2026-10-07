"""Engineered local-activity bridge; not a physiological fit or live controller.

Activity is dimensionless, release is activity * rate_scale_hz. Input signs
come from the frozen graph; upgraded GABA/MIP outputs are inhibitory by an
explicit type-level hypothesis. Millivolts remain outside the local integrator.
"""
import copy
import hashlib
import json
import numpy as np

MODEL = 'qualified-local-inhibition-adapter-v1'


class LocalInhibitionAdapter:
    def __init__(self, nodes, coupling, tau_s=.015, rate_scale_hz=100., dt_s=.0001):
        self.nodes = tuple(nodes)
        self.coupling = np.asarray(coupling, float).copy()
        n = len(self.nodes)
        if (not n or len(set(self.nodes)) != n or self.coupling.shape != (n, n)
                or not np.isfinite(self.coupling).all() or np.any(self.coupling > 0)
                or not all(np.isfinite(x) and x > 0 for x in (tau_s, rate_scale_hz, dt_s))
                or dt_s > tau_s / 10):
            raise ValueError('Invalid adapter configuration')
        self.tau_s, self.rate_scale_hz, self.dt_s = map(float, (tau_s, rate_scale_hz, dt_s))
        self.activity = np.zeros(n)
        self.steps = 0

    def step(self, drive):
        """Signed synapse-normalized rate/rate_scale drive, constant this step.

        Fourth-order Runge–Kutta integration of tau*dx/dt=-x+clip(drive+C*x,0,1).
        Bounded release and independence absent anatomical coupling are
        engineering choices, not evidence of electrically isolated branches.
        """
        d = np.asarray(drive, float)
        if d.shape != self.activity.shape or not np.isfinite(d).all():
            raise ValueError('Invalid dimensionless input drive')
        x = self.activity
        def rhs(y): return (-y + np.clip(d + self.coupling @ y, 0., 1.)) / self.tau_s
        k1 = rhs(x)
        k2 = rhs(x + .5*self.dt_s*k1)
        k3 = rhs(x + .5*self.dt_s*k2)
        k4 = rhs(x + self.dt_s*k3)
        next_x = np.clip(x + self.dt_s*(k1+2*k2+2*k3+k4)/6., 0., 1.)
        if not np.isfinite(next_x).all(): raise FloatingPointError('Nonfinite activity')
        self.activity = next_x
        self.steps += 1
        return self.release_hz()

    def release_hz(self):
        return (self.activity * self.rate_scale_hz).copy()

    def configuration(self):
        return {'nodes':list(self.nodes),'coupling':self.coupling.tolist(),
                'tau_s':self.tau_s,'rate_scale_hz':self.rate_scale_hz,'dt_s':self.dt_s}

    def checkpoint(self):
        c = self.configuration()
        digest = hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        return {'version':1,'model':MODEL,'configuration':c,'configuration_sha256':digest,
                'activity':self.activity.tolist(),'steps':self.steps}

    @classmethod
    def restore(cls, checkpoint):
        c = copy.deepcopy(checkpoint)
        if c.get('version') != 1 or c.get('model') != MODEL: raise ValueError('Incompatible checkpoint')
        cfg=c['configuration']
        digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        if digest != c.get('configuration_sha256'): raise ValueError('Corrupt checkpoint configuration')
        state=cls(**cfg); x=np.asarray(c['activity'],float)
        if (x.shape != state.activity.shape or not np.isfinite(x).all()
                or np.any(x<0) or np.any(x>1) or type(c['steps']) is not int or c['steps']<0):
            raise ValueError('Invalid checkpoint state')
        state.activity=x.copy();state.steps=c['steps'];return state


def make_drive(routes, rates_hz, input_site_totals, rate_scale_hz):
    """Compile signed count-weighted drive into local nodes.

    Rates are keyed by exact root strings or local node keys. Root rates
    represent the remaining spiking source; local-to-local inhibition belongs
    to coupling and is excluded here to prevent counting it twice.
    """
    drive=np.zeros(len(input_site_totals),float)
    for r in routes:
        if r['post_node'] < 0 or r['pre_node'] >= 0: continue
        rate=float(rates_hz[r['pre_root_id']])
        if not np.isfinite(rate) or rate<0: raise ValueError('Invalid spike rate')
        drive[r['post_node']] += r['original_sign']*r['site_count']*rate
    totals=np.asarray(input_site_totals,float)
    if np.any(totals<=0) or not np.isfinite(totals).all():raise ValueError('Missing input allocation')
    return drive / totals / rate_scale_hz


def output_drive_mv_per_s(routes, release_hz, mv_per_site_spike=.275):
    """Average inhibitory drive equivalent, not receptor conductance.

    To be used only by a separately validated clock/delay-aware full-network
    integrator. This function does not write to Brian2 neurons.
    """
    if not np.isfinite(mv_per_site_spike) or mv_per_site_spike<0:raise ValueError('Invalid scale')
    rates=np.asarray(release_hz,float)
    if not np.isfinite(rates).all() or np.any(rates<0):raise ValueError('Invalid release')
    result={}
    for r in routes:
        if r['pre_node']>=0 and r['post_node']<0:
            key=r['post_root_id']
            result[key]=result.get(key,0.)-r['site_count']*rates[r['pre_node']]*mv_per_site_spike
    return result
