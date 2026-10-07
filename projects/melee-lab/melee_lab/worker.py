"""Isolated training process; the UI never owns the game loop."""
import argparse
import os
import fcntl
import json
import time
import traceback
from collections import defaultdict,deque
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from .config import ROOT, Config
from .environment import MeleeEnv,Stopped
from .execution import VERSION as EXECUTION_VERSION
from .slot import make_env,EmulatorVecEnv,watch_stacks,reap_emulators,reap_workspace_dolphins,shutdown
from . import layout
from .state import OBS_SIZE
from .storage import write_json,save_model,validate_checkpoint
from .teacher import choose

class Runtime:
    def __init__(self,directory,config):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.config=config; self.model=None; self.last_poll=0; self.started=time.time()
        self.state={'status':'starting','phase':'booting','steps':0,'matches':0,'wins':0,
                    'cpu_level':config.cpu_level,'history':[],'started':self.started,'pid':os.getpid()}
        previous=json.loads((self.directory/'status.json').read_text()) if (self.directory/'status.json').exists() else {}
        self.started=previous.get('started',self.started)
        self.state['started']=self.started
        if (self.directory/'matches.jsonl').exists():
            from .analytics import MatchIndex
            rows,_=MatchIndex().read(self.directory/'matches.jsonl')
            self.state.update(matches=len(rows),wins=sum(r['result']=='win' for r in rows),history=rows[-100:])
        self.save_id=None
        # Set by run() to release the controllers and pump Dolphin. A pause that stops
        # calling console.step() desynchronises the Slippi stream exactly like a crash.
        self.idle=lambda:None
    def publish(self,**kwargs):
        self.state.update(kwargs); self.state['elapsed']=round(time.time()-self.started,1)
        write_json(self.directory/'status.json',self.state)
    def save(self,name='latest'):
        if self.model is not None:
            path=save_model(self.model,self.directory/name,self.config,
                bootstrap_samples=self.state.get('bootstrap_samples',0),
                training_cpu_level=self.state.get('cpu_level',self.config.cpu_level))
            self.publish(checkpoint=path)
    def control(self):
        if time.monotonic()-self.last_poll<.1: return
        self.last_poll=time.monotonic()
        path=self.directory/'control.json'
        while True:
            command=json.loads(path.read_text()) if path.exists() else {}
            if command.get('stop'): raise Stopped()
            if command.get('save') and command['save']!=self.save_id:
                self.save(); self.idle(); self.save_id=command['save']
            if not command.get('pause'):
                if self.state['status']=='paused': self.publish(status='running')
                break
            if self.state['status']!='paused': self.publish(status='paused')
            self.idle()
            time.sleep(.15)
    def result(self,info,kind):
        row={k:info[k] for k in ('result','players','cpu_level','frame','starting_stocks')}
        row['opponent_type']=info.get('opponent_type','cpu'); row['kind']=kind; row['time']=time.time(); row['return']=info['episode']['r']
        for key in ('stock_losses','self_destructs','self_destruct_rate','execution','opponent_execution','league_opponent'):
            if info.get(key) is not None: row[key]=info[key]
        with (self.directory/'matches.jsonl').open('a') as out: out.write(json.dumps(row)+'\n')
        history=(self.state['history']+[row])[-100:]
        # The win rate cannot tell being outplayed apart from walking off the stage.
        # This can, and it is the number every offstage change is judged against.
        recent=[r for r in history if r.get('stock_losses')]
        lost=sum(r['stock_losses'] for r in recent); thrown=sum(r.get('self_destructs',0) for r in recent)
        self.publish(matches=self.state['matches']+1,
            wins=self.state['wins']+int(info['result']=='win'),history=history,
            self_destruct_rate=round(thrown/lost,3) if lost else None,
            stocks_lost_recent=lost,stocks_self_destructed_recent=thrown)

def set_cpu_level(env,vec,value):
    """One emulator or many; the curriculum should not have to care which."""
    if vec is not None: vec.set_attr('cpu_level',value)
    elif env is not None: env.cpu_level=value


class Progress(BaseCallback):
    def __init__(self,runtime,env,curriculum,slots=(),vec=None,level=1,win_threshold=None):
        super().__init__(); self.runtime=runtime; self.env=env; self.curriculum=curriculum
        self.slots=[Path(s) for s in slots]; self.vec=vec; self.level=level
        self.win_threshold=.80 if win_threshold is None else float(win_threshold)
        # One window per level. Matches in flight when a level changes belong to the
        # level they started at, so they must not be counted toward the new one.
        self.windows=defaultdict(lambda: deque(maxlen=25))
        self.char_windows=defaultdict(lambda: defaultdict(lambda: deque(maxlen=10)))
        self.last_save=0; self.last_merge=0; self.last_publish=0

    def set_level(self,value):
        self.level=value
        set_cpu_level(self.env,self.vec,value)
        self.runtime.publish(cpu_level=value)
    def _on_step(self):
        r=self.runtime; r.control(); self.merge()
        # Delta, not modulo: with N envs the counter advances by N from a resumed
        # offset, so a modulo test can miss every multiple forever.
        if self.num_timesteps-self.last_publish>=32:
            r.publish(steps=self.num_timesteps); self.last_publish=self.num_timesteps
        for info,done in zip(self.locals['infos'],self.locals['dones']):
            if done:
                r.result(info,'self-play' if info.get('opponent_type') == 'policy' else 'ppo')
                if self.curriculum:
                    # Credit the match to the level it was actually played at.
                    match_level=info.get('cpu_level',self.level)
                    is_win=info.get('result')=='win'
                    self.windows[match_level].append(is_win)
                    opp_char=info.get('players',{}).get('2',{}).get('character')
                    if opp_char: self.char_windows[match_level][opp_char].append(is_win)
                    
                    current=self.windows[self.level]
                    chars=self.char_windows[self.level]
                    promoted=False
                    if len(current)>=20 and (sum(current)/len(current))>=self.win_threshold:
                        promoted=True
                    elif len(chars)>=3 and all(len(w)>=4 for w in chars.values()):
                        mean_char_wr=sum(sum(w)/len(w) for w in chars.values())/len(chars)
                        if mean_char_wr>=self.win_threshold:
                            promoted=True
                    
                    if promoted and self.level<9:
                        current.clear(); chars.clear()
                        self.set_level(self.level+1)
                    elif len(current)>=20 and (sum(current)/len(current))<=0.20 and self.level>1:
                        current.clear(); chars.clear()
                        self.set_level(self.level-1)
        if self.num_timesteps-self.last_save>=2048:
            r.save(); r.idle(); self.last_save=self.num_timesteps
        if self.num_timesteps-getattr(self,'last_candidate',r.state.get('initial_steps',0)) >= 250000:
            r.save(f'candidate-{self.num_timesteps}')
            self.last_candidate=self.num_timesteps
        return True

    def merge(self):
        """Slot processes own their own status files; surface the first one's live
        telemetry so the dashboard keeps working with N emulators."""
        if not self.slots: return
        now=time.monotonic()
        if now-self.last_merge<.25: return
        self.last_merge=now
        try: live=json.loads((self.slots[0]/'status.json').read_text())
        except (OSError,ValueError): return
        fields={k:live[k] for k in ('players','frame','action','connection') if k in live}
        if fields: self.runtime.publish(**fields)

    def _on_rollout_end(self):
        # PPO blocks for seconds optimising. Single emulator: the trainer pumps it.
        # Parallel: each slot's Serviced wrapper pumps itself, and idle is a no-op.
        self.runtime.idle()

    def _on_rollout_start(self):
        self.runtime.idle()


# Holding a sampled decision across frames was measured against states PROS visit,
# where the policy churns inputs 3x faster than a human. On the states the agent
# actually reaches its distribution is far more peaked and it already repeats 60% of
# frames, so a hold of 3 pushed persistence to 88% -- twice human -- and evaluated
# slightly worse. Off by default; --decision-hold keeps it available to experiment with.
DECISION_HOLD=1

ROLLOUT=1024   # total samples per PPO update, the single-emulator design point


def rollout_steps(n_envs):
    """SB3's n_steps is PER ENVIRONMENT, so the buffer is n_steps*n_envs. Left at
    1024 with four emulators every update saw 4096 samples and 16 minibatches
    instead of 4 -- a different optimiser regime than these settings were chosen for."""
    return max(64,ROLLOUT//max(1,int(n_envs)))


def retune_rollout(model,n_envs):
    """A checkpoint restores the n_steps it was saved with, so resuming into a
    different emulator count silently changes the update size too."""
    want=rollout_steps(n_envs)
    if model.n_steps==want: return want
    model.n_steps=want
    cls=getattr(model,'rollout_buffer_class',None) or type(model.rollout_buffer)
    if 'Recurrent' in cls.__name__ or hasattr(model.policy,'lstm_actor'):
        import torch as th
        from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
        from sb3_contrib.common.recurrent.type_aliases import RNNStates
        lstm = model.policy.lstm_actor
        n_layers = lstm.num_layers if lstm else 1
        h_dim = getattr(model.policy, 'lstm_output_dim', 128)
        single_shape = (n_layers, model.n_envs, h_dim)
        buffer_shape = (want, n_layers, model.n_envs, h_dim)
        model._last_lstm_states = RNNStates(
            (th.zeros(single_shape, device=model.device), th.zeros(single_shape, device=model.device)),
            (th.zeros(single_shape, device=model.device), th.zeros(single_shape, device=model.device)),
        )
        model.rollout_buffer = RecurrentRolloutBuffer(
            want, model.observation_space, model.action_space,
            hidden_state_shape=buffer_shape,
            device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda,
            n_envs=model.n_envs
        )
    else:
        model.rollout_buffer=cls(want,model.observation_space,
            model.action_space,device=model.device,gamma=model.gamma,
            gae_lambda=model.gae_lambda,n_envs=model.n_envs,**model.rollout_buffer_kwargs)
    return want


WIDTH=256
GAMMA=.997     # ~333 frames of lookahead; imitation returns must use the same discount


def new_model(env,seed,n_envs=1,net_arch=None):
    # net_arch is overridable so a controller transfer can rebuild the SOURCE's
    # feature extractor. Widening the default would otherwise make every existing
    # checkpoint untransferable, since only the action head is meant to change.
    net_arch=net_arch or dict(pi=[WIDTH,WIDTH],vf=[WIDTH,WIDTH])
    return PPO('MlpPolicy',env,learning_rate=1e-4,n_steps=rollout_steps(n_envs),batch_size=256,
        n_epochs=4,gamma=GAMMA,gae_lambda=.95,ent_coef=.005,clip_range=.15,
        target_kl=.025,policy_kwargs={'net_arch':net_arch},
        device='cpu',seed=seed,verbose=0)


def new_recurrent_model(env,seed,n_envs=1,lstm_hidden_size=128,net_arch=None):
    from sb3_contrib import RecurrentPPO
    from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
    net_arch=net_arch or dict(pi=[WIDTH],vf=[WIDTH])
    return RecurrentPPO(MlpLstmPolicy,env,learning_rate=1e-4,n_steps=rollout_steps(n_envs),batch_size=256,
        n_epochs=4,gamma=GAMMA,gae_lambda=.95,ent_coef=.005,clip_range=.15,
        target_kl=.025,policy_kwargs=dict(
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=1,
            net_arch=net_arch
        ),
        device='cpu',seed=seed,verbose=0)


class Anchor(BaseCallback):
    """Keeps cloning the tournament footage while PPO learns from live matches.

    A policy cloned from replays and then handed to PPO loses the clone within a few
    thousand steps: the value head starts random, so the first advantages are noise,
    and the entropy bonus pulls a sharp policy back toward uniform. Rather than choose
    between imitation and reinforcement, both run at once -- every rollout is preceded
    by a few supervised batches drawn from the demonstrations, weighted by a
    coefficient that decays to zero across the budget. Early on the footage holds the
    policy together while the value head calibrates; by the end PPO is unconstrained.

    The supervised steps run at rollout START, never at rollout end: PPO measures its
    ratio against the log-probabilities recorded during collection, so moving the
    policy between collection and the update would make every ratio wrong."""

    BATCHES=4          # roughly a fifth of the gradient steps in one PPO update
    BATCH=256

    def __init__(self,path,runtime,coefficient=.5,floor=0.):
        super().__init__()
        from .controller import DIMENSIONS
        from .imitation import Demonstrations
        self.data=Demonstrations.load(path)
        actions=self.data.actions
        if actions.ndim!=2 or actions.shape[1]!=len(DIMENSIONS) or (actions>=np.array(DIMENSIONS)).any() or (actions<0).any():
            raise ValueError('Demonstrations were recorded against a different controller. Re-parse the replays.')
        self.runtime=runtime; self.coefficient=float(coefficient); self.floor=float(floor)
        self.start=0; self.total=0; self.loss=0.

    def weight(self):
        """Linear hand-off with a permanent floor so PPO never forgets human tournament play."""
        if self.total<=0: return max(self.floor,self.coefficient)
        decayed=self.coefficient*max(0.,1.-(self.num_timesteps-self.start)/self.total)
        return max(self.floor,decayed)

    def _on_training_start(self):
        # self.num_timesteps only tracks the model from the first _on_step, so a resumed
        # run would read zero here and retire the anchor before its first batch.
        self.start=int(self.model.num_timesteps)
        # SB3 folds the steps already on the checkpoint into total_timesteps, so the
        # hand-off has to span what THIS call will run, not the counter's final value --
        # otherwise a run resumed at 600k of 10M retires the anchor at 0.03 instead of 0.
        self.total=max(1,int(self.locals.get('total_timesteps') or 0)-self.start)
        # The hand-off spans the REQUESTED budget. Ask for 100M steps, run 2M of them, and
        # the weight is still at 98% of its starting value when the run ends -- the policy
        # is cloning tournament footage the whole way and PPO never gets the wheel. The
        # horizon is published so a budget nobody intends to reach is visible up front.
        self.runtime.publish(anchor_samples=len(self.data),anchor_coefficient=self.coefficient,
                             anchor_horizon_steps=self.total,anchor_floor=self.floor,
                             anchor_weight_at_horizon=round(max(self.floor,0.),4))

    def _on_rollout_start(self):
        self.num_timesteps=int(self.model.num_timesteps)
        weight=self.weight()
        if weight<=0:
            self.runtime.publish(anchor_weight=0.); return
        self.runtime.control()
        policy=self.model.policy
        policy.set_training_mode(True)
        is_recurrent = hasattr(policy, 'lstm_actor') or 'Recurrent' in type(policy).__name__
        if not is_recurrent:
            for _ in range(self.BATCHES):
                index=np.random.randint(0,len(self.data),self.BATCH)
                x=torch.as_tensor(self.data.observations(index,reuse=True))
                y=torch.as_tensor(self.data.actions[index],dtype=torch.long)
                loss=-weight*policy.get_distribution(x).log_prob(y).mean()
                policy.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(),self.model.max_grad_norm)
                policy.optimizer.step()
                self.loss=float(loss.detach())/weight
                # PPO already blocks Dolphin for seconds; these batches must not add to it.
                self.runtime.idle()
        else:
            T = 16
            B = max(1, self.BATCH // T)
            n_layers = getattr(policy, 'lstm_actor', None).num_layers if hasattr(policy, 'lstm_actor') else 1
            hidden_size = getattr(policy, 'lstm_output_dim', 128)
            groups = self.data.groups if self.data.groups is not None else np.zeros(len(self.data), dtype=np.int32)
            n_samples = max(1, len(self.data) - T)
            for _ in range(self.BATCHES):
                starts = []
                for _ in range(B * 3):
                    idx = np.random.randint(0, n_samples)
                    if groups[idx] == groups[idx + T - 1]:
                        starts.append(idx)
                        if len(starts) == B: break
                while len(starts) < B: starts.append(np.random.randint(0, n_samples))
                indices = np.array([np.arange(s, s + T) for s in starts]).reshape(-1)
                x = torch.as_tensor(self.data.observations(indices, reuse=True), dtype=torch.float32)
                y = torch.as_tensor(self.data.actions[indices], dtype=torch.long)
                episode_starts = torch.zeros(B * T, dtype=torch.float32)
                episode_starts[0::T] = 1.0
                lstm_states = (torch.zeros(n_layers, B, hidden_size), torch.zeros(n_layers, B, hidden_size))
                dist, _ = policy.get_distribution(x, lstm_states, episode_starts)
                loss = -weight * dist.log_prob(y).mean()
                policy.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.model.max_grad_norm)
                policy.optimizer.step()
                self.loss = float(loss.detach()) / weight
                self.runtime.idle()
        policy.set_training_mode(False)
        self.runtime.publish(anchor_weight=round(weight,4),anchor_loss=round(self.loss,4))

    def _on_step(self): return True


def finetune(model,learning_rate=3e-5,entropy=.001):
    """PPO's defaults here are tuned to escape a random start. Applied to a cloned
    policy they undo it: 1e-4 is a large step for weights that already fit, and the
    entropy bonus rewards spreading probability back across 97 stick angles."""
    from stable_baselines3.common.utils import FloatSchedule
    model.learning_rate=learning_rate
    model.lr_schedule=FloatSchedule(learning_rate)
    model.ent_coef=entropy
    return model


def fit_demonstrations(model,observations,labels,epochs=12,keepalive=None):
    x=torch.as_tensor(np.asarray(observations),dtype=torch.float32)
    y=torch.as_tensor(np.asarray(labels),dtype=torch.long)
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=1e-3)
    model.policy.set_training_mode(True)
    loss_value=0
    for _ in range(epochs):
        for idx in torch.randperm(len(y)).split(256):
            dist=model.policy.get_distribution(x[idx])
            loss=-dist.log_prob(y[idx]).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),1)
            optimizer.step(); loss_value=float(loss.detach())
            if keepalive is not None: keepalive()
    model.policy.set_training_mode(False)
    with torch.no_grad():
        match=model.policy.get_distribution(x).mode()==y
        # MultiDiscrete: a sample is correct only when every axis matches.
        if match.dim()>1: match=match.all(dim=-1)
        accuracy=float(match.float().mean())
    return loss_value,accuracy


def bootstrap(model,env,runtime,steps):
    runtime.publish(phase='learning demonstrations',status='running')
    factored=getattr(env,'factored',False)
    if factored:
        from .teacher import choose_vector
        from .actions import vocabulary
        demo_actions=vocabulary(env.config)
    obs,_=env.reset(); observations=[]; labels=[]
    for i in range(steps):
        runtime.control()
        if factored:
            action=choose_vector(env.state,demo_actions)
        else:
            action=choose(env.state)
        observations.append(obs.copy()); labels.append(action)
        obs,_,done,trunc,info=env.step(action)
        if i%64==0: runtime.publish(bootstrap_samples=i+1)
        if done or trunc:
            runtime.result(info,'scripted demonstration'); obs,_=env.reset()
    runtime.publish(phase='fitting demonstration policy')
    for c in getattr(env,'controllers',[]): c.release_all()
    keepalive=lambda: env.pump(2)
    loss,accuracy=fit_demonstrations(model,observations,labels,keepalive=keepalive)
    np.savez_compressed(runtime.directory/'demonstrations.npz',observations=np.array(observations),actions=np.array(labels))
    env.pump(4)
    runtime.publish(bootstrap_samples=steps,imitation_loss=loss,imitation_training_accuracy=accuracy)
    runtime.save('bootstrap'); env.pump(4); runtime.save(); env.pump(4)
    env.close()


def evaluate(model,env,runtime,episodes,benchmark=False,human=False,deterministic=None,hold=None):
    """Play the policy and record what happened.

    A factored controller is SAMPLED, not taken at its mode. Each axis is independent,
    so the joint mode is every axis at its own most common value -- and on a pad that
    is stick-neutral with nothing pressed, because a human's thumb rests between
    inputs. Cloned from tournament footage the mode sits fully neutral on 41% of
    states against the demonstrated 28%, and since a neutral input leaves the next
    observation almost unchanged, the first such state is absorbing: the agent stands
    still until something hits it. Sampling reproduces the demonstrated distribution.

    The discrete vocabularies keep the mode, where an action is a named move and the
    most likely one is a meaningful answer."""
    from .analytics import summarize
    from .slot import COMPETITIVE_ROSTER
    factored=getattr(env,'factored',False)
    if deterministic is None: deterministic=not factored
    # Only a one-frame decision floor needs this; a longer action_frames already holds.
    if hold is None: hold=DECISION_HOLD if factored and env.config.action_frames==1 else 1
    results=[]; groups=defaultdict(list); execution=[]
    env.config.strict_evaluation=True
    # A benchmark is balanced: N episodes per matchup, all at CPU 9.
    roster=list(COMPETITIVE_ROSTER) if benchmark else [env.config.opponent]
    runtime.publish(status='running',phase='human challenge' if human else 'roster benchmark' if benchmark else 'evaluating neural policy',
                    target_episodes=episodes*len(roster))
    env.config.randomize_opponent=False
    for i in range(episodes*len(roster)):
        # Seed policy sampling as well as environment opponent selection.
        torch.manual_seed(env.config.seed+i)
        np.random.seed(env.config.seed+i)
        env.config.opponent=roster[i%len(roster)]
        obs,_=env.reset(seed=env.config.seed+i)
        action=None; frame=0
        lstm_states=None; ep_start=np.ones((1,),dtype=bool)
        policy=getattr(model,'policy',None)
        is_recurrent=bool(policy and (hasattr(policy,'lstm_actor') or 'Recurrent' in type(policy).__name__))
        while True:
            runtime.control()
            if action is None or frame%hold==0:
                predict_start=time.perf_counter()
                if is_recurrent:
                    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_start, deterministic=deterministic)
                    ep_start=np.zeros((1,),dtype=bool)
                else:
                    action,_=model.predict(obs,deterministic=deterministic)
                latency=(time.perf_counter()-predict_start)*1000
                if not hasattr(runtime,'inference_samples'): runtime.inference_samples=deque(maxlen=256)
                runtime.inference_samples.append(latency)
                if len(runtime.inference_samples)%64==0:
                    runtime.publish(inference_ms_p50=float(np.percentile(runtime.inference_samples,50)),inference_ms_p95=float(np.percentile(runtime.inference_samples,95)))
            frame+=1
            obs,_,done,trunc,info=env.step(action if factored else int(action))
            if done or trunc:
                runtime.result(info,'human challenge' if human else 'evaluation'); results.append(info['result']); groups[env.config.opponent].append(info)
                execution.append({k:info[k] for k in ('execution','opponent_execution') if k in info})
                break
    wins=results.count('win'); n=len(results); p=wins/n; z=1.96
    denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    radius=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/denominator
    report={'episodes':n,'wins':wins,'losses':results.count('loss'),
        'timeouts':results.count('timeout'),'win_rate':p,
        'win_rate_95_interval':[max(0,center-radius),min(1,center+radius)],
        'results':results,'config':asdict(env.config),
        'policy':'deterministic neural network' if deterministic else 'sampled neural network',
        'scripted_overrides':factored and env.config.execution_mode=='assisted','three_stock_start_verified':True,
        'execution_version':EXECUTION_VERSION,'execution_mode':env.config.execution_mode,
        'execution':execution,'strict_cpu_level':not getattr(env,'level_mismatches',0),
        'benchmark':benchmark,'opponent_type':'human' if human else 'policy' if getattr(env,'opponent_checkpoint',None) else 'cpu','decision_hold_frames':hold,
        'matchups':[dict(opponent=name, **summarize([{'result':v['result'],'return':v['episode']['r']} for v in rows])) for name,rows in groups.items()],
        'action_frames':env.config.action_frames,'observation_delay_frames':0,
        'input_transport':'local browser bridge' if human else None}
    write_json(runtime.directory/('challenge.json' if human else 'evaluation.json'),report)
    runtime.publish(evaluation=report)


def run(args):
    torch.set_num_threads(2)
    config=Config.load(args.config)
    runtime=Runtime(args.run_dir,config)
    watch_stacks(runtime.directory/'stacks-parent.log')
    write_json(runtime.directory/'config.json',asdict(config))
    slot_count=max(1,int(getattr(args,'envs',1) or 1))
    opponent_checkpoint=getattr(args,'opponent_checkpoint',None)
    runtime.publish(mode=args.mode,envs=slot_count,opponent_checkpoint=opponent_checkpoint)
    control_path=runtime.directory/'control.json'
    env=None; vec=None; slots=[]
    if slot_count==1:
        from .human import HumanInput
        human=HumanInput(runtime.directory/'human-input.json') if args.mode=='play' else None
        env=MeleeEnv(config,runtime.directory,runtime.control,runtime.publish,human_input=human,opponent_checkpoint=opponent_checkpoint)
        def idle():
            for c in getattr(env,'controllers',[]): c.release_all()
            env.pump(2)
        runtime.idle=idle
        trainer=env
    else:
        # Guard before spawning: these paths are not implemented across slots yet.
        if args.mode!='train':
            raise ValueError('Parallel emulators support training only.')
        if not args.checkpoint and not getattr(args,'anchor',None) and not getattr(args,'recurrent',False):
            raise ValueError('Demonstration bootstrap needs the single-emulator path; '
                             'pass --bootstrap 0 with a checkpoint, or --envs 1.')
        slots=[runtime.directory/f'env{i}' for i in range(slot_count)]
    model=None
    (ROOT/'.runtime').mkdir(exist_ok=True)
    emulator_lock=(ROOT/'.runtime/emulator.lock').open('w')
    try:
        try: fcntl.flock(emulator_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another Melee Lab process owns the emulator.')
        # Never reap another worker's emulators before owning the workspace lock.
        reap_workspace_dolphins(ROOT)
        config.validate()
        if args.mode == 'train':
            from .panel import open_window
            open_window(runtime.directory, envs=slot_count)
        anchor=None
        if getattr(args,'anchor',None):
            if args.mode!='train': raise ValueError('Demonstration anchoring applies to training only.')
            if getattr(config,'action_set','legacy')!='controller':
                raise ValueError('Demonstration anchoring needs the factored controller action set.')
            anchor=Anchor(args.anchor,runtime,args.anchor_coef,floor=getattr(args,'anchor_floor',.12))
        if args.mode in ('evaluate','play') and not args.checkpoint:
            raise ValueError('Evaluation and challenge require a checkpoint.')
        if slot_count>1:
            from .panel import BAND, enabled as panel_enabled
            boxes=layout.load(slot_count, reserve=BAND if panel_enabled(ROOT) else 0)
            # Contention grows with slot count, so give boot and menu navigation
            # proportionally longer before declaring a slot dead.
            patience=max(1.0,slot_count/3)
            vec=EmulatorVecEnv([make_env(config,runtime.directory,i,control_path,
                                         window=boxes[i] if i<len(boxes) else None,
                                         opponent_checkpoint=opponent_checkpoint,
                                         patience=patience)
                                for i in range(slot_count)],start_method='spawn')
            trainer=vec
        level=config.cpu_level
        if args.checkpoint:
            if getattr(args,'fast_inputs',False):
                from .transfer import transfer_policy
                meta=json.loads(Path(args.checkpoint).with_suffix('.json').read_text())
                model=transfer_policy(args.checkpoint,trainer,config.seed,slot_count)
                runtime.publish(controller_transfer=True)
            else:
                meta=validate_checkpoint(args.checkpoint,config)
                from .storage import load_model
                model=load_model(args.checkpoint,env=trainer,device='cpu')
            if args.mode=='train':
                level=meta.get('training_cpu_level',config.cpu_level)
            if type(model).__name__ == 'RecurrentPPO' or meta.get('architecture') == 'recurrent':
                runtime.publish(architecture='recurrent')
        elif getattr(args,'recurrent',False):
            model=new_recurrent_model(trainer,config.seed,slot_count)
            runtime.publish(architecture='recurrent')
        else: model=new_model(trainer,config.seed,slot_count)
        # An explicit level beats the checkpoint's, so a resumed run can restart the
        # curriculum somewhere the agent can actually win instead of at the top.
        if getattr(args,'level',None): level=int(args.level)
        config.cpu_level=level          # so a saved report names the level actually played
        set_cpu_level(env,vec,level)
        if args.mode=='train': retune_rollout(model,slot_count)
        if getattr(args,'finetune',False) and args.mode=='train':
            # 3e-5 and an entropy of .001 are a small step for weights that already fit.
            # Applied to a random initialisation they are small enough that PPO never
            # escapes the start -- a run launched this way spent 2.2M steps producing a
            # policy whose action distribution was identical in every game state.
            if not args.checkpoint:
                raise ValueError('--finetune lowers the learning rate for a policy that is '
                                 'already trained. There is nothing to fine-tune without '
                                 '--checkpoint; drop the flag to train from scratch.')
            finetune(model); runtime.publish(finetune=True)
        runtime.publish(cpu_level=level,anchor=str(args.anchor) if getattr(args,'anchor',None) else None)
        runtime.model=model
        if getattr(args,'fast_inputs',False): runtime.save('controller-transfer')
        if args.mode in ('evaluate','play'):
            if args.mode=='play': config.speed=1.0
            evaluate(model,env,runtime,args.episodes,benchmark=getattr(args,'benchmark',False),
                     human=args.mode=='play',deterministic=getattr(args,'deterministic',None) or None,
                     hold=getattr(args,'decision_hold',None))
        else:
            if args.bootstrap and not args.checkpoint: bootstrap(model,env,runtime,args.bootstrap)
            runtime.publish(status='running',phase='Frozen-opponent training' if opponent_checkpoint else 'PPO training',initial_steps=json.loads((runtime.directory/'request.json').read_text()).get('checkpoint_identity',{}).get('steps',0) if (runtime.directory/'request.json').exists() else int(model.num_timesteps),requested_steps=args.steps)
            progress=Progress(runtime,env,not args.no_curriculum and not opponent_checkpoint and not config.league_manifest,slots,vec,level,win_threshold=getattr(args,'win_threshold',.60))
            model.learn(total_timesteps=args.steps,
                        callback=[progress,anchor] if anchor else progress,
                        reset_num_timesteps=not bool(args.checkpoint))
            runtime.save()
        runtime.publish(status='completed',phase='finished')
    except (Stopped,KeyboardInterrupt):
        if args.mode=='train': runtime.save()
        runtime.publish(status='stopped',phase='saved and stopped')
    except Exception as exc:
        # A slot that exits on a requested stop leaves the trainer holding an EOFError;
        # that is a stop, not a failure.
        try: stopping=json.loads((runtime.directory/'control.json').read_text()).get('stop')
        except (OSError,ValueError): stopping=False
        if stopping:
            if args.mode=='train': runtime.save()
            runtime.publish(status='stopped',phase='saved and stopped')
            return 0
        if model is not None and args.mode=='train': runtime.save('interrupted')
        # A slot that dies leaves the parent with a bare EOFError; the real cause is
        # in that slot's own status file, so surface it on the run.
        detail=str(exc)
        for path in sorted(runtime.directory.glob('env*/status.json')):
            try: failed=json.loads(path.read_text()).get('failed')
            except (OSError,ValueError): continue
            if failed: detail=f'{path.parent.name}: {failed}'+(f' (trainer saw {type(exc).__name__})' if detail else '')
        runtime.publish(status='failed',error=detail or type(exc).__name__,phase='needs attention')
        traceback.print_exc()
        return 1
    finally:
        if vec is not None:
            forced=shutdown(vec)
            if forced: print('force-closed slots:',forced)
        if env is not None:
            try: env.close()
            except Exception: pass
        reaped=reap_emulators(runtime.directory)
        if reaped: print('reaped emulators:',reaped)
        emulator_lock.close()
    return 0


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['train','evaluate','play'],default='train')
    p.add_argument('--config'); p.add_argument('--run-dir',required=True)
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--bootstrap',type=int,default=4096)
    p.add_argument('--fast-inputs',action='store_true')
    p.add_argument('--opponent-checkpoint'); p.add_argument('--benchmark',action='store_true')
    p.add_argument('--checkpoint'); p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--no-curriculum',action='store_true')
    p.add_argument('--deterministic',action='store_true',help='evaluate at the policy mode even for a factored controller')
    p.add_argument('--decision-hold',type=int,help='frames to hold each sampled decision (default 3 for a 1-frame controller)')
    p.add_argument('--anchor',help='.npz demonstrations to keep cloning during PPO')
    p.add_argument('--anchor-coef',type=float,default=.5,help='initial weight of the demonstration loss (decays to zero)')
    p.add_argument('--anchor-floor',type=float,default=.12,help='minimum weight of demonstration loss to prevent forgetting (default 0.12)')
    p.add_argument('--win-threshold',type=float,default=.60,help='curriculum win rate threshold to advance level (default 0.60)')
    p.add_argument('--finetune',action='store_true',help='lower learning rate and entropy for a policy that is already trained')
    p.add_argument('--recurrent',action='store_true',help='use Recurrent PPO with LSTM memory core')
    p.add_argument('--envs',type=int,default=1,help='parallel emulators (each gets port+i)')
    p.add_argument('--level',type=int,help='override the CPU level (ignores the checkpoint of origin)')
    args=p.parse_args()
    if args.envs<1 or args.envs>12: p.error('--envs must be between 1 and 12.')
    if args.steps<1 or args.bootstrap<0 or args.episodes<1: p.error('Counts must be positive (bootstrap may be zero).')
    if not 0<=args.anchor_coef<=10: p.error('--anchor-coef must be between 0 and 10.')
    if not 0<=args.anchor_floor<=10: p.error('--anchor-floor must be between 0 and 10.')
    if not 0.1<=args.win_threshold<=1.0: p.error('--win-threshold must be between 0.1 and 1.0.')
    raise SystemExit(run(args))

if __name__=='__main__':main()
