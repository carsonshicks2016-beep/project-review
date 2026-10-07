from pathlib import Path
import os
import random
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .codec import AXES, OBS, FRAME, HISTORY, CONTINUOUS, SCHEMA
from .storage import atomic_json, digest

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.animation=nn.Embedding(400,16)
        self.character=nn.Embedding(33,8)
        self.stage=nn.Embedding(32,8)
        self.body=nn.Sequential(nn.Linear(HISTORY*(CONTINUOUS+56),256),nn.Tanh(),nn.Linear(256,128),nn.Tanh())
        self.actor=nn.Linear(128,sum(AXES));self.critic=nn.Linear(128,1)
        nn.init.orthogonal_(self.actor.weight,.01);nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight,1.);nn.init.zeros_(self.critic.bias)
    def forward(self,obs):
        x=obs.reshape(-1,HISTORY,FRAME); ids=x[:,:,CONTINUOUS:].long()
        encoded=torch.cat([x[:,:,:CONTINUOUS],self.animation(ids[:,:,0]),self.animation(ids[:,:,1]),
                           self.character(ids[:,:,2]),self.character(ids[:,:,3]),self.stage(ids[:,:,4])],dim=-1)
        hidden=self.body(encoded.flatten(1))
        distributions=[Categorical(logits=v) for v in self.actor(hidden).split(AXES,dim=-1)]
        return distributions,self.critic(hidden).squeeze(-1)
    def act(self,obs,deterministic=False):
        d,v=self(obs)
        action=torch.stack([q.logits.argmax(-1) if deterministic else q.sample() for q in d],dim=-1)
        return action,torch.stack([q.log_prob(action[:,i]) for i,q in enumerate(d)]).sum(0),v
    def score(self,obs,action):
        d,v=self(obs)
        return torch.stack([q.log_prob(action[:,i]) for i,q in enumerate(d)]).sum(0),torch.stack([q.entropy() for q in d]).sum(0),v

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)

def contract(config):
    return {'schema':SCHEMA,'axes':list(AXES),'character':config.character,'action_frames':config.action_frames,'execution':'raw'}

def save(path,policy,optimizer,config,progress):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise ValueError('Immutable checkpoint already exists')
    payload={'contract':contract(config),'model':policy.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,
             'config':config.asdict(),'progress':progress,'torch_rng':torch.get_rng_state()}
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary);os.replace(temporary,path)
    identity={'path':str(path.resolve()),'sha256':digest(path),'contract':contract(config),'backend':config.backend,**progress}
    atomic_json(path.with_suffix('.json'),identity)
    return identity

def load(path,policy,config,optimizer=None):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('config',{}).get('backend') != config.backend: raise ValueError('Synthetic and Melee checkpoints cannot be mixed')
    if payload['contract'] != contract(config): raise ValueError('Checkpoint character, timing, or input schema differs from this run')
    policy.load_state_dict(payload['model'])
    if optimizer is not None and payload.get('optimizer'):optimizer.load_state_dict(payload['optimizer'])
    if payload.get('torch_rng') is not None:torch.set_rng_state(payload['torch_rng'])
    return payload['progress']

def publish(path,policy,version):
    path=Path(path);temporary=path.with_suffix('.tmp')
    torch.save({'model':policy.state_dict(),'version':version},temporary);os.replace(temporary,path)
