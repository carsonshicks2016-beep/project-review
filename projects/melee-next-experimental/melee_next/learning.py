"""PPO with strictly version-matched fragments and separate terminal/bootstrap masks."""
import numpy as np
import torch

def advantages(rewards,values,next_values,terminated,boundaries,discounts,lam=.95):
    out=np.zeros(len(rewards),np.float32);carry=0.
    for i in reversed(range(len(rewards))):
        delta=rewards[i]+discounts[i]*next_values[i]*(1-float(terminated[i]))-values[i]
        carry=delta+discounts[i]*lam*(1-float(boundaries[i]))*carry
        out[i]=carry
    return out,out+values

def update(policy,optimizer,fragments,config):
    data={key:torch.as_tensor(np.concatenate([f[key] for f in fragments])) for key in ('obs','actions','logp','advantages','returns')}
    adv=data['advantages'];adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    metrics=[];halt=False
    for epoch in range(config.epochs):
        for ix in torch.randperm(len(adv)).split(config.minibatch):
            logp,entropy,value=policy.score(data['obs'][ix],data['actions'][ix])
            logratio=logp-data['logp'][ix];ratio=logratio.exp()
            kl=((ratio-1)-logratio).mean()
            if kl.detach().item()>config.target_kl*1.5:
                halt=True;break
            surrogate=torch.minimum(ratio*adv[ix],ratio.clamp(1-config.clip,1+config.clip)*adv[ix])
            value_loss=.5*(value-data['returns'][ix]).square().mean()
            loss=-surrogate.mean()+.5*value_loss-config.entropy*entropy.mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite learner loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
            metrics.append([loss.item(),kl.item(),entropy.mean().item(),value_loss.item()])
        if halt:break
    means=np.mean(metrics,axis=0).tolist() if metrics else [0.,float(kl.detach()),0.,0.]
    return dict(zip(('loss','kl','entropy','value_loss'),means),early_stop=halt,minibatches=len(metrics))
