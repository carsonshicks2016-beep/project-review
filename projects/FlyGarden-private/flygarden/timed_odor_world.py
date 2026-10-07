"""Prospective physical odor-field fixture; no cue identity enters the brain."""
import math
from .world import World


class TimedOdorWorld(World):
    version='physical-sided-odor-fixture-v1'

    def configure(self,cue):
        if cue not in ('odor_left','odor_right'):raise ValueError('Unknown odor fixture')
        self.cue=cue;self.odor_started=False;self.odor_stopped=False
        side=1 if cue=='odor_left' else -1
        self.arena['predator']['enabled']=False
        self.arena['foods']=[dict(id='odor_source',x=12*math.cos(.9),y=side*12*math.sin(.9),odor=0,units=0)]

    def advance(self,dt,body):
        reward=super().advance(dt,body)
        if self.time>=.5 and not self.odor_started:
            self.odor_started=True;self.arena['foods'][0]['units']=3
            self.event('odor_onset','Odor source enabled; fly samples its local antenna field')
        if self.time>=1.5 and not self.odor_stopped:
            self.odor_stopped=True;self.arena['foods'][0]['units']=0
            self.event('odor_offset','Odor source disabled')
        return reward
