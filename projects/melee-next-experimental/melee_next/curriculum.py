import numpy as np

class Curriculum:
    def __init__(self,config,state=None):
        self.config=config;self.level=config.cpu_level;self.history={x:[] for x in config.opponents}
        if state:
            self.level=state['level'];self.history={x:list(state.get('history',{}).get(x,[])) for x in config.opponents}
    def record(self,row):
        if row.get('cpu_level')!=self.level or row.get('opponent') not in self.history:return
        history=self.history[row['opponent']];history.append(int(row['result']=='win'));del history[:-30]
        if self.config.curriculum and self.level<9 and all(len(h)>=10 and sum(h)/len(h)>=.7 for h in self.history.values()):
            self.level+=1;self.history={x:[] for x in self.config.opponents}
    def state(self):
        hardness=np.array([1-(sum(h)+1)/(len(h)+2) for h in self.history.values()])
        weights=.5/len(hardness)+.5*hardness/hardness.sum()
        return dict(level=self.level,history=self.history,opponents=list(self.history),weights=weights.tolist())
