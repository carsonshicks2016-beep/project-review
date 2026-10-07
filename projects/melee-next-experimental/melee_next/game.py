"""One process owns one Dolphin, one port, and all controller/console calls."""
import errno
import os
import time
from pathlib import Path
import numpy as np
import melee
from .codec import History, Reward, apply, players, OBS
from .storage import atomic_json

class Cancelled(Exception): pass

class DolphinGame:
    def __init__(self,config,directory,index=0,control=lambda:None,telemetry=lambda **kw:None):
        self.config=config;self.directory=Path(directory);self.index=index;self.control=control;self.telemetry=telemetry
        self.console=None;self.state=None;self.controllers=[];self.history=History();self.reward=Reward()
        self.directory.mkdir(parents=True,exist_ok=True)
        self.started_frame=0;self.episode_return=0.;self.opponent=None;self.level=None
    def launch(self):
        c=self.config
        home=self.directory/'dolphin-user'
        self.console=melee.Console(path=c.dolphin,dolphin_home_path=str(home),tmp_home_directory=False,
            slippi_port=c.base_port+self.index,blocking_input=True,polling_mode=True,polling_timeout=.05,
            fullscreen=False,disable_audio=True,gfx_backend='OGL',emulation_speed=c.speed,
            save_replays=c.save_replays,replay_dir=str(self.directory/'replays'),replay_monthly_folders=False)
        codes=home/'GameSettings/GALE01r2.ini'
        content=codes.read_text().replace('[Gecko_Enabled]','[Gecko_Enabled]\n$Melee Next Three Stocks',1)
        codes.write_text(content+'\n$Melee Next Three Stocks\n043D4A4C 03000A00\n')
        # Small native windows keep parallel play inspectable without large GPU surfaces.
        import configparser
        ini=Path(self.console._get_dolphin_config_path())/'Dolphin.ini'
        settings=configparser.ConfigParser(strict=False);settings.read(ini)
        if not settings.has_section('Display'):settings.add_section('Display')
        for key,value in {'RenderWindowWidth':400,'RenderWindowHeight':300,'RenderWindowXPos':(self.index%3)*410,
                          'RenderWindowYPos':40+(self.index//3)*330}.items():settings.set('Display',key,str(value))
        with ini.open('w') as f:settings.write(f)
        self.controllers=[melee.Controller(self.console,port=p) for p in (1,2)]
        self.console.run(iso_path=c.iso)
        atomic_json(self.directory/'dolphin.json',{'pid':self.console._process.pid,'port':c.base_port+self.index})
        deadline=time.monotonic()+60
        for pad in self.controllers:
            while True:
                self.control();self.alive()
                try:
                    fd=os.open(pad.pipe_path,os.O_WRONLY|os.O_NONBLOCK)
                    pad.pipe=os.fdopen(fd,'w');self.console.controllers.append(pad);break
                except OSError as exc:
                    if exc.errno not in (errno.ENXIO,errno.ENOENT):raise
                    if time.monotonic()>deadline:raise TimeoutError('Controller connection timed out')
                    time.sleep(.05)
        from melee.slippstream import SlippstreamClient
        while not self.console.connect():
            self.control();self.alive()
            if time.monotonic()>deadline:raise TimeoutError('Slippi connection timed out')
            self.console._slippstream.shutdown()
            self.console._slippstream=SlippstreamClient(self.console.slippi_address,self.console.slippi_port)
            time.sleep(.2)
    def alive(self):
        if self.console._process.poll() is not None:raise RuntimeError(f'Dolphin exited ({self.console._process.returncode})')
    def next_frame(self):
        deadline=time.monotonic()+20
        while True:
            self.control();self.alive()
            state=self.console.step()
            if state is not None:return state
            if time.monotonic()>deadline:raise TimeoutError('No game frames for 20 seconds')
            time.sleep(.001)
    def reset(self,opponent,level):
        self.opponent=opponent;self.level=level
        start=time.monotonic();self.telemetry(phase='starting' if self.console is None else 'resetting')
        if self.console is None:self.launch()
        helpers=[melee.MenuHelper(),melee.MenuHelper()]
        exiting=self.state is not None and self.state.menu_state==melee.Menu.IN_GAME
        deadline=time.monotonic()+75;last_nudge=time.monotonic();nudge=0;tick=0;candidate=None;settled=0
        while time.monotonic()<deadline:
            g=self.next_frame();tick+=1
            ingame=g.menu_state==melee.Menu.IN_GAME and 1 in g.players and 2 in g.players
            if exiting:
                for pad in self.controllers:pad.release_all()
                if ingame:
                    if tick%2:
                        for name in ('BUTTON_L','BUTTON_R','BUTTON_A','BUTTON_START'):self.controllers[0].press_button(melee.Button[name])
                    self.state=g;continue
                exiting=False
            if ingame and g.frame>=0:
                for pad in self.controllers:pad.release_all()
                if candidate is None:
                    if [g.players[p].stock for p in (1,2)] != [3,3]:raise RuntimeError('Start did not have three stocks each')
                    if g.stage!=melee.Stage[self.config.stage]:raise RuntimeError('Incorrect stage')
                    if g.players[1].character!=melee.Character[self.config.character] or g.players[2].character!=melee.Character[opponent]:
                        raise RuntimeError('Incorrect characters')
                    candidate=g
                settled+=1
                if settled<30:continue
                actual=int(g.players[2].cpu_level)
                if actual!=level:raise RuntimeError(f'CPU difficulty mismatch: requested {level}, observed {actual}')
                self.state=g;self.started_frame=g.frame;self.episode_return=0.;self.history=History();self.reward=Reward()
                self.telemetry(phase='playing',opponent=opponent,cpu_level=actual,reset_seconds=round(time.monotonic()-start,3),players=players(g))
                return self.history.push(g)
            self.state=g
            self.telemetry(phase='resetting',menu=g.menu_state.name,reset_seconds=round(time.monotonic()-start,2))
            if nudge:
                for pad in self.controllers:
                    pad.release_all();pad.press_button(melee.Button.BUTTON_B);pad.tilt_analog(melee.Button.BUTTON_MAIN,.5,0.)
                nudge-=1;continue
            if g.menu_state==melee.Menu.CHARACTER_SELECT and time.monotonic()-last_nudge>10:
                nudge=24;last_nudge=time.monotonic();helpers=[melee.MenuHelper(),melee.MenuHelper()];continue
            p2=g.players.get(2)
            target=melee.Character['ZELDA' if opponent=='SHEIK' else opponent]
            ready=p2 is not None and p2.coin_down and p2.character==target and p2.cpu_level==level
            for i,pad in enumerate(self.controllers):
                helpers[i].menu_helper_simple(g,pad,melee.Character[self.config.character] if i==0 else melee.Character[opponent],
                    melee.Stage[self.config.stage],cpu_level=0 if i==0 else level,costume=i,
                    autostart=i==0 and (ready or g.menu_state==melee.Menu.STAGE_SELECT))
        raise TimeoutError('Match setup deadline exceeded')
    def step(self,action):
        reward=0.;result=None;frames=0
        for _ in range(self.config.action_frames):
            apply(self.controllers[0],action);self.controllers[1].release_all()
            g=self.next_frame()
            if g.menu_state!=melee.Menu.IN_GAME or 1 not in g.players or 2 not in g.players:
                raise RuntimeError('Game left the match before a decisive result')
            delta=g.frame-self.state.frame
            if delta!=1:raise RuntimeError(f'Input/state continuity lost: frame gap {delta}')
            r,result=self.reward.step(self.state,g);reward+=r;frames+=1;self.state=g;self.history.push(g)
            if result:break
            if g.frame-self.started_frame>=self.config.max_frames:
                result='timeout';reward-=2.;break
        self.episode_return+=reward
        info=dict(result=result,frames=frames,players=players(self.state),opponent=self.opponent,cpu_level=self.level,
                  episode_frames=int(self.state.frame-self.started_frame),episode_return=float(self.episode_return))
        self.telemetry(phase='playing',players=info['players'],game_frame=int(self.state.frame))
        return self.history.get(),reward,result in ('win','loss','draw'),result=='timeout',info
    def close(self):
        if self.console:
            try:self.console.stop()
            except Exception:
                process=getattr(self.console,'_process',None)
                if process and process.poll() is None:process.kill();process.wait(timeout=5)
        for pad in self.controllers:
            if getattr(pad,'pipe',None):
                try:pad.pipe.close()
                except OSError:pass
        self.console=None;self.state=None;self.controllers=[]

class SyntheticGame:
    """Pipeline fixture. Has no Melee physics or claims about playing strength."""
    def __init__(self,config,directory,index=0,control=lambda:None,telemetry=lambda **kw:None):
        self.config=config;self.index=index;self.control=control;self.telemetry=telemetry;self.counter=0
    def reset(self,opponent,level):
        if self.index==self.config.mock_slow_worker:
            until=time.monotonic()+self.config.mock_slow_seconds
            while time.monotonic()<until:self.control();time.sleep(.01)
        self.opponent=opponent;self.level=level;self.counter=0;self.telemetry(phase='playing',opponent=opponent,cpu_level=level)
        return np.zeros(OBS,np.float32)
    def step(self,action):
        self.control();time.sleep(self.config.mock_delay);self.counter+=1
        done=self.counter%80==0
        return np.zeros(OBS,np.float32),float(action[2])*.01,done,False,dict(result='loss' if done else None,frames=self.config.action_frames,
            opponent=self.opponent,cpu_level=self.level,players={},episode_frames=self.counter*self.config.action_frames,episode_return=0.)
    def close(self):pass

def game_class(config):return DolphinGame if config.backend=='dolphin' else SyntheticGame
