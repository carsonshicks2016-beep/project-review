import os
import errno
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
import melee
from .config import ROOT, Config
from .actions import ACTIONS, apply, frames_for, vocabulary
from . import controller as pad
from . import layout
from .state import OBS_SIZE, History, in_control, reward, outcome, snapshot

# Normal servicing between steps advances a handful of frames; a slot parked behind
# another slot's reset advances hundreds. Anything past this is a stall, not jitter.
GAP_TOLERANCE = 8

# libmelee sets a CPU's level by walking the cursor onto the level slider and dragging
# it. That grab sometimes fails, and when it does the selection never reads as ready:
# menu_helper_simple loops, the reset runs to its hard deadline, and the slot relaunches
# Dolphin -- six minutes with every other slot idle, and it comes back to an empty
# character select that makes the next reset harder still. Dropping the coin with B puts
# the helper back through the full pick, which is the path that works.
# Every second a slot spends stuck here freezes EVERY OTHER SLOT, because stepping is
# vectorised and blocks on the slowest. So the recovery has to fire early and often:
# waiting 12s before the first nudge, with a 90s setup deadline behind it, meant one
# wedged character select could cost the whole farm a minute and a half. Measured on an
# 8-slot run, all eight slots logged a stall at CHARACTER_SELECT.
# Held above 8s because ordinary menu navigation takes about six: nudging sooner drops
# the coin on resets that were about to succeed and makes them longer, which is the
# opposite of the goal. The gain here comes from more attempts inside the same deadline,
# not from a hair trigger -- 9s x 8 tries fits the 90s budget where 12s x 4 wasted half
# of it waiting.
CSS_NUDGE_SECONDS = 9
CSS_NUDGE_LIMIT = 8
# Frames to hold the cursor upward during a nudge. Four was enough to drop the coin and
# nowhere near enough to move the cursor off the player panel, which is the state that
# actually wedges. The cursor travels a few units a frame, and the panel is ~10 below the
# character grid.
CSS_HOME_FRAMES = 24
# How long to wait for the CPU level to read back before starting anyway.
#
# Raised to 6.0 to give the slider longer to take, then reverted: the failure is
# per-slot and structural, so on a slot that will never set the level this is six
# seconds of EVERY OTHER SLOT frozen at pump speed, buying nothing. Vectorised stepping
# blocks on the slowest slot, so a reset is not local -- it costs the whole farm. The
# level_mismatches counter now surfaces the problem without paying for it in wall time.
CSS_LEVEL_PATIENCE = 2.0
# A reset longer than this is no longer routine; publish where it is stuck.
RESET_REPORT_SECONDS = 12.0
# Menu retries are cheap; emulator relaunches are not. And the level read needs a few
# frames in game before it can be trusted.
LEVEL_RETRIES = 4
LEVEL_SETTLE_FRAMES = 30

# An air-dodge while airborne and offstage enters helpless fall, which ends the stock.
# Flip to True only alongside a metric showing the policy conditions on offstage state.
ALLOW_OFFSTAGE_AIRDODGE = False

# A stock lost with no damage and no hitstun inside this window was not taken by the
# opponent. 45 frames is three quarters of a second -- long enough to cover the travel
# from the last hit, short enough that a genuine KO is never counted as a self-destruct.
SELF_DESTRUCT_WINDOW = 45

# Fox and Falco's up-B charges for about 42 frames and then travels in whatever direction
# the stick points at the end of it. At one decision per frame the policy re-picks that
# angle 42 separate times inside a single charge -- measured spread was 150 degrees, so
# the launch direction is a coin flip. These two rules make the charge behave like the
# one decision the game treats it as, and stop a recovery B press coming out as a laser.
#
# Measured on the Sept-14 Fox at CPU 5, over the 21 launches it died from: it never
# attempted a special at all on 17 of them, and of the B presses it did make offstage
# only 32% were aimed up -- 34% were side-B, which is horizontal-only and cannot recover
# from below the ledge.
FIREFOX_CHARGE = ('SWORD_DANCE_3_LOW', 'SWORD_DANCE_3_MID', 'SWORD_DANCE_3_HIGH',
                  'SWORD_DANCE_3_LOW_AIR', 'SWORD_DANCE_3_MID_AIR', 'SWORD_DANCE_3_HIGH_AIR')
FIREFOX_CHARACTERS = ('FOX', 'FALCO')
# Aim a recovery B press upward only when it is the last option: below the stage surface
# and already outside it. Above that line a side-B back onto the stage is real recovery.
RECOVERY_AIM_ASSIST = True
RECOVERY_AIM_BELOW_Y = 0.0


class Stopped(Exception):
    pass


class LevelNotSet(RuntimeError):
    """The match began with the wrong CPU difficulty.

    Distinct from the connection failures reset() recovers from, because the emulator is
    perfectly healthy -- only the menu went wrong. Tearing Dolphin down and relaunching it
    costs minutes; walking the menus again costs seconds."""
    pass

class MeleeEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, config: Config, run_dir=None, control=None, telemetry=None, window=None, human_input=None, opponent_checkpoint=None, patience=1.0):
        # Booting and menu navigation slow down as emulators contend for the machine;
        # deadlines tuned for one emulator make ten of them fail by the clock alone.
        self.patience=max(1.0,float(patience))
        self.human_input=human_input
        self.opponent_checkpoint=opponent_checkpoint
        self.frozen_opponent=None
        self.config=config
        self.window=window
        self.run_dir=Path(run_dir or ROOT / '.runtime')
        self.run_dir.mkdir(parents=True,exist_ok=True)
        self.control=control or (lambda:None)
        self.telemetry=telemetry or (lambda **kw:None)
        # 'controller' chooses each axis of the pad independently, so every
        # combination of stick angle, C-stick, buttons and shield depth is reachable.
        self.factored=getattr(config,'action_set','legacy')=='controller'
        self.actions=[] if self.factored else vocabulary(config)
        self.action_space=pad.space() if self.factored else gym.spaces.Discrete(len(self.actions))
        self.recoveries=0
        self.observation_space=gym.spaces.Box(-5,5,shape=(OBS_SIZE,),dtype=np.float32)
        self.history=History(); self.console=None; self.state=None; self.ended=True
        self.cpu_level=config.cpu_level
        self.last_status_time=0
        self.episode_return=0
        self.matches=0
        from .execution import ControllerRules
        self.rules = ControllerRules(config.execution_mode)
        from .league import League
        self.league = League(config.league_manifest) if config.league_manifest else None
        self.league_opponent = None

    def launch(self):
        delay = getattr(self, 'launch_delay', 0)
        if delay:
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline:
                self.control()
                time.sleep(min(0.2, max(0.01, deadline - time.monotonic())))
            self.launch_delay = 0
        self.config.validate()
        if self.opponent_checkpoint and self.frozen_opponent is None:
            from .opponent import FrozenOpponent
            self.frozen_opponent=FrozenOpponent(self.opponent_checkpoint,self.config.action_frames,self.config.execution_mode)
        home=self.run_dir/'dolphin-user'
        self.console=melee.Console(path=self.config.dolphin,
            dolphin_home_path=str(home),tmp_home_directory=False,
            blocking_input=True,polling_mode=True,polling_timeout=.05,
            slippi_port=self.config.port,fullscreen=False,disable_audio=not bool(self.human_input),
            gfx_backend='OGL',emulation_speed=self.config.speed,
            save_replays=True,replay_dir=str(self.run_dir/'replays'),
            replay_monthly_folders=False)
        if self.window:
            layout.write_geometry(
                Path(self.console._get_dolphin_config_path())/'Dolphin.ini',self.window)
        # User gecko code runs after the stock default in Slippi's system codes.
        codes=home/'GameSettings/GALE01r2.ini'
        original=codes.read_text()
        original=original.replace('\n$Melee Lab: Three Stocks\n043D4A4C 03000A00\n','').replace('\n$Melee Lab: Three Stocks','')
        text=original.replace('[Gecko_Enabled]',
            '[Gecko_Enabled]\n$Melee Lab: Three Stocks')
        text += '\n$Melee Lab: Three Stocks\n043D4A4C 03000A00\n'
        codes.write_text(text)
        self.controllers=[melee.Controller(self.console,port=p) for p in (1,2)]
        self.helpers=[melee.MenuHelper(),melee.MenuHelper()]
        self.console.run(iso_path=self.config.iso)
        # Open FIFOs non-blocking with a deadline; Controller.connect otherwise hangs forever.
        deadline=time.monotonic()+90*self.patience
        for controller in self.controllers:
            while True:
                self.control()
                if self.console._process.poll() is not None:
                    raise RuntimeError('Dolphin exited during startup. See emulator.log.')
                try:
                    fd=os.open(controller.pipe_path,os.O_WRONLY|os.O_NONBLOCK)
                    os.set_blocking(fd,True)
                    controller.pipe=os.fdopen(fd,'w')
                    self.console.controllers.append(controller)
                    break
                except OSError as exc:
                    if exc.errno != errno.ENXIO: raise
                    if time.monotonic()>deadline:
                        raise TimeoutError(f'Dolphin did not open its controller pipes within {90*self.patience:.0f} seconds.')
                    time.sleep(.05)
        from melee.slippstream import SlippstreamClient
        deadline=time.monotonic()+90*self.patience
        while True:
            self.control()
            if self.console._process.poll() is not None:
                raise RuntimeError('Dolphin exited during startup. See emulator.log.')
            if self.console.connect():
                break
            if time.monotonic()>deadline:
                raise RuntimeError(f'Could not connect to Slippi on the configured port within {90*self.patience:.0f} seconds.')
            self.console._slippstream = SlippstreamClient(self.console.slippi_address, self.console.slippi_port)
            time.sleep(.5)

    def _frame(self):
        deadline=time.monotonic()+30*self.patience
        while True:
            self.control()
            if self.console._process.poll() is not None:
                raise RuntimeError('Dolphin exited unexpectedly.')
            state=self.console.step()
            if state is not None:
                return state
            if time.monotonic()>deadline:
                raise TimeoutError(f'No frames from Dolphin for {30*self.patience:.0f} seconds.')
            time.sleep(.001)

    def pump(self,frames=2):
        """Service Dolphin during CPU-heavy work. libmelee expects step() every frame;
        a multi-second gap desynchronises the Slippi stream and the run never recovers.

        Only an ENCODABLE frame is adopted as the current state. Pumping across the menu
        transition at the end of a match otherwise left self.state on a frame with no
        players, and the next step() encoded it: KeyError 1 out of state.encode, three
        quarters of the way through a training run. Servicing the stream is this method's
        job; replacing a usable observation with an unusable one never was. Keeping the
        last in-game frame is also what the KO-detection and 'exiting' checks want."""
        if self.console is None: return
        for _ in range(frames):
            if self.console._process.poll() is not None: return
            g=self.console.step()
            if g is None: continue
            if g.menu_state==melee.Menu.IN_GAME and 1 in g.players and 2 in g.players:
                self.state=g

    def reset(self,seed=None,options=None):
        menu_retries=0; attempt=-1
        while True:
            attempt+=1
            if attempt>=3+LEVEL_RETRIES: raise RuntimeError('Reset exhausted every attempt.')
            self._level_retries_left=max(0,LEVEL_RETRIES-1-menu_retries)
            try: return self._reset_once(seed=seed,options=options)
            except LevelNotSet as exc:
                # Leave the match and pick again. _reset_once sees an in-game state and
                # runs its own exit path, so this costs a few seconds, not a relaunch.
                menu_retries+=1
                self.telemetry(recovery_reason=f'{exc} (menu retry {menu_retries}/{LEVEL_RETRIES})')
                if menu_retries>=LEVEL_RETRIES: raise
                continue
            except (TimeoutError,BrokenPipeError,ConnectionError,RuntimeError) as exc:
                if self.recoveries>=12 or attempt-menu_retries>=2: raise
                self.recoveries+=1
                self.telemetry(connection='recovering',recoveries=self.recoveries,recovery_reason=str(exc))
                self.close(); self.state=None; self.ended=True
                deadline=time.monotonic()+2
                while time.monotonic()<deadline:
                    self.control(); time.sleep(.05)

    def _reset_once(self,seed=None,options=None):
        reset_started=time.monotonic()
        super().reset(seed=seed if seed is not None else self.config.seed if self.console is None else None)
        if self.league:
            entry = self.league.sample(self.np_random)
            self.league_opponent = entry
            self.config.opponent = entry['character']
            checkpoint = entry.get('checkpoint')
            if checkpoint != self.opponent_checkpoint:
                from .opponent import FrozenOpponent
                self.frozen_opponent = FrozenOpponent(checkpoint, self.config.action_frames, self.config.execution_mode) if checkpoint else None
            self.opponent_checkpoint = checkpoint
            self.cpu_level = entry.get('level', 1)
        if self.config.randomize_opponent and not self.league and not self.human_input and not self.opponent_checkpoint:
            from .slot import COMPETITIVE_ROSTER
            self.config.opponent=str(self.np_random.choice(COMPETITIVE_ROSTER))
        if self.console is None: self.launch()
        # Leave the previous match, including truncated episodes; never train across menus.
        deadline=time.monotonic()+90*self.patience
        setup_level=self.cpu_level
        exiting=self.state is not None and self.state.menu_state==melee.Menu.IN_GAME
        tick=0
        stuck_since=None; nudges=0; nudge_frames=0
        self.helpers=[melee.MenuHelper(),melee.MenuHelper()]
        while True:
            g=self._frame(); tick+=1
            if nudge_frames>0:
                # Drop the coin AND drive the cursor up, off the player panel.
                #
                # libmelee toggles HMN->CPU by walking the cursor to the controller-type
                # box at y=-2.2 with a wiggleroom of 1 and mashing A every other frame.
                # The stock counter lives in that same panel: when the cursor lands
                # slightly off, those presses increment stocks instead of toggling the
                # type, so the level never becomes settable. Observed live -- the counter
                # climbed to x9, x10, x11 while the slot sat wedged for three minutes.
                # Pressing B alone re-entered the same cursor position every time, which
                # is why eight nudges changed nothing.
                #
                # Drive it DOWN, not up. libmelee guards its entire CPU-level block with
                #
                #   if use_cpu and correct_character and (coin_down or cursor_y < 0) \
                #       and (cpu_level != ai_state.cpu_level) or is_holding_cpu_slider:
                #
                # and `and` binds tighter than `or`, so the level logic only runs when the
                # coin is down OR the cursor is below y=0. This nudge presses B, which
                # REMOVES the coin -- so an upward cursor put the slot in the one state
                # where libmelee never even attempts the slider. Measured: two wedged
                # slots both sat at cursor (-31.0, 1.2), holding_cpu_slider False, having
                # spent all eight nudges. The controller-type box it needs is at y=-2.2.
                for c in self.controllers:
                    c.release_all()
                    c.press_button(melee.Button.BUTTON_B)
                    c.tilt_analog(melee.Button.BUTTON_MAIN,.5,0.)
                nudge_frames-=1; self.state=g
                continue
            if time.monotonic()>deadline:
                raise TimeoutError(f'Match setup stalled at {g.menu_state.name}; target={self.config.opponent}, CPU={self.cpu_level}.')
            # A reset that outlives a normal one is the thing that starves every other
            # slot, so report where it is rather than going quiet until it succeeds.
            waited=time.monotonic()-reset_started
            if waited>RESET_REPORT_SECONDS and time.monotonic()-getattr(self,'_reset_reported',0)>1.0:
                self._reset_reported=time.monotonic()
                cursor=getattr(g.players.get(1),'cursor',None) if 1 in g.players else None
                holding=bool(getattr(g.players.get(1),'is_holding_cpu_slider',False)) if 1 in g.players else False
                self.telemetry(connection='resetting',reset_phase=g.menu_state.name,
                               reset_waited=round(waited,1),reset_nudges=nudges,
                               reset_cursor=[round(float(cursor.x),1),round(float(cursor.y),1)] if cursor is not None else None,
                               holding_cpu_slider=holding)
            ingame=g.menu_state==melee.Menu.IN_GAME
            if exiting:
                if ingame:
                    for c in self.controllers: c.release_all()
                    if tick%2:
                        for button in ('BUTTON_L','BUTTON_R','BUTTON_A','BUTTON_START'):
                            self.controllers[0].press_button(melee.Button[button])
                    continue
                exiting=False
            if ingame and 1 in g.players and 2 in g.players:
                for c in self.controllers: c.release_all()
                if g.frame<0: continue
                stocks=[int(g.players[p].stock) for p in (1,2)]
                if stocks != [3,3]:
                    raise RuntimeError(f'Expected a fresh three-stock match, got {stocks}. Refusing to mislabel training.')
                if g.stage != melee.Stage[self.config.stage]:
                    raise RuntimeError('Unexpected stage at match start.')
                if g.players[1].character != melee.Character[self.config.character] or g.players[2].character != melee.Character[self.config.opponent]:
                    raise RuntimeError('Unexpected characters at match start.')
                # The stage and characters are asserted, and the CPU level was not -- so a
                # level the helper failed to set slipped through as a real match. Measured
                # over one training run, 10% of matches were played against a level-0 CPU:
                # an opponent that does not act, 40% longer matches, and roughly 14% of
                # training frames spent learning that nobody fights back. The curriculum
                # credits by actual level so promotions were not inflated, but the frames
                # were still wasted. Refuse it here and let reset() try again, exactly as
                # a wrong stage or character already does.
                wanted_cpu=0 if self.human_input or self.opponent_checkpoint else setup_level
                if 'cpu_level' in dir(g.players[2]):
                    # The value is not dependable on the first in-game frame, so give it a
                    # moment before calling it wrong -- an assertion that fires on a
                    # transient read costs a whole match to a problem that was not there.
                    for _ in range(LEVEL_SETTLE_FRAMES):
                        if int(g.players[2].cpu_level)==wanted_cpu: break
                        g=self._frame()
                        if g.menu_state!=melee.Menu.IN_GAME or 2 not in g.players: break
                    got=int(getattr(g.players[2],'cpu_level',wanted_cpu)) if 2 in g.players else wanted_cpu
                    if got != wanted_cpu:
                        if self.config.strict_evaluation:
                            raise LevelNotSet(f'Evaluation requires CPU {wanted_cpu}, observed {got}.')
                        # Retry the menus while retries remain -- some of these are
                        # transient. But the failure is PER-SLOT and persistent: in one
                        # run env1 and env2 set the level every time while env0 and env3
                        # never did, four retries each. Refusing outright turns a 10%
                        # data-quality problem into a dead run, which is a worse trade, so
                        # once the retries are spent the match is played and counted. The
                        # curriculum already credits each match to the level it was
                        # actually played at, so promotions are not inflated by these.
                        if getattr(self,'_level_retries_left',0)>0:
                            raise LevelNotSet(f'CPU level did not take: asked for {wanted_cpu}, got {got}.')
                        self.level_mismatches=getattr(self,'level_mismatches',0)+1
                        self.telemetry(level_mismatches=self.level_mismatches,
                                       recovery_reason=f'playing at CPU {got}, asked for {wanted_cpu}; '
                                                       f'menu retries exhausted on this slot')
                self.recoveries=0
                actual_cpu=int(g.players[2].cpu_level) if (2 in g.players and getattr(g.players[2],'cpu_level',None) is not None) else setup_level
                self.state=g; self.ended=False; self.episode_return=0; self.match_cpu=0 if self.human_input or self.opponent_checkpoint else actual_cpu
                self.start_frame=g.frame; self.observed_frame=int(g.frame)
                self.stock_losses=0; self.self_destructs=0; self.last_hit_frame=int(g.frame); self.firefox_latch=None
                self.controlled_since_hit=True
                self.rules.reset()
                if self.frozen_opponent: self.frozen_opponent.reset(g)
                # Every slot waits out every other slot's reset, so this number is the
                # single biggest lever on parallel throughput. Measure it, don't guess.
                seconds=time.monotonic()-reset_started
                self.reset_seconds=round(seconds,2)
                self.reset_total=getattr(self,'reset_total',0.)+seconds
                self.reset_count=getattr(self,'reset_count',0)+1
                self.telemetry(connection='playing',cpu_level=self.match_cpu,players=snapshot(g),
                               reset_seconds=self.reset_seconds,
                               reset_seconds_mean=round(self.reset_total/self.reset_count,2))
                return self.history.reset(g),{'starting_stocks':stocks,'cpu_level':self.cpu_level}
            # CPU helper must finish choosing its character and difficulty before P1 starts.
            p1_ready=(1 in g.players and g.players[1].coin_down and g.players[1].character==melee.Character[self.config.character])
            p2_coin=(2 in g.players and g.players[2].coin_down)
            p2_char=(2 in g.players and g.players[2].character==melee.Character['ZELDA' if self.config.opponent=='SHEIK' else self.config.opponent])
            p2_cpu=(2 in g.players and g.players[2].cpu_level==(0 if self.human_input or self.opponent_checkpoint else setup_level))
            banner=(getattr(g,'ready_to_start',1)==0)
            time_stuck=(time.monotonic()-stuck_since) if stuck_since is not None else 0.
            # Waiting on the level costs seconds; starting without it costs the whole match.
            ready=(p2_coin and p2_char and (p2_cpu or banner or (p1_ready and time_stuck>CSS_LEVEL_PATIENCE)))
            if g.menu_state==melee.Menu.CHARACTER_SELECT and not ready:
                if stuck_since is None: stuck_since=time.monotonic()
                elif time.monotonic()-stuck_since>CSS_NUDGE_SECONDS and nudges<CSS_NUDGE_LIMIT:
                    nudges+=1; stuck_since=None; nudge_frames=CSS_HOME_FRAMES
                    self.helpers=[melee.MenuHelper(),melee.MenuHelper()]
                    cursor=getattr(g.players.get(1),'cursor',None) if 1 in g.players else None
                    where=f' cursor=({cursor.x:.1f},{cursor.y:.1f})' if cursor is not None else ''
                    self.telemetry(recovery_reason=f'character select would not settle; homed the cursor ({nudges}/{CSS_NUDGE_LIMIT}){where}')
                    self.css_nudges=getattr(self,'css_nudges',0)+1
                    continue
            else: stuck_since=None
            for i,c in enumerate(self.controllers):
                self.helpers[i].menu_helper_simple(g,c,
                    melee.Character[self.config.character if i==0 else self.config.opponent],
                    melee.Stage[self.config.stage],cpu_level=0 if i==0 or self.human_input or self.opponent_checkpoint else setup_level,
                    costume=i,autostart=(i==0 and (ready or g.menu_state==melee.Menu.STAGE_SELECT)))
            if (ready or banner) and p1_ready and p2_coin and g.menu_state==melee.Menu.CHARACTER_SELECT:
                if tick%2==0:
                    self.controllers[0].press_button(melee.Button.BUTTON_START)
            self.state=g

    def step(self,action):
        try: return self._step_once(action)
        except (TimeoutError,BrokenPipeError,ConnectionError) as exc:
            if self.recoveries>=12 or self.state is None: raise
            self.recoveries+=1
            self.telemetry(connection='recovering',recoveries=self.recoveries,recovery_reason=str(exc))
            info=dict(cpu_level=self.match_cpu,frame=int(self.state.frame),players=snapshot(self.state),
                      action='Connection interrupted',result='interrupted',starting_stocks=[3,3],
                      episode={'r':self.episode_return,'l':max(0,int(self.state.frame-self.start_frame))},
                      TimeLimit_truncated=True)
            obs=self.history.get()
            self.close(); self.state=None; self.ended=True
            return obs,0.,False,True,info

    def _sanitize_action(self, action):
        return self.rules.apply(action, self.state)

    def _note_stock_loss(self,previous,current):
        """Record whether each lost stock was taken by the opponent or thrown away.

        The first version of this asked only whether damage landed in the 45 frames
        before the death, and it was wrong in the case that matters most: the agent is
        launched offstage, survives the hit, flies out, and dies half a second later
        having taken no further damage. Measured at CPU 5 that single pattern was 19 of
        27 offstage deaths, every one of them mislabelled a self-destruct -- which hid a
        real recovery failure behind a number that said the agent was throwing stocks
        away on its own.

        A stock belongs to the opponent if a hit started the sequence and the agent never
        got back under its own command before dying. It is a self-destruct only when the
        agent was safe and in control, and then lost the stock anyway."""
        before,after = previous.players[1],current.players[1]
        frame = int(current.frame)
        struck = (float(after.percent) > float(before.percent)
                  or int(getattr(after,'hitstun_frames_left',0)) > 0)
        if struck:
            self.last_hit_frame = frame
            self.controlled_since_hit = False
        elif in_control(after):
            self.controlled_since_hit = True
        if int(after.stock) < int(before.stock):
            self.stock_losses = getattr(self,'stock_losses',0)+1
            recently_struck = frame-getattr(self,'last_hit_frame',frame) <= SELF_DESTRUCT_WINDOW
            never_recovered = not getattr(self,'controlled_since_hit',True)
            if not (recently_struck or never_recovered):
                self.self_destructs = getattr(self,'self_destructs',0)+1
            # A respawn is neither a hit nor a recovery; the next stock starts clean.
            self.last_hit_frame = frame
            self.controlled_since_hit = True

    def _step_once(self,action):
        if self.ended: raise RuntimeError('Call reset before stepping a finished match.')
        if self.factored:
            if not self.action_space.contains(np.asarray(action,dtype=np.int64)):
                raise ValueError('Invalid controller action.')
        elif not self.action_space.contains(int(action)): raise ValueError('Invalid controller action.')
        total=0.; result=None; truncated=False
        # A macro always runs to completion: action_frames sets the floor, not the
        # ceiling, so a 5-frame wavedash is not truncated into a bare jump.
        window=self.config.action_frames if self.factored else max(
            self.config.action_frames,frames_for(action,self.actions))
        applied_action = self._sanitize_action(action) if self.factored else action
        for frame in range(window):
            if self.factored: pad.apply(self.controllers[0],applied_action,frame)
            else: apply(self.controllers[0],action,frame,self.actions)
            if self.human_input: self.human_input.apply(self.controllers[1])
            elif self.frozen_opponent: self.frozen_opponent.apply(self.controllers[1],self.state)
            else: self.controllers[1].release_all()
            g=self._frame()
            if g.menu_state!=melee.Menu.IN_GAME or 1 not in g.players or 2 not in g.players:
                # If a decisive KO was already observed in game, honor it.
                # Never evaluate outcome() on a menu frame where stocks default to 0.
                if self.state and self.state.menu_state==melee.Menu.IN_GAME and self.state.frame>0 and 1 in self.state.players and 2 in self.state.players:
                    s1,s2=int(self.state.players[1].stock),int(self.state.players[2].stock)
                    if s1==0 and s2>0: result='loss'
                    elif s2==0 and s1>0: result='win'
                    else: truncated=True; result='interrupted'
                else:
                    truncated=True; result='interrupted'
                break
            if g.frame <= self.state.frame:
                truncated=True; result='unexpected_restart'; break
            self._note_stock_loss(self.state,g)
            total+=reward(self.state,g)
            self.state=g
            result=outcome(g)
            if result: break
            if g.frame-self.start_frame>=self.config.max_frames:
                truncated=True; result='timeout'; break
        terminated=result in ('win','loss','draw')
        self.ended=terminated or truncated
        self.episode_return+=total
        # The background pump advances Dolphin while this slot waits on the trainer --
        # during another slot's menu navigation that is hundreds of game frames. Pairing
        # the frame after such a gap with the one from before it produces a stacked
        # observation whose implied velocities are nonsense, and the policy never saw
        # anything like it in training. Start the stack fresh instead.
        gap=int(self.state.frame)-getattr(self,'observed_frame',int(self.state.frame))
        if gap>window+GAP_TOLERANCE:
            self.stalled_observations=getattr(self,'stalled_observations',0)+1
            obs=self.history.reset(self.state)
        else:
            obs=self.history.push(self.state)
        self.observed_frame=int(self.state.frame)
        info={'cpu_level':self.match_cpu,'frame':int(self.state.frame),
              'players':snapshot(self.state),
              'action':pad.describe_vector(action) if self.factored else self.actions[int(action)].name,
              'opponent_type':'human' if self.human_input else 'policy' if self.opponent_checkpoint else 'cpu',
              'human_connected':bool(self.human_input and self.human_input.connected),
              'stalled_observations':getattr(self,'stalled_observations',0)}
        info['execution'] = self.rules.report()
        if self.league_opponent:
            info['league_opponent'] = {k:v for k,v in self.league_opponent.items() if k != 'checkpoint'}
        if self.frozen_opponent:
            info['opponent_execution'] = self.frozen_opponent.rules.report()
        if self.ended:
            self.matches+=1
            losses=getattr(self,'stock_losses',0); sd=getattr(self,'self_destructs',0)
            info.update(result=result,starting_stocks=[3,3],
                        episode={'r':self.episode_return,'l':int(self.state.frame-self.start_frame)},
                        stock_losses=losses,self_destructs=sd,
                        self_destruct_rate=round(sd/losses,3) if losses else None,
                        TimeLimit_truncated=truncated)
        now=time.monotonic()
        if self.ended or now-self.last_status_time>.25:
            self.telemetry(**info); self.last_status_time=now
        return obs,total,terminated,truncated,info

    def close(self):
        if self.console is not None:
            for c in getattr(self,'controllers',[]):
                try: c.disconnect()
                except Exception: pass
            proc = getattr(self.console, '_process', None)
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception: pass
            try: self.console.stop()
            except Exception: pass
            self.console=None
