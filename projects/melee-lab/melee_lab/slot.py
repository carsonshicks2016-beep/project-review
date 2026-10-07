"""One emulator slot: its own Slippi port, Dolphin home, control polling and telemetry.

Runs inside the environment process. A slot never owns the model, so it reads the
shared control file but never writes a checkpoint -- saving stays with the trainer.
"""
import faulthandler
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
import multiprocessing as mp
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper,VecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import _worker
from .environment import MeleeEnv,Stopped
from .storage import write_json

# How long an idle window has to be before the background pump steps in. Normal
# inter-step gaps are milliseconds and must be left alone: with blocking input the
# handshake is what keeps the game synchronised to the agent's actions. Only a real
# stall -- a PPO update, a checkpoint write, a pause -- needs servicing.
STALL_SECONDS = .5
# The pump exists to keep the Slippi stream from going quiet for multiple seconds, not
# to play the game. Every frame it advances is a frame nobody is acting on, so it runs
# as slowly as that purpose allows: a stalled slot now advances ~25 frames across a
# 6.3s reset instead of ~126.
#
# This is also why a blocked emulator reads FPS 4 on screen while one navigating menus
# reads 60. That contrast looks like a hang and is not one -- but the underlying waste is
# real: vectorised stepping blocks on the slowest slot, so a 6s reset costs 6s on every
# other emulator too. Measured across 6 slots, throughput was 28 fps per slot against a
# possible 60. The cure is a shorter reset or fewer slots, not a faster pump; raising the
# rate here only feeds the policy frames it never chose.
PUMP_INTERVAL = .25


# faulthandler needs its file to outlive the register() call, so keep a reference.
_STACK_FILES=[]


def watch_stacks(path):
    """Dump every thread's stack to `path` on SIGUSR1, and on a hard crash.

    The emulator handshake can wedge with every process asleep and no exception
    raised; without this there is nothing to read but CPU counters and mtimes.
    """
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    handle=open(path,'a',buffering=1)
    _STACK_FILES.append(handle)
    faulthandler.enable(handle)
    faulthandler.register(signal.SIGUSR1,file=handle,all_threads=True,chain=False)
    return handle


class Slot:
    def __init__(self,directory,control_path,index=0):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.control_path=Path(control_path); self.index=index
        self.state={'index':index}; self.last_poll=0
        self.idle=lambda:None
    def publish(self,**kwargs):
        self.state.update(kwargs)
        write_json(self.directory/'status.json',self.state)
    def control(self):
        if time.monotonic()-self.last_poll<.1: return
        self.last_poll=time.monotonic()
        while True:
            command=json.loads(self.control_path.read_text()) if self.control_path.exists() else {}
            if command.get('stop'): raise Stopped()
            if not command.get('pause'): break
            self.idle()
            time.sleep(.15)


class Serviced(gym.Wrapper):
    """Keeps Dolphin serviced while this process waits on the trainer.

    Under SubprocVecEnv the worker blocks in recv() during the parent's optimisation
    pass, so nothing would call console.step() for seconds -- the exact gap that
    desynchronises the Slippi stream. libmelee's console is not thread-safe, so the
    background pump and the env share a lock and never step concurrently.
    """
    def __init__(self,env,slot=None,on_crash=None,exit_on_close=False):
        super().__init__(env)
        self.slot=slot
        self.exit_on_close=exit_on_close
        # Overridable so tests can exercise the crash path without killing pytest.
        self.on_crash=on_crash or (lambda code: os._exit(code))
        self.lock=threading.RLock(); self.stopping=threading.Event()
        self.last_activity=time.monotonic()
        self.thread=threading.Thread(target=self._loop,daemon=True); self.thread.start()
    def _loop(self):
        while not self.stopping.is_set():
            time.sleep(PUMP_INTERVAL)
            if time.monotonic()-self.last_activity<STALL_SECONDS: continue
            if not self.lock.acquire(blocking=False): continue
            try:
                # Release first. Nothing is choosing actions while this slot waits, so
                # without this the pad keeps asserting the last sampled input -- a full
                # stick throw held for the whole of another slot's reset, walking the
                # agent off the stage while the CPU keeps attacking. The pad re-asserts
                # every axis on the next real step, so releasing costs nothing.
                for controller in getattr(self.env,'controllers',[]): controller.release_all()
                self.env.pump(1)
            except Exception: pass
            finally: self.lock.release()
    @property
    def cpu_level(self):
        return self.env.cpu_level

    @cpu_level.setter
    def cpu_level(self,value):
        # gymnasium's Wrapper does not forward attribute writes, so SB3's set_attr
        # would otherwise land on the wrapper and the emulator would never see it.
        self.env.cpu_level=value

    def guard(self,call):
        """Never let an exception reach SB3's worker.

        Its shutdown path hangs forever in multiprocessing's atexit handler joining
        libmelee's slippstream child, so the process never dies, its pipe stays open,
        and the trainer waits in recv() with no error and no traceback. Record what
        happened, tear the emulator down, then exit hard so the parent sees EOF and
        fails the run cleanly.
        """
        try: return call()
        except Stopped:
            # A requested stop is control flow, not a crash: leave quietly and let the
            # trainer's own control poll drive the clean 'stopped' path.
            try: self.env.close()
            except Exception: pass
            self.stopping.set()
            return self.on_crash(0)
        except BaseException as exc:
            detail=traceback.format_exc()
            if self.slot is not None:
                try: self.slot.publish(failed=f'{type(exc).__name__}: {exc}',
                                       traceback=detail[-2000:])
                except Exception: pass
            try: self.env.close()
            except Exception: pass
            sys.stderr.write(f'slot crashed:\n{detail}'); sys.stderr.flush()
            self.stopping.set()
            return self.on_crash(1)

    def step(self,action):
        with self.lock:
            try: return self.guard(lambda: self.env.step(action))
            finally: self.last_activity=time.monotonic()
    def reset(self,**kwargs):
        with self.lock:
            try: return self.guard(lambda: self.env.reset(**kwargs))
            finally: self.last_activity=time.monotonic()
    def close(self):
        self.stopping.set(); self.thread.join(2)
        with self.lock: result=self.env.close()
        if self.exit_on_close:
            # SB3's worker returns into normal interpreter shutdown, where
            # multiprocessing's atexit blocks forever joining libmelee's slippstream
            # child -- the trainer then hangs in vec.close() holding the emulator
            # lock. Everything worth keeping is already on disk, so leave now.
            sys.stderr.flush()
            return self.on_crash(0)
        return result


def reap_emulators(run_dir):
    """Terminate any Dolphin still holding this run's user directory."""
    marker=str(Path(run_dir).resolve())
    try:
        out=subprocess.run(['ps','-A','-o','pid=,command='],capture_output=True,text=True).stdout
    except (PermissionError, OSError):
        try:
            out=subprocess.run(['pgrep','-fl','Slippi Dolphin'],capture_output=True,text=True).stdout
        except (PermissionError, OSError):
            return []
    reaped=[]
    for line in out.splitlines():
        parts=line.strip().split(None,1)
        if len(parts)!=2: continue
        if marker not in parts[1]: continue
        try:
            os.kill(int(parts[0]),signal.SIGKILL); reaped.append(int(parts[0]))
        except (ProcessLookupError,ValueError,PermissionError): pass
    return reaped


def shutdown(vec,timeout=15):
    """Tear down the emulator processes without SB3's unbounded waits.

    SubprocVecEnv.close() first drains pending step results, then joins every child
    with no deadline. When one slot has already exited -- which a requested stop
    makes routine -- the surviving slots sit in recv() waiting for a command the
    trainer will never send, while close() waits on them. That mutual wait left the
    emulators running and the lock held, so no further run could start.
    """
    if vec is None: return []
    vec.waiting=False; vec.closed=True
    for remote in getattr(vec,'remotes',[]):
        try: remote.send(('close',None))
        except (OSError,EOFError,BrokenPipeError,ValueError): pass
    deadline=time.monotonic()+timeout
    processes=list(getattr(vec,'processes',[]))
    for process in processes:
        try: process.join(max(0.1,deadline-time.monotonic()))
        except (OSError,ValueError,AssertionError): pass
    forced=[]
    for process in processes:
        try:
            if process.is_alive(): process.terminate(); forced.append(process)
        except (OSError,ValueError): pass
    if forced:
        time.sleep(1)
        for process in forced:
            try:
                if process.is_alive(): process.kill()
            except (OSError,ValueError): pass
    for remote in getattr(vec,'remotes',[]):
        try: remote.close()
        except Exception: pass
    return [getattr(p,'pid',None) for p in forced]


def reap_workspace_dolphins(root=None):
    """Terminate any orphaned Dolphin processes belonging to this workspace."""
    root_str=str(Path(root or Path(__file__).resolve().parent.parent).resolve())
    try:
        out=subprocess.run(['ps','-A','-o','pid=,command='],capture_output=True,text=True).stdout
    except (PermissionError, OSError):
        try:
            out=subprocess.run(['pgrep','-fl','Slippi Dolphin'],capture_output=True,text=True).stdout
        except (PermissionError, OSError):
            return []
    reaped=[]
    mine=os.getpid()
    vendor=root_str+'/vendor/'
    for line in out.splitlines():
        parts=line.strip().split(None,1)
        if len(parts)!=2: continue
        # Anchor on argv[0]. A substring match over the whole command line matches the
        # worker's own arguments -- which name the workspace -- so this function used
        # to SIGKILL its own caller on the first line of run().
        if not parts[1].startswith(vendor): continue
        try:
            pid=int(parts[0])
            if pid==mine: continue
            os.kill(pid,signal.SIGKILL); reaped.append(pid)
        except (ProcessLookupError,ValueError,PermissionError): pass
    return reaped


COMPETITIVE_ROSTER = ('MARIO', 'MARTH', 'FALCO', 'CPTFALCON', 'PEACH', 'FOX')


def make_env(config,run_dir,index,control_path,stagger=5,window=None,opponent_checkpoint=None,patience=1.0):
    """Build one slot's env. Each gets port+index and its own Dolphin home."""
    def build():
        slot=Slot(Path(run_dir)/f'env{index}',control_path,index)
        watch_stacks(slot.directory/'stacks.log')
        slot_config=replace(config,port=config.port+index,seed=config.seed+index)
        if getattr(config,'randomize_opponent',False) and not opponent_checkpoint:
            # Each slot owns one matchup, and keeps it. Diversity comes from running the
            # roster side by side, so every rollout already contains every character --
            # a stratified sample rather than a multinomial one.
            #
            # Re-drawing per episode as well would undo that AND make every reset
            # expensive: an LRAS exit returns to the character select screen with the
            # previous picks still selected and their coins down, so an unchanged
            # opponent is ready to start on the first frame back, while a changed one
            # has to be walked to with the cursor. Half the wall clock was going into
            # menus nobody needed to visit.
            slot_config=replace(slot_config,opponent=COMPETITIVE_ROSTER[index%len(COMPETITIVE_ROSTER)],
                                randomize_opponent=False)
        env=MeleeEnv(slot_config,slot.directory,slot.control,slot.publish,window=window,opponent_checkpoint=opponent_checkpoint,patience=patience)
        if index and stagger:
            env.launch_delay = index * stagger
        def idle():
            for c in getattr(env,'controllers',[]): c.release_all()
            env.pump(2)
        slot.idle=idle
        return Serviced(env,slot,exit_on_close=True)
    return build


class EmulatorVecEnv(SubprocVecEnv):
    """SubprocVecEnv whose workers are not daemonic.

    libmelee runs its Slippi reader in a process of its own (slippstream.py), and a
    daemonic process may not have children, so SB3's default workers cannot host a
    Melee env at all. Non-daemonic workers can outlive a crashed trainer, so run()
    closes this in its finally block and terminates anything still standing.
    """
    def __init__(self,env_fns,start_method='spawn'):
        self.waiting=False; self.closed=False
        count=len(env_fns)
        ctx=mp.get_context(start_method)
        self.remotes,self.work_remotes=zip(*[ctx.Pipe() for _ in range(count)])
        self.processes=[]
        for work_remote,remote,fn in zip(self.work_remotes,self.remotes,env_fns):
            process=ctx.Process(target=_worker,
                args=(work_remote,remote,CloudpickleWrapper(fn)),daemon=False)
            process.start(); self.processes.append(process); work_remote.close()
        self.remotes[0].send(('get_spaces',None))
        observation_space,action_space=self.remotes[0].recv()
        VecEnv.__init__(self,count,observation_space,action_space)
