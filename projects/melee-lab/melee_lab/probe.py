from .config import Config
from .environment import MeleeEnv

def main():
    env=MeleeEnv(Config.load(),telemetry=lambda **kw:print(kw,flush=True))
    try:
        obs,info=env.reset()
        print('MATCH VERIFIED',obs.shape,info,flush=True)
        for _ in range(30): env.step(0)
    finally: env.close()

if __name__=='__main__':main()
