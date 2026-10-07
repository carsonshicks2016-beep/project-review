import argparse
from .config import Config

def main():
    parser=argparse.ArgumentParser(description='Melee Lab: train a real-game neural agent')
    sub=parser.add_subparsers(dest='command',required=True)
    serve=sub.add_parser('serve'); serve.add_argument('--port',type=int,default=8765)
    sub.add_parser('doctor')
    sub.add_parser('probe')
    args=parser.parse_args()
    if args.command=='serve':
        import json
        import uvicorn
        from .config import ROOT
        # The training worker opens the companion window and has to know where the
        # dashboard is listening; nothing else in the process tree knows the port.
        runtime=ROOT/'.runtime'; runtime.mkdir(exist_ok=True)
        (runtime/'server.json').write_text(json.dumps({'port':args.port}))
        uvicorn.run('melee_lab.server:app',host='127.0.0.1',port=args.port,reload=True)
    elif args.command=='doctor':
        from importlib.metadata import version
        config=Config.load(); config.validate()
        print(f'Paths ready. melee {version("melee")}, PPO {version("stable-baselines3")}.')
        print(f'{config.character} vs CPU {config.cpu_level} {config.opponent}; 3 stocks, {config.stage}.')
        print('Run probe to verify the actual emulator handshake and match settings.')
    elif args.command=='probe':
        from .probe import main as probe
        probe()

if __name__=='__main__':main()
