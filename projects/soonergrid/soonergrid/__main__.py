"""SoonerGrid command line entry point."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from soonergrid.research.benchmark import ROOT, benchmark


def main():
    parser=argparse.ArgumentParser(description='SoonerGrid | traffic research workspace')
    sub=parser.add_subparsers(dest='command', required=True)
    run=sub.add_parser('benchmark', help='Run matched-condition network experiments')
    run.add_argument('--duration',type=float,default=28800)
    run.add_argument('--dt',type=float,default=5)
    run.add_argument('--workers',type=int,default=2)
    run.add_argument('--scales',type=float,nargs='+',default=[1.0])
    run.add_argument('--shocks',nargs='+',choices=['NOMINAL','SHOCK_COLLISION','SHOCK_THUNDERSTORM','SHOCK_OVERTIME'],default=['NOMINAL','SHOCK_COLLISION'])
    serve=sub.add_parser('serve',help='Open the local research dashboard')
    serve.add_argument('--port',type=int,default=8790)
    sub.add_parser('export',help='Build compact dashboard data and provenance inventory')
    args=parser.parse_args()
    if args.command=='benchmark':
        if args.workers < 1 or any(s <= 0 for s in args.scales): parser.error('Workers and demand scales must be positive')
        print(benchmark(args.duration,args.dt,args.workers,args.scales,args.shocks))
    elif args.command=='export':
        from soonergrid.research.export import export
        print(export())
    else:
        print(f'SoonerGrid: http://127.0.0.1:{args.port}/visualizer/',flush=True)
        server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(ROOT)))
        try: server.serve_forever()
        except KeyboardInterrupt: server.server_close()

if __name__=='__main__': main()
