import argparse
from .server import serve

def main():
    p = argparse.ArgumentParser(description='Hybrid Lab — standalone PPO race/drift dashboard')
    p.add_argument('--port', type=int, default=8766)
    p.add_argument('--runs', help='Separate folder for run artifacts')
    p.add_argument('--open', action='store_true', help='Open your browser')
    args = p.parse_args()
    serve(args.port, args.runs, args.open)

if __name__ == '__main__':
    main()
