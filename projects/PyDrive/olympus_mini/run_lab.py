import argparse
from olympus_mini.lab.server import run_server, sim_manager

def main():
    parser = argparse.ArgumentParser(description="Olympus Mini Laboratory Dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Web dashboard port (default: 5050)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--task", type=str, default="sprint", choices=["sprint", "hurdle", "vault"], help="Initial discipline")
    args = parser.parse_args()
    
    if args.task != sim_manager.task:
        sim_manager.set_task(args.task)
        
    run_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
