# Mission Control

Mission Control is a local dashboard for browsing Python workspaces and opening selected standalone web projects. The project catalog lives in `config/projects.json`; it does not crawl the whole home folder at runtime.

## Run

Use the Desktop launcher, or run `npm start` from this directory. The server listens on `127.0.0.1:9000` by default. Set `PORT` to change the port.

## Project catalog

Each item has a stable `id`, display name, absolute folder path, and optional metadata. Embedded projects can set `type: "embed"` and `staticIndex`. Projects with a launch command use an argument array, for example `"command": ["python3", "app.py"]`; command strings and shell expansion are intentionally unsupported. The server only stops child processes it started itself.

Use `python3 scan_all_projects.py` to print likely project roots at the configured Python workspace. The scanner is read-only and examines only direct child folders.

## Safety and scope

Mission Control binds to loopback, serves only catalogued embedded folders, validates project IDs, and never kills a process based on a port number. It does not install dependencies or modify project folders. Cards without a validated launch command remain available as folder links.
