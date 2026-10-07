/* ==========================================================================
   Carson's Coding Control Center (app.js)
   ========================================================================== */

// 1. Projects Database
const PROJECTS = [
  {
    id: "supra-ai-v1",
    name: "Supra AI v1",
    path: "~/Desktop/supra-ai",
    category: "ai-racing",
    desc: "Original neuro-evolution driving sim. Evolve neural networks or train PPO policies with Pacejka 4-wheel physics, twin-turbo spooling, and interactive 2D/3D visualizations.",
    tech: ["Python", "PyTorch", "PyGame", "NEAT", "RL"],
    icon: "🏎️",
    scripts: [
      {
        label: "Live GA Evolution",
        command: "python3 run.py",
        mode: "native",
        args: []
      },
      {
        label: "Train GA (Headless)",
        command: "python3 run.py --train {gens} --pop {pop} --seed {seed}",
        mode: "background",
        args: [
          { name: "gens", label: "Generations", type: "range", min: 50, max: 2000, default: 300 },
          { name: "pop", label: "Population Size", type: "range", min: 10, max: 200, default: 50 },
          { name: "seed", label: "RNG Seed", type: "number", default: 42 }
        ]
      },
      {
        label: "Watch GA Champion",
        command: "python3 run.py --watch",
        mode: "native",
        args: []
      },
      {
        label: "Train PPO RL (Headless)",
        command: "python3 run.py --ppo {timesteps} --workers {workers} --car {car} {recurrent}",
        mode: "background",
        args: [
          { name: "timesteps", label: "Timesteps", type: "range", min: 10000, max: 1000000, step: 10000, default: 100000 },
          { name: "workers", label: "Worker Processes", type: "range", min: 1, max: 16, default: 8 },
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" },
          { name: "recurrent", label: "Recurrent LSTM Policy", type: "checkbox", flag: "--recurrent", default: false }
        ]
      },
      {
        label: "Train PPO Live (GUI)",
        command: "python3 run.py --ppo-live --fresh --workers 4",
        mode: "native",
        args: []
      },
      {
        label: "Watch Trained PPO",
        command: "python3 run.py --watch-ppo",
        mode: "native",
        args: []
      },
      {
        label: "Train Drift Objective",
        command: "python3 run.py --drift {timesteps} --workers {workers} --car {car} {recurrent}",
        mode: "background",
        args: [
          { name: "timesteps", label: "Timesteps", type: "range", min: 10000, max: 1000000, step: 10000, default: 100000 },
          { name: "workers", label: "Worker Processes", type: "range", min: 1, max: 16, default: 8 },
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" },
          { name: "recurrent", label: "Recurrent LSTM Policy", type: "checkbox", flag: "--recurrent", default: true }
        ]
      },
      {
        label: "Watch Drift Policy",
        command: "python3 run.py --watch-drift",
        mode: "native",
        args: []
      },
      {
        label: "Train Gears (PPO v2 - Background)",
        command: "./train_gears.sh {iters} {workers}",
        mode: "background",
        args: [
          { name: "iters", label: "Iterations", type: "number", default: 8000 },
          { name: "workers", label: "Rollout Workers", type: "number", default: 8 }
        ]
      },
      {
        label: "Watch Gears (PPO v2)",
        command: "./watch_gears.sh {submode}",
        mode: "native",
        args: [
          { name: "submode", label: "Watch Target", type: "select", options: ["", "reload", "best"], default: "" }
        ]
      }
    ]
  },
  {
    id: "supra-ai-2",
    name: "Supra AI 2",
    path: "~/Desktop/Supra Ai 2",
    category: "ai-racing",
    desc: "From-scratch RL drift & grip-racing sim: 4-wheel Pacejka physics, raycast vision + live telemetry dashboard, real-time engine-audio synthesis, a NumPy genetic algorithm and PyTorch PPO race/drift policies with a difficulty curriculum and procedural tracks. Hybrid-ready (one mode-conditioned policy).",
    tech: ["Python", "PyTorch", "PyGame", "NumPy", "sounddevice"],
    icon: "🏎️",
    scripts: [
      {
        label: "🕹️ Drive Manually (Keyboard)",
        command: "python3 run.py --drive --car {car} --track {track} {noaudio}",
        mode: "native",
        args: [
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" },
          { name: "track", label: "Track", type: "select", options: ["club", "national", "coast", "sprint", "tech", "oval2", "akina", "pass", "speedbowl", "superspeed", "oval", "random", "touge"], default: "club" },
          { name: "noaudio", label: "Mute Engine Audio", type: "checkbox", flag: "--no-audio", default: false }
        ]
      },
      {
        label: "🛠️ Drive a Generated Track",
        command: "python3 run.py --drive --car {car} --gen {style} --difficulty {difficulty} --length {length}",
        mode: "native",
        args: [
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" },
          { name: "style", label: "Archetype", type: "select", options: ["gp", "technical", "speedway", "touge"], default: "gp" },
          { name: "difficulty", label: "Difficulty (0 easy → 1 hard)", type: "range", min: 0, max: 1, step: 0.05, default: 0.5 },
          { name: "length", label: "Length (m)", type: "range", min: 600, max: 2000, step: 50, default: 1200 }
        ]
      },
      {
        label: "🧬 GA · Watch Evolution Live",
        command: "python3 run.py --train {gens} --live --pop {pop} --seed {seed} --car {car}",
        mode: "native",
        args: [
          { name: "gens", label: "Generations", type: "range", min: 10, max: 200, default: 60 },
          { name: "pop", label: "Population", type: "range", min: 20, max: 120, step: 5, default: 60 },
          { name: "seed", label: "Track Seed", type: "number", default: 7 },
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" }
        ]
      },
      {
        label: "🧬 GA · Train Headless (live charts)",
        command: "python3 run.py --train {gens} --pop {pop} --seed {seed} --car {car}",
        mode: "background",
        args: [
          { name: "gens", label: "Generations", type: "range", min: 20, max: 500, step: 10, default: 120 },
          { name: "pop", label: "Population", type: "range", min: 20, max: 120, step: 5, default: 60 },
          { name: "seed", label: "Track Seed", type: "number", default: 7 },
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" }
        ]
      },
      {
        label: "🧬 GA · Watch Champion",
        command: "python3 run.py --watch --seed {seed} --car {car}",
        mode: "native",
        args: [
          { name: "seed", label: "Track Seed", type: "number", default: 7 },
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" }
        ]
      },
      {
        label: "🏁 PPO Race · Train Headless (live charts)",
        command: "python3 run.py --ppo {iters} --car {car}",
        mode: "background",
        args: [
          { name: "iters", label: "Iterations", type: "range", min: 100, max: 5000, step: 100, default: 600 },
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" }
        ]
      },
      {
        label: "🏁 PPO Race · Train LIVE (S=save / L=load)",
        command: "python3 run.py --ppo {iters} --live --car {car}",
        mode: "native",
        args: [
          { name: "iters", label: "Max Iterations (quit anytime)", type: "number", default: 100000 },
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" }
        ]
      },
      {
        label: "🏁 PPO Race · Continue Training (resume)",
        command: "python3 run.py --ppo {iters} --resume {checkpoint} --car {car}",
        mode: "background",
        args: [
          { name: "iters", label: "More Iterations", type: "range", min: 100, max: 10000, step: 100, default: 1000 },
          { name: "checkpoint", label: "Checkpoint to resume", type: "text", default: "ppo_race.pt" },
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" }
        ]
      },
      {
        label: "🏁 Watch PPO Race",
        command: "python3 run.py --watch-ppo --car {car} --track {track}",
        mode: "native",
        args: [
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" },
          { name: "track", label: "Track", type: "select", options: ["club", "national", "coast", "sprint", "tech", "oval2", "akina", "pass", "speedbowl", "superspeed", "oval", "random", "touge"], default: "national" }
        ]
      },
      {
        label: "💨 PPO Drift · Train Headless (live charts)",
        command: "python3 run.py --drift {iters} --car {car}",
        mode: "background",
        args: [
          { name: "iters", label: "Iterations", type: "range", min: 100, max: 5000, step: 100, default: 1000 },
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" }
        ]
      },
      {
        label: "💨 PPO Drift · Train LIVE (S=save / L=load)",
        command: "python3 run.py --drift {iters} --live --car {car}",
        mode: "native",
        args: [
          { name: "iters", label: "Max Iterations (quit anytime)", type: "number", default: 100000 },
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" }
        ]
      },
      {
        label: "💨 PPO Drift · Continue Training (resume)",
        command: "python3 run.py --drift {iters} --resume {checkpoint} --car {car}",
        mode: "background",
        args: [
          { name: "iters", label: "More Iterations", type: "range", min: 100, max: 10000, step: 100, default: 1500 },
          { name: "checkpoint", label: "Checkpoint to resume", type: "text", default: "ppo_drift.pt" },
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" }
        ]
      },
      {
        label: "💨 Watch PPO Drift",
        command: "python3 run.py --watch-drift --car {car} --track {track}",
        mode: "native",
        args: [
          { name: "car", label: "Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" },
          { name: "track", label: "Track", type: "select", options: ["club", "national", "coast", "sprint", "tech", "oval2", "akina", "pass", "speedbowl", "superspeed", "oval", "random", "touge"], default: "akina" }
        ]
      }
    ]
  },
  {
    id: "supra-ai-multiagent",
    name: "Supra Multi-Agent Racing",
    path: "~/Desktop/Supra Ai 2",
    category: "ai-racing",
    desc: "Train or watch your AI race competitively against historical frozen policies on a staggered starting grid. Fully supports live collision physics, aero damage, and adversarial reward shaping.",
    tech: ["Python", "PyTorch", "PPO", "Multi-Agent"],
    icon: "🏎️",
    scripts: [
      {
        label: "🏁 Train Multi-Agent (Headless)",
        command: "python3 run.py --ppo {iters} --car {car} --opponents \"{opp1},{opp2},{opp3}\"",
        mode: "background",
        args: [
          { name: "iters", label: "Iterations", type: "range", min: 100, max: 10000, step: 100, default: 1000 },
          { name: "car", label: "Learning Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" },
          { name: "opp1", label: "Opponent 1", type: "select", options: ["Generalist_pt1.pt", "RX7generalistnew.pt", "ppo_race.pt", "touge_race.pt"], default: "Generalist_pt1.pt" },
          { name: "opp2", label: "Opponent 2", type: "select", options: ["none", "Generalist_pt1.pt", "RX7generalistnew.pt", "ppo_race.pt", "touge_race.pt"], default: "none" },
          { name: "opp3", label: "Opponent 3", type: "select", options: ["none", "Generalist_pt1.pt", "RX7generalistnew.pt", "ppo_race.pt", "touge_race.pt"], default: "none" }
        ]
      },
      {
        label: "🏁 Train Multi-Agent LIVE (UI)",
        command: "python3 run.py --ppo {iters} --live --car {car} --opponents \"{opp1},{opp2},{opp3}\"",
        mode: "native",
        args: [
          { name: "iters", label: "Iterations", type: "number", default: 10000 },
          { name: "car", label: "Learning Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" },
          { name: "opp1", label: "Opponent 1", type: "select", options: ["Generalist_pt1.pt", "RX7generalistnew.pt", "ppo_race.pt", "touge_race.pt"], default: "Generalist_pt1.pt" },
          { name: "opp2", label: "Opponent 2", type: "select", options: ["none", "Generalist_pt1.pt", "RX7generalistnew.pt", "ppo_race.pt", "touge_race.pt"], default: "none" },
          { name: "opp3", label: "Opponent 3", type: "select", options: ["none", "Generalist_pt1.pt", "RX7generalistnew.pt", "ppo_race.pt", "touge_race.pt"], default: "none" }
        ]
      },
      {
        label: "🏁 Watch Multi-Agent Race",
        command: "python3 run.py --watch-ppo --car {car} --opponents \"{opp1},{opp2},{opp3}\"",
        mode: "native",
        args: [
          { name: "car", label: "Learning Chassis", type: "select", options: ["supra", "rx7", "skyline"], default: "supra" },
          { name: "opp1", label: "Opponent 1", type: "select", options: ["Generalist_pt1.pt", "RX7generalistnew.pt", "ppo_race.pt", "touge_race.pt"], default: "Generalist_pt1.pt" },
          { name: "opp2", label: "Opponent 2", type: "select", options: ["none", "Generalist_pt1.pt", "RX7generalistnew.pt", "ppo_race.pt", "touge_race.pt"], default: "none" },
          { name: "opp3", label: "Opponent 3", type: "select", options: ["none", "Generalist_pt1.pt", "RX7generalistnew.pt", "ppo_race.pt", "touge_race.pt"], default: "none" }
        ]
      }
    ]
  },
  {
    id: "supra-drift-edition",
    name: "Supra Drift Edition",
    path: "~/Desktop/ai supra drift edition",
    category: "ai-racing",
    desc: "A dedicated codebase branch focused on drifting objectives, featuring Skyline LSTM sequence-prediction policies tuning combined slip physics.",
    tech: ["Python", "PyTorch", "PyGame", "LSTM", "RNN"],
    icon: "🏎️",
    scripts: [
      {
        label: "Watch Skyline LSTM Drift",
        command: "python3 run.py --watch-drift --checkpoint skyline_drift_lstm.best.pt --recurrent --car skyline --track {track}",
        mode: "native",
        args: [
          { name: "track", label: "Track Name", type: "select", options: ["club", "national", "coast", "sprint", "tech", "oval2", "akina", "pass", "speedbowl", "superspeed"], default: "akina" }
        ]
      },
      {
        label: "Watch PPO LSTM Race",
        command: "python3 run.py --watch-ppo --checkpoint ppo_lstm.best.pt --recurrent --car skyline --track {track}",
        mode: "native",
        args: [
          { name: "track", label: "Track Name", type: "select", options: ["club", "national", "coast", "sprint", "tech", "oval2", "akina", "pass", "speedbowl", "superspeed"], default: "club" }
        ]
      }
    ]
  },
  {
    id: "race-evolution",
    name: "Race Evolution",
    path: "~/Desktop/race_evolution",
    category: "ai-racing",
    desc: "Genetic neuroevolution racer utilizing customizable layouts, collision vectors, physics drafting, custom maps, and live network graph overlays.",
    tech: ["Python", "PyGame", "NEAT-Python", "Genetic-Algorithm"],
    icon: "🏎️",
    scripts: [
      {
        label: "Run NEAT Evolution",
        command: "python3 main.py",
        mode: "native",
        args: []
      }
    ]
  },
  {
    id: "mario-tas-desktop",
    name: "Mario TAS (Desktop)",
    path: "~/Desktop/Mario-TAS/Mario-TAS",
    category: "game-ai",
    desc: "Super Mario World NEAT Tool-Assisted Speedrun bot. Integrates with the BizHawk emulator, extracting SNES RAM data (level structure, enemies) via Lua sockets.",
    tech: ["Python", "Lua", "BizHawk", "NEAT-Python", "Retro-Emulation"],
    icon: "🎮",
    scripts: [
      {
        label: "Launch Python Server",
        command: "python3 launch_tas.py",
        mode: "background",
        args: []
      }
    ]
  },
  {
    id: "mario-tas-downloads",
    name: "Mario TAS Downloads",
    path: "~/Downloads/files",
    category: "game-ai",
    desc: "RetroArch keyboard automation script driving SMW NEAT bot utilizing Quartz screen capture and pynput virtual keyboard inputs.",
    tech: ["Python", "Quartz", "pynput", "MSS", "OpenCV"],
    icon: "🎮",
    scripts: [
      {
        label: "Run Keyboard TAS Bot",
        command: "python3 launch_keyboard_tas.py",
        mode: "native",
        args: []
      }
    ]
  },
  {
    id: "neural-viz",
    name: "Neural Viz",
    path: "~/Desktop/neural_viz.py",
    category: "game-ai",
    desc: "Visual dashboard displaying real-time NEAT connection topologies, weight matrices, fitness graphs, and raw state matrices for Mario TAS.",
    tech: ["Python", "PyGame", "Matplotlib", "Visualization"],
    icon: "📊",
    scripts: [
      {
        label: "Launch Neural Viz Panel",
        command: "python3 neural_viz.py",
        mode: "native",
        args: []
      }
    ]
  },
  {
    id: "trading-ai",
    name: "Trading AI HFT",
    path: "~/Desktop/trading-ai-hft",
    category: "trading",
    desc: "High-Frequency AI trading sandbox utilizing historical price data, FinBERT headlines, Alpaca APIs, NEAT/PPO agents, and voting Councils of 9.",
    tech: ["Python", "Stable-Baselines3", "Alpaca-API", "FinBERT", "Backtesting"],
    icon: "📈",
    scripts: [
      {
        label: "NEAT Train (train.py)",
        command: "python3 train.py --ticker {ticker} --generations {gens} {resume} {use_sent}",
        mode: "background",
        args: [
          { name: "ticker", label: "Crypto Ticker", type: "select", options: ["BTC-USD", "ETH-USD", "SOL-USD"], default: "BTC-USD" },
          { name: "gens", label: "Generations Limit", type: "number", default: 100 },
          { name: "resume", label: "Resume Checkpoint", type: "checkbox", flag: "--resume", default: false },
          { name: "use_sent", label: "Use Synthetic Sentiment", type: "checkbox", flag: "--use-synthetic-sentiment", default: true }
        ]
      },
      {
        label: "PPO RL Train (train_ppo.py)",
        command: "python3 train_ppo.py --tickers {tickers} --timesteps {steps} {use_sent}",
        mode: "background",
        args: [
          { name: "tickers", label: "Tickers (space separated)", type: "text", default: "BTC-USD ETH-USD SOL-USD" },
          { name: "steps", label: "Training Timesteps", type: "number", default: 500000 },
          { name: "use_sent", label: "Use Synthetic Sentiment", type: "checkbox", flag: "--use-synthetic-sentiment", default: true }
        ]
      },
      {
        label: "Train Ensemble Council (train_ensemble_factory.py)",
        command: "python3 train_ensemble_factory.py --timesteps {steps} --count {count} {no_promote}",
        mode: "background",
        args: [
          { name: "steps", label: "Steps per agent", type: "number", default: 100000 },
          { name: "count", label: "Council Size", type: "range", min: 3, max: 15, default: 9 },
          { name: "no_promote", label: "Train only (No active promotion)", type: "checkbox", flag: "--no-promote", default: false }
        ]
      },
      {
        label: "Backtest NEAT Genome (backtest.py)",
        command: "python3 backtest.py --ticker {ticker} --genome {genome}",
        mode: "background",
        args: [
          { name: "ticker", label: "Evaluation Ticker", type: "select", options: ["BTC-USD", "ETH-USD", "SOL-USD"], default: "ETH-USD" },
          { name: "genome", label: "Genome File Path", type: "text", default: "checkpoints/best_genome.pkl" }
        ]
      },
      {
        label: "Backtest PPO Model (backtest_ppo.py)",
        command: "python3 backtest_ppo.py --ticker {ticker} --model {model}",
        mode: "background",
        args: [
          { name: "ticker", label: "Evaluation Ticker", type: "select", options: ["BTC-USD", "ETH-USD", "SOL-USD"], default: "BTC-USD" },
          { name: "model", label: "Model File Path (.zip)", type: "text", default: "models/ppo_trader.zip" }
        ]
      },
      {
        label: "Walk Forward NEAT Validation (walk_forward.py)",
        command: "python3 walk_forward.py --ticker {ticker} --generations {gens}",
        mode: "background",
        args: [
          { name: "ticker", label: "Ticker", type: "select", options: ["BTC-USD", "ETH-USD"], default: "BTC-USD" },
          { name: "gens", label: "Gens per window", type: "number", default: 200 }
        ]
      },
      {
        label: "Run Live Council (live_ensemble.py)",
        command: "python3 live_ensemble.py {allow_live} --tickers {tickers}",
        mode: "background",
        args: [
          { name: "allow_live", label: "Allow Real-Money Trading (Danger!)", type: "checkbox", flag: "--allow-live", default: false },
          { name: "tickers", label: "Tickers (comma separated)", type: "text", default: "BTC-USD" }
        ]
      },
      {
        label: "Run Live Web Dashboard",
        command: "python3 live_dashboard.py --port {port}",
        mode: "background",
        args: [
          { name: "port", label: "HTTP Server Port", type: "number", default: 5050 }
        ]
      }
    ]
  },
  {
    id: "healthbridge-desktop",
    name: "HealthBridge (Desktop)",
    path: "~/Desktop/HealthBridge",
    category: "health",
    desc: "Precision health analysis engine. Parses blood panel PDFs, aggregates wearable biometrics (Whoop), scores clinical recommendations, and structures 255 genetic SNP variants.",
    tech: ["Python", "Flask", "AWS-Bedrock", "ReportLab", "genomics"],
    icon: "🧬",
    scripts: [
      {
        label: "Start Flask Server v2 (Templated)",
        command: "python3 app_v2_templated.py",
        mode: "background",
        args: []
      },
      {
        label: "Start Flask Server v2",
        command: "python3 app_v2.py",
        mode: "background",
        args: []
      },
      {
        label: "Parse Blood Report PDF",
        command: "python3 blood_parser.py {pdf}",
        mode: "background",
        args: [
          { name: "pdf", label: "Report PDF Path", type: "text", default: "blood_report.pdf" }
        ]
      },
      {
        label: "Process genomic txt SNPs",
        command: "python3 snp_processor.py {file}",
        mode: "background",
        args: [
          { name: "file", label: "SNPs TXT File Path", type: "text", default: "raw_dna.txt" }
        ]
      }
    ]
  },
  {
    id: "healthbridge-home",
    name: "HealthBridge (Home)",
    path: "~/healthbridge",
    category: "health",
    desc: "Alternative path variant of the HealthBridge genomic system, optimized for AWS Bedrock integration and clinical pattern queries.",
    tech: ["Python", "Flask", "Bedrock-API", "genomics"],
    icon: "🧬",
    scripts: [
      {
        label: "Start Flask app",
        command: "python3 app.py",
        mode: "background",
        args: []
      }
    ]
  },
  {
    id: "spotify-sync",
    name: "Spotify Sync",
    path: "~/Desktop/spotify_sync.py",
    category: "health",
    desc: "Daily sync script downloading Spotify music habits to Obsidian vaults, mapping track parameters (tempo, energy, valence) for neuroacoustics research.",
    tech: ["Python", "Spotipy", "Obsidian-Integration"],
    icon: "🎵",
    scripts: [
      {
        label: "Sync Music Telemetry",
        command: "python3 spotify_sync.py",
        mode: "background",
        args: []
      }
    ]
  },
  {
    id: "carson-brain",
    name: "Carson Brain",
    path: "~/Desktop/carson-brain",
    category: "health",
    desc: "Obsidian knowledge vault processing genomic profiles and biometric data. Features Gemini scribe tools and vault automated librarians.",
    tech: ["Obsidian", "Markdown", "Vite", "Node.js", "Python"],
    icon: "🧠",
    scripts: [
      {
        label: "Sync Whoop Data",
        command: "./\"Sync Whoop.command\"",
        mode: "native",
        args: []
      },
      {
        label: "Run Vault Librarian",
        command: "./\"projects/ai-librarian/Vault Librarian.command\"",
        mode: "native",
        args: []
      },
      {
        label: "Run Scribe Backend (Flask)",
        command: "python3 backend/main.py",
        mode: "background",
        args: []
      },
      {
        label: "Launch Local Vault Web Viewer",
        command: "npm run dev",
        mode: "background",
        args: []
      }
    ]
  },
  {
    id: "synapse",
    name: "Synapse",
    path: "~/synapse.py",
    category: "health",
    desc: "Flask ingestion gateway utilizing Claude AI models to parse clipboard entries or quick text notes and auto-route structured Markdown into Obsidian.",
    tech: ["Python", "Flask", "Claude-API", "Obsidian"],
    icon: "⚡",
    scripts: [
      {
        label: "Run Synapse Server (Port 7337)",
        command: "python3 synapse.py",
        mode: "background",
        args: []
      }
    ]
  },
  {
    id: "night-cruise",
    name: "Night Cruise",
    path: "~/Desktop/night-cruise",
    category: "creative",
    desc: "Interactive synthwave audio-visualizer built with vanilla HTML Canvas and Web Audio API. Supports audio frequencies beating, file drop, and microphone inputs.",
    tech: ["HTML5", "CSS3", "JavaScript", "Web-Audio", "Canvas"],
    icon: "🌌",
    scripts: [
      {
        label: "Open in Browser",
        command: "open index.html",
        mode: "native",
        args: []
      }
    ]
  },
  {
    id: "fishtank",
    name: "Fishtank (Terminal / HTML)",
    path: "~/Desktop/Code Projects",
    category: "creative",
    desc: "Ultimate curses-based ASCII fish tank. Features animated marine organisms, hermit crabs, randomized quotes, file persistence, and duplicate HTML template.",
    tech: ["Python", "Curses", "Terminal-Art", "HTML", "Vampire-Hermits"],
    icon: "🐠",
    scripts: [
      {
        label: "Run Curses Fish Tank in Terminal",
        command: "python3 fishtank.py",
        mode: "native",
        args: []
      },
      {
        label: "Open HTML Aquarium in Browser",
        command: "open fishtank.html",
        mode: "native",
        args: []
      }
    ]
  },
  {
    id: "spotify-car",
    name: "Spotify Car Visualizer",
    path: "~/Spotify car",
    category: "creative",
    desc: "Vite + React visualizer dashboard tailored for custom in-car displays, syncing real-time Spotify playing state with smooth neon graphics.",
    tech: ["JavaScript", "Vite", "React", "Spotify-API", "WebSockets"],
    icon: "🚘",
    scripts: [
      {
        label: "Start Dev Server",
        command: "npm run dev",
        mode: "background",
        args: []
      }
    ]
  },
  {
    id: "carson-cage",
    name: "Carson Cage",
    path: "~/carson_cage.py",
    category: "tools",
    desc: "Quirky retro-styled terminal widget depicting Carson trapped inside a grid cage, reciting biometrics, Obsidian quotes, and genomic telemetry.",
    tech: ["Python", "Curses", "Terminal-Art"],
    icon: "⛓️",
    scripts: [
      {
        label: "Run Carson Cage",
        command: "python3 carson_cage.py",
        mode: "native",
        args: []
      }
    ]
  },
  {
    id: "carson-jar",
    name: "Carson Jar",
    path: "~/carson_jar.py",
    category: "tools",
    desc: "ASCII console art drawing a microscopic Carson inside a glass beaker bottle, outputting Obsidian snippets.",
    tech: ["Python", "ASCII-Art"],
    icon: "🫙",
    scripts: [
      {
        label: "Run Carson Jar",
        command: "python3 carson_jar.py",
        mode: "native",
        args: []
      }
    ]
  },
  {
    id: "carson-status",
    name: "Carson Status Panel",
    path: "~/carson_status.py",
    category: "tools",
    desc: "Comprehensive terminal console control board displaying daily Whoop sleep summaries, obsidian counts, and health bridge logs in real time.",
    tech: ["Python", "Curses", "Terminal-Dashboard"],
    icon: "📟",
    scripts: [
      {
        label: "Run Carson Status",
        command: "python3 carson_status.py",
        mode: "native",
        args: []
      }
    ]
  },
  {
    id: "cre-study-guide",
    name: "CRE Study Guide PDF",
    path: "~/create_study_guide_pdf.py",
    category: "tools",
    desc: "ReportLab automation script compiling 50+ real estate valuation questions, formulas, and chemistry quiz tables into a clean multi-page document.",
    tech: ["Python", "ReportLab", "PDF-Generation"],
    icon: "📄",
    scripts: [
      {
        label: "Generate Study Guide PDF",
        command: "python3 create_study_guide_pdf.py",
        mode: "background",
        args: []
      }
    ]
  }
];

// 2. Global State Variables
let currentCategory = "all";
let searchFilterQuery = "";
let runningProcesses = new Map(); // pid -> { pid, label, command, active, logsStream: EventSource, metricsHistory: [], labels: [], datasetsData: {}, keysList: Set }
let currentTerminalPid = null; // Which process's logs are we currently reading
let activeConfigModal = { project: null, script: null };
let activeChart = null; // Active Chart.js instance for training metrics

const METRIC_COLORS = [
  '#00e5ff', // Cyan
  '#d500f9', // Magenta
  '#ffab00', // Amber
  '#00e676', // Emerald
  '#ff1744', // Red
  '#2979ff', // Blue
  '#ffeb3b', // Yellow
  '#9c27b0'  // Purple
];

// DOM Selectors
const projectsGrid = document.getElementById("projects-grid");
const searchInput = document.getElementById("search-input");
const categoryLinks = document.querySelectorAll(".category-nav li");
const clockDisplay = document.getElementById("clock-display");
const runningTasksList = document.getElementById("running-tasks-list");

// Telemetry Selectors
const cpuBar = document.getElementById("cpu-bar");
const cpuVal = document.getElementById("cpu-val");
const memBar = document.getElementById("mem-bar");
const memVal = document.getElementById("mem-val");
const hostVal = document.getElementById("host-val");
const uptimeVal = document.getElementById("uptime-val");

// Modal Selectors
const configModal = document.getElementById("config-modal");
const modalTitle = document.getElementById("modal-title");
const modalFieldsContainer = document.getElementById("modal-fields-container");
const commandPreviewText = document.getElementById("command-preview-text");
const closeBtn = document.getElementById("close-modal-btn");
const cancelBtn = document.getElementById("cancel-launch-btn");
const launchBtn = document.getElementById("launch-task-btn");

// Terminal Selectors
const terminalDrawer = document.getElementById("terminal-drawer");
const terminalBody = document.getElementById("terminal-body");
const terminalStdout = document.getElementById("terminal-stdout");
const terminalProcessTabs = document.getElementById("terminal-process-tabs");
const termClearBtn = document.getElementById("term-clear");
const termAutoscrollBtn = document.getElementById("term-autoscroll");
const termStopBtn = document.getElementById("term-stop-proc");
const termToggleSizeBtn = document.getElementById("term-toggle-size");
const terminalHeaderBar = document.getElementById("terminal-header-bar");
const terminalWorkspaceWrapper = document.getElementById("terminal-workspace-wrapper");
const terminalViewSelector = document.getElementById("terminal-view-selector");
const metricsStatsGrid = document.getElementById("metrics-stats-grid");

let autoscrollEnabled = true;

// Setup View mode toggles
document.querySelectorAll(".view-tab").forEach(tab => {
  tab.addEventListener("click", (e) => {
    e.stopPropagation();
    document.querySelectorAll(".view-tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    
    const view = tab.dataset.view;
    terminalWorkspaceWrapper.className = `terminal-workspace-wrapper view-${view}`;
    
    // Resize charts to fit layout container
    if (activeChart) {
      setTimeout(() => {
        activeChart.resize();
      }, 50);
    }
  });
});

// 3. UI Helpers: Clock
function updateClock() {
  const now = new Date();
  clockDisplay.textContent = now.toLocaleTimeString();
}
setInterval(updateClock, 1000);
updateClock();

// 4. Render Project Cards
function getAccentGradient(category) {
  switch (category) {
    case "ai-racing": return "var(--gradient-ai-racing)";
    case "game-ai": return "var(--gradient-game-ai)";
    case "trading": return "var(--gradient-trading)";
    case "health": return "var(--gradient-health)";
    case "creative": return "var(--gradient-creative)";
    case "tools": return "var(--gradient-tools)";
    default: return "linear-gradient(135deg, #1e293b, #0f172a)";
  }
}

function getAccentColor(category) {
  switch (category) {
    case "ai-racing": return "var(--color-ai-racing)";
    case "game-ai": return "var(--color-game-ai)";
    case "trading": return "var(--color-trading)";
    case "health": return "var(--color-health)";
    case "creative": return "var(--color-creative)";
    case "tools": return "var(--color-tools)";
    default: return "rgba(255,255,255,0.2)";
  }
}

function renderCards() {
  projectsGrid.innerHTML = "";
  
  const filtered = PROJECTS.filter(p => {
    const matchesCategory = currentCategory === "all" || p.category === currentCategory;
    const matchesSearch = p.name.toLowerCase().includes(searchFilterQuery) ||
                          p.desc.toLowerCase().includes(searchFilterQuery) ||
                          p.tech.some(t => t.toLowerCase().includes(searchFilterQuery));
    return matchesCategory && matchesSearch;
  });

  filtered.forEach(p => {
    // Check if any scripts in this project are running
    const runningScript = Array.from(runningProcesses.values()).find(rp => rp.cwd === p.path && rp.active);
    const accentGrad = getAccentGradient(p.category);
    const accentColor = getAccentColor(p.category);

    const card = document.createElement("div");
    card.className = `project-card ${runningScript ? "active-proc" : ""}`;
    card.style.setProperty("--card-accent-gradient", accentGrad);
    card.style.setProperty("--card-accent-border", accentColor);
    card.style.setProperty("--card-shadow-color", accentColor.replace("1)", "0.2)"));

    let runningBadgeHtml = "";
    let runButtonHtml = "";

    if (runningScript) {
      runningBadgeHtml = `
        <div class="active-badge" style="color: ${accentColor}; border-color: ${accentColor}; background: ${accentColor.replace('1)', '0.15)')}">
          <span class="task-dot" style="background-color: ${accentColor}; box-shadow: 0 0 8px ${accentColor}"></span>
          Running (PID ${runningScript.pid})
        </div>
      `;
      
      runButtonHtml = `
        <button class="btn btn-stop" onclick="stopTask(${runningScript.pid})">
          ⏹️ Stop Task
        </button>
      `;
    } else {
      runButtonHtml = `
        <button class="btn btn-run-cmd" onclick="openConfigModal('${p.id}')">
          🚀 Run / Launch Options
        </button>
      `;
    }

    const techBadgesHtml = p.tech.map(t => `<span class="tech-badge">${t}</span>`).join("");

    card.innerHTML = `
      ${runningBadgeHtml}
      <div class="card-header">
        <div class="card-title-group">
          <span class="project-icon">${p.icon}</span>
          <div>
            <h3 class="project-title">${p.name}</h3>
            <span class="category-tag" style="background: ${accentGrad}">${p.category.toUpperCase().replace("-", " ")}</span>
          </div>
        </div>
      </div>
      <p class="project-desc">${p.desc}</p>
      <div class="tech-badges">${techBadgesHtml}</div>
      <div class="card-actions">
        <button class="btn btn-secondary" onclick="openFolder('${p.path}')" title="Open Folder in Finder">📂 Folder</button>
        <button class="btn btn-secondary" onclick="openEditor('${p.path}')" title="Open in VS Code / Cursor">💻 Edit</button>
        <button class="btn btn-secondary" onclick="openShell('${p.path}')" title="Open blank terminal shell">🐚 Shell</button>
        ${runButtonHtml}
      </div>
    `;

    projectsGrid.appendChild(card);
  });

  updateCounts();
}

// 5. Update counts on categories in sidebar
function updateCounts() {
  document.getElementById("count-all").textContent = PROJECTS.length;
  
  const categories = ["ai-racing", "game-ai", "trading", "health", "creative", "tools"];
  categories.forEach(cat => {
    const count = PROJECTS.filter(p => p.category === cat).length;
    const countEl = document.getElementById(`count-${cat}`);
    if (countEl) countEl.textContent = count;
  });
}

// Category selection
categoryLinks.forEach(link => {
  link.addEventListener("click", () => {
    categoryLinks.forEach(l => l.classList.remove("active"));
    link.classList.add("active");
    currentCategory = link.dataset.category;
    renderCards();
  });
});

// Search input
searchInput.addEventListener("input", (e) => {
  searchFilterQuery = e.target.value.toLowerCase().trim();
  renderCards();
});

// 6. Config / Launch Modal Logic
function openConfigModal(projectId) {
  const p = PROJECTS.find(proj => proj.id === projectId);
  if (!p) return;

  activeConfigModal.project = p;
  activeConfigModal.script = p.scripts[0]; // Select first script by default

  modalTitle.textContent = `${p.name} - Run Launch Options`;
  renderModalFields();
  
  configModal.classList.add("active");
}

function renderModalFields() {
  const p = activeConfigModal.project;
  const currentScript = activeConfigModal.script;

  if (!p || !currentScript) return;

  let html = "";

  // Script selection dropdown if multiple scripts exist
  if (p.scripts.length > 1) {
    html += `
      <div class="form-group">
        <label for="script-select">Select Script/Mode</label>
        <select class="form-control" id="script-select">
          ${p.scripts.map((s, idx) => `<option value="${idx}" ${s.label === currentScript.label ? "selected" : ""}>${s.label}</option>`).join("")}
        </select>
      </div>
    `;
  }

  // Dynamic Argument fields
  if (currentScript.args && currentScript.args.length > 0) {
    html += `<h4 style="margin: 20px 0 10px 0; font-size: 0.9rem; color: var(--text-secondary); text-transform: uppercase;">Parameters</h4>`;
    
    currentScript.args.forEach(arg => {
      html += `<div class="form-group">`;
      html += `<label for="arg-${arg.name}">${arg.label}</label>`;

      if (arg.type === "select") {
        html += `
          <select class="form-control arg-input" id="arg-${arg.name}" data-name="${arg.name}" data-type="select">
            ${arg.options.map(opt => `<option value="${opt}" ${opt === arg.default ? "selected" : ""}>${opt}</option>`).join("")}
          </select>
        `;
      } else if (arg.type === "range") {
        html += `
          <div class="range-slider-wrapper">
            <input type="range" class="arg-input" id="arg-${arg.name}" data-name="${arg.name}" data-type="range" min="${arg.min}" max="${arg.max}" step="${arg.step || 1}" value="${arg.default}">
            <span class="slider-val" id="val-display-${arg.name}">${arg.default}</span>
          </div>
        `;
      } else if (arg.type === "checkbox") {
        html += `
          <div class="form-row-checkbox">
            <input type="checkbox" class="arg-input" id="arg-${arg.name}" data-name="${arg.name}" data-type="checkbox" data-flag="${arg.flag}" ${arg.default ? "checked" : ""}>
            <span>Enable flag (${arg.flag})</span>
          </div>
        `;
      } else if (arg.type === "number") {
        html += `<input type="number" class="form-control arg-input" id="arg-${arg.name}" data-name="${arg.name}" data-type="number" value="${arg.default}">`;
      } else {
        // Default text type
        html += `<input type="text" class="form-control arg-input" id="arg-${arg.name}" data-name="${arg.name}" data-type="text" value="${arg.default}">`;
      }

      html += `</div>`;
    });
  } else {
    html += `<p style="color: var(--text-secondary); font-size: 0.88rem; font-style: italic;">This script has no extra arguments. Will launch with default command.</p>`;
  }

  modalFieldsContainer.innerHTML = html;

  // Add event listeners to select script
  const scriptSelect = document.getElementById("script-select");
  if (scriptSelect) {
    scriptSelect.addEventListener("change", (e) => {
      activeConfigModal.script = p.scripts[parseInt(e.target.value)];
      renderModalFields();
    });
  }

  // Add event listeners to inputs to update command preview
  const inputs = document.querySelectorAll(".arg-input");
  inputs.forEach(input => {
    input.addEventListener("input", (e) => {
      if (input.dataset.type === "range") {
        document.getElementById(`val-display-${input.dataset.name}`).textContent = e.target.value;
      }
      updateCommandPreview();
    });
  });

  updateCommandPreview();
}

function updateCommandPreview() {
  const currentScript = activeConfigModal.script;
  if (!currentScript) return;

  let finalCmd = currentScript.command;

  if (currentScript.args && currentScript.args.length > 0) {
    currentScript.args.forEach(arg => {
      const input = document.getElementById(`arg-${arg.name}`);
      if (input) {
        let val = "";
        if (arg.type === "checkbox") {
          val = input.checked ? arg.flag : "";
        } else {
          val = input.value;
        }

        // Replace template placeholder in script command template
        finalCmd = finalCmd.replace(`{${arg.name}}`, val);
      }
    });
  }

  // Strip duplicate whitespace
  finalCmd = finalCmd.replace(/\s+/g, " ").trim();
  commandPreviewText.textContent = finalCmd;
}

// Close modals
function closeModal() {
  configModal.classList.remove("active");
  activeConfigModal = { project: null, script: null };
}

closeBtn.addEventListener("click", closeModal);
cancelBtn.addEventListener("click", closeModal);
window.addEventListener("click", (e) => {
  if (e.target === configModal) closeModal();
});

// Launch Task Trigger
launchBtn.addEventListener("click", () => {
  const p = activeConfigModal.project;
  const script = activeConfigModal.script;
  const command = commandPreviewText.textContent;
  
  if (!p || !script) return;

  closeModal();

  if (script.mode === "native") {
    // Open in native Terminal via AppleScript
    postApi("/api/run/native", { command, cwd: p.path });
  } else {
    // Run in background and stream logs
    postApi("/api/run/background", { command, cwd: p.path })
      .then(res => {
        if (res && res.success) {
          // Initialize logs stream client-side
          startLogStreaming(res.pid, script.label, command, p.path);
        }
      });
  }
});

// 7. EventSource & SSE Log Streaming
function startLogStreaming(pid, label, command, cwd) {
  // If the terminal panel is minimized, expand it
  expandTerminal();

  // Create stream reference
  const streamInfo = {
    pid,
    label,
    command,
    cwd,
    active: true,
    logs: "",
    metricsHistory: [], // stores entries { t, d }
    labels: [],         // X axis ticks
    datasetsData: {},   // key -> array of float coordinates
    keysList: new Set() // unique variables list
  };

  runningProcesses.set(pid, streamInfo);

  // Setup EventSource listener to route stream from backend
  const source = new EventSource(`/api/stream-logs/${pid}`);
  streamInfo.logsStream = source;

  source.onmessage = (event) => {
    const data = JSON.parse(event.data);

    // 1. Process metrics history if available
    if (data.historyMetrics && data.historyMetrics.length > 0) {
      data.historyMetrics.forEach(entry => {
        processIncomingMetrics(pid, entry.d, entry.t);
      });
    }

    // 2. Process new metrics batch
    let hasNewMetrics = false;
    if (data.metrics && data.metrics.length > 0) {
      data.metrics.forEach(m => {
        processIncomingMetrics(pid, m, Date.now());
      });
      hasNewMetrics = true;
    }

    streamInfo.logs += data.text;
    
    // If this is the active tab, append to terminal UI screen
    if (currentTerminalPid === pid) {
      appendLogs(data.text);
      if (hasNewMetrics) {
        updateChartAndStats(pid);
      }
    }
  };

  source.onerror = (err) => {
    console.log(`[SSE Client] Log stream disconnected for PID ${pid}`);
    source.close();
    streamInfo.active = false;
    updateTasksUI();
  };

  // Switch terminal tab focus to new task
  switchTerminalTab(pid);
  updateTasksUI();
}

function stopTask(pid) {
  postApi(`/api/stop/${pid}`)
    .then(res => {
      console.log(`Stop requested for task ${pid}:`, res);
      // Backend handles SIGINT & SIGKILL
    });
}

// 8. Terminal GUI Drawer Operations
function toggleTerminalSize() {
  if (terminalDrawer.classList.contains("minimized")) {
    expandTerminal();
  } else if (terminalDrawer.classList.contains("expanded")) {
    maximizeTerminal();
  } else {
    minimizeTerminal();
  }
}

function expandTerminal() {
  terminalDrawer.className = "terminal-drawer terminal-panel-floating expanded";
  termToggleSizeBtn.innerHTML = "&#9660;"; // Down caret
  
  if (autoscrollEnabled) scrollTerminalToBottom();
}

function maximizeTerminal() {
  terminalDrawer.className = "terminal-drawer terminal-panel-floating maximized";
  termToggleSizeBtn.innerHTML = "_"; // Dash line
  
  if (autoscrollEnabled) scrollTerminalToBottom();
}

function minimizeTerminal() {
  terminalDrawer.className = "terminal-drawer terminal-panel-floating minimized";
  termToggleSizeBtn.innerHTML = "&#9654;"; // Up caret
}

termToggleSizeBtn.addEventListener("click", toggleTerminalSize);
terminalHeaderBar.addEventListener("click", (e) => {
  // Only toggle if we didn't click inside control buttons or tabs
  if (!e.target.closest(".terminal-controls") && !e.target.closest(".terminal-tabs-container")) {
    toggleTerminalSize();
  }
});

// Clear Logs Panel
termClearBtn.addEventListener("click", () => {
  terminalStdout.textContent = "";
  if (currentTerminalPid) {
    const rp = runningProcesses.get(currentTerminalPid);
    if (rp) rp.logs = "";
    
    // Clear logs buffer on backend as well
    postApi(`/api/clear-logs/${currentTerminalPid}`);
  }
});

// Auto scroll toggle button
termAutoscrollBtn.addEventListener("click", () => {
  autoscrollEnabled = !autoscrollEnabled;
  termAutoscrollBtn.classList.toggle("active", autoscrollEnabled);
  if (autoscrollEnabled) scrollTerminalToBottom();
});

// Stop process button from logs panel
termStopBtn.addEventListener("click", () => {
  if (currentTerminalPid) {
    stopTask(currentTerminalPid);
  }
});

function scrollTerminalToBottom() {
  terminalBody.scrollTop = terminalBody.scrollHeight;
}

function appendLogs(text) {
  // Fast scroll handling
  const atBottom = terminalBody.scrollHeight - terminalBody.clientHeight <= terminalBody.scrollTop + 30;
  
  terminalStdout.textContent += text;
  
  if (autoscrollEnabled && atBottom) {
    scrollTerminalToBottom();
  }
}

function switchTerminalTab(pid) {
  currentTerminalPid = pid;
  
  // Destroy old chart
  if (activeChart) {
    activeChart.destroy();
    activeChart = null;
  }
  
  // Re-render tabs list highlights
  const tabsList = document.querySelectorAll(".term-tab");
  tabsList.forEach(t => {
    t.classList.toggle("active", parseInt(t.dataset.pid) === pid);
  });

  const rp = runningProcesses.get(pid);
  if (rp) {
    // Show logs
    terminalStdout.textContent = rp.logs;
    termStopBtn.style.display = rp.active ? "inline-block" : "none";
    if (autoscrollEnabled) scrollTerminalToBottom();
    
    // Render chart and stats history for the active tab
    updateChartAndStats(pid);
  } else {
    terminalStdout.textContent = "Select a running project to stream logs...";
    termStopBtn.style.display = "none";
    metricsStatsGrid.innerHTML = `<div class="stat-placeholder">No active metrics running. Start training to plot live stats.</div>`;
  }
}

// Process metrics data stream
function processIncomingMetrics(pid, metrics, timestamp) {
  const rp = runningProcesses.get(pid);
  if (!rp) return;

  rp.metricsHistory.push({ t: timestamp, d: metrics });
  if (rp.metricsHistory.length > 5000) rp.metricsHistory.shift();

  // Determine index / sequence label (x axis)
  // Use generation if available, otherwise just use current series index
  let tickVal = rp.labels.length + 1;
  if (metrics.generation !== undefined) {
    tickVal = metrics.generation;
  }
  rp.labels.push(tickVal);
  if (rp.labels.length > 5000) rp.labels.shift();

  // Populate datasetsMap values
  Object.keys(metrics).forEach(key => {
    // Skip 'generation' key on chart series since it maps to the X axis ticks!
    if (key === "generation") return;
    
    rp.keysList.add(key);

    if (!rp.datasetsData[key]) {
      rp.datasetsData[key] = [];
    }

    rp.datasetsData[key].push(metrics[key]);
    if (rp.datasetsData[key].length > 5000) rp.datasetsData[key].shift();
  });
}

// Update stats grid cards and recreate / update the charts
function updateChartAndStats(pid) {
  const rp = runningProcesses.get(pid);
  if (!rp || currentTerminalPid !== pid) return;

  // 1. Update Telemetry Summary Cards
  if (rp.keysList.size === 0) {
    metricsStatsGrid.innerHTML = `<div class="stat-placeholder">Listening for parsed telemetry prints...</div>`;
  } else {
    let statsHtml = "";
    
    Array.from(rp.keysList).forEach(key => {
      const dataArr = rp.datasetsData[key] || [];
      if (dataArr.length === 0) return;

      const latest = dataArr[dataArr.length - 1];
      const maxVal = Math.max(...dataArr);
      const avgVal = dataArr.reduce((sum, v) => sum + v, 0) / dataArr.length;

      // Clean label (replace underbars with spaces, capitalize)
      const label = key.replace(/_/g, " ").toUpperCase();
      let displayLatest = typeof latest === "number" ? latest.toFixed(3).replace(/\.?0+$/, "") : latest;
      let displayMax = typeof maxVal === "number" ? maxVal.toFixed(2).replace(/\.?0+$/, "") : maxVal;
      let displayAvg = typeof avgVal === "number" ? avgVal.toFixed(2).replace(/\.?0+$/, "") : avgVal;

      statsHtml += `
        <div class="metric-stat-card">
          <span class="metric-stat-name">${label}</span>
          <span class="metric-stat-value">${displayLatest}</span>
          <span class="metric-stat-meta">Max: ${displayMax} | Avg: ${displayAvg}</span>
        </div>
      `;
    });

    metricsStatsGrid.innerHTML = statsHtml;
  }

  // 2. Re-create or Update Chart.js lines
  const ctx = document.getElementById("metrics-chart-canvas");
  if (!ctx) return;

  const datasetKeys = Array.from(rp.keysList);
  if (datasetKeys.length === 0) return;

  const datasets = datasetKeys.map((key, index) => {
    const color = METRIC_COLORS[index % METRIC_COLORS.length];
    return {
      label: key.replace(/_/g, " ").toUpperCase(),
      data: rp.datasetsData[key] || [],
      borderColor: color,
      backgroundColor: color + "0d", // Very light fill tint
      tension: 0.25,
      borderWidth: 2,
      pointRadius: rp.labels.length > 100 ? 0 : 2, // Hide points for large series to make it clean
      fill: true
    };
  });

  if (activeChart) {
    // Simply update data and re-render
    activeChart.data.labels = rp.labels;
    activeChart.data.datasets = datasets;
    activeChart.update("none"); // Smooth update without reset animation
  } else {
    // Initialize brand new chart
    activeChart = new Chart(ctx.getContext("2d"), {
      type: "line",
      data: {
        labels: rp.labels,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false, // Turn off heavy animation updates
        plugins: {
          legend: {
            position: "top",
            labels: {
              color: "#9ca3af",
              font: { family: "Inter", size: 10, weight: 600 }
            }
          },
          tooltip: {
            mode: "index",
            intersect: false
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.03)" },
            ticks: { color: "#6b7280", maxTicksLimit: 10 }
          },
          y: {
            grid: { color: "rgba(255, 255, 255, 0.03)" },
            ticks: { color: "#6b7280" }
          }
        }
      }
    });
  }
}

// 9. Status Poller: Uptime and running list updates
function updateTasksUI() {
  // Update sidebar list of running tasks
  runningTasksList.innerHTML = "";

  const activeTasks = Array.from(runningProcesses.values()).filter(t => t.active);

  if (activeTasks.length === 0) {
    runningTasksList.innerHTML = `<li class="empty-state">No background tasks</li>`;
  } else {
    activeTasks.forEach(task => {
      const li = document.createElement("li");
      li.innerHTML = `
        <div style="display:flex; align-items:center; cursor:pointer;" onclick="focusLogs(${task.pid})">
          <span class="task-dot"></span>
          <span>${task.label} (${task.pid})</span>
        </div>
        <button style="background:transparent; border:none; color:var(--color-creative); cursor:pointer; font-size:0.75rem;" onclick="stopTask(${task.pid})">⏹️</button>
      `;
      runningTasksList.appendChild(li);
    });
  }

  // Update tabs in terminal panel
  terminalProcessTabs.innerHTML = "";
  
  runningProcesses.forEach((task, pid) => {
    const tab = document.createElement("div");
    tab.className = `term-tab ${currentTerminalPid === pid ? "active" : ""}`;
    tab.dataset.pid = pid;
    tab.innerHTML = `
      <span class="term-tab-dot" style="background-color: ${task.active ? "var(--color-health)" : "var(--text-muted)"}"></span>
      <span>${task.label}</span>
      <span class="term-tab-close" onclick="closeTerminalTab(event, ${pid})">&times;</span>
    `;
    
    tab.addEventListener("click", (e) => {
      if (!e.target.classList.contains("term-tab-close")) {
        switchTerminalTab(pid);
      }
    });

    terminalProcessTabs.appendChild(tab);
  });

  renderCards();
}

function focusLogs(pid) {
  expandTerminal();
  switchTerminalTab(pid);
}

function closeTerminalTab(event, pid) {
  event.stopPropagation(); // Avoid triggering switchTerminalTab
  
  const rp = runningProcesses.get(pid);
  if (rp) {
    if (rp.active) {
      if (confirm(`Process ${rp.label} is still running. Stop it and close log stream?`)) {
        stopTask(pid);
        if (rp.logsStream) rp.logsStream.close();
        runningProcesses.delete(pid);
      } else {
        return; // Don't close
      }
    } else {
      runningProcesses.delete(pid);
    }
  }

  // Select another tab if current tab is closed
  if (currentTerminalPid === pid) {
    const remaining = Array.from(runningProcesses.keys());
    if (remaining.length > 0) {
      switchTerminalTab(remaining[remaining.length - 1]);
    } else {
      switchTerminalTab(null);
    }
  }

  updateTasksUI();
}

// Polling telemetry values from server
function pollSystemStatus() {
  fetch("/api/status")
    .then(res => res.json())
    .then(data => {
      // Update UI components
      cpuBar.style.width = `${data.system.cpu}%`;
      cpuVal.textContent = `${data.system.cpu}%`;
      memBar.style.width = `${data.system.memory}%`;
      memVal.textContent = `${data.system.memory}%`;
      
      hostVal.textContent = data.system.hostname;
      
      const uptimeSec = data.system.uptime;
      const hours = Math.floor(uptimeSec / 3600);
      const minutes = Math.floor((uptimeSec % 3600) / 60);
      uptimeVal.textContent = `${hours}h ${minutes}m`;

      // Keep running processes list updated from backend
      // Compare active background PIDs with our client runningProcesses
      data.processes.forEach(bp => {
        if (bp.active) {
          if (!runningProcesses.has(bp.pid)) {
            // Found a process running on backend that we didn't hook up
            const matchedProj = PROJECTS.find(p => p.path === bp.cwd);
            const scriptLabel = matchedProj ? (matchedProj.scripts.find(s => bp.command.includes(s.command.split(" ")[0]))?.label || "Task") : "Script Task";
            
            console.log(`[Dashboard] Hooking up backend active process: PID ${bp.pid}`);
            startLogStreaming(bp.pid, scriptLabel, bp.command, bp.cwd);
          }
        } else {
          // If we have it marked active, but backend says inactive
          const rp = runningProcesses.get(bp.pid);
          if (rp && rp.active) {
            rp.active = false;
            if (rp.logsStream) rp.logsStream.close();
            updateTasksUI();
          }
        }
      });
    })
    .catch(err => console.error("Telemetry fetch failed:", err));
}

// 10. Native API helper triggers
function postApi(url, body = {}) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  }).then(async res => {
    if (!res.ok) {
      let errMsg = res.statusText;
      try {
        const errData = await res.json();
        if (errData && errData.error) {
          errMsg = errData.error;
        }
      } catch (e) {
        // Response is not JSON or parsing failed
      }
      throw new Error(`API Error: ${errMsg}`);
    }
    return res.json();
  }).catch(err => {
    alert(`Action Failed: ${err.message}`);
    console.error(err);
  });
}

function openFolder(cwd) {
  postApi("/api/open/folder", { cwd });
}

function openEditor(cwd) {
  postApi("/api/open/vscode", { cwd });
}

function openShell(cwd) {
  postApi("/api/open/terminal", { cwd });
}

// Initial setup
renderCards();
setInterval(pollSystemStatus, 3000);
pollSystemStatus();

// Expose functions globally for click handlers
window.openFolder = openFolder;
window.openEditor = openEditor;
window.openShell = openShell;
window.stopTask = stopTask;
window.openConfigModal = openConfigModal;
window.closeTerminalTab = closeTerminalTab;
window.focusLogs = focusLogs;
