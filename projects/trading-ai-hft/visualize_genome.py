"""
visualize_genome.py — Visualize the evolved neural network topology and weights.

Shows which input features (RSI, MACD, Sentiment, etc.) have the strongest
connections to the output decisions (BUY/SELL/HOLD), revealing what the AI
actually learned during evolution.

Usage:
    python3 visualize_genome.py
    python3 visualize_genome.py --genome checkpoints/best_genome.pkl
"""

import sys, os, pickle, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import neat
from features.state_builder import FEATURE_NAMES

# ── Styling ──
COLORS = {
    "bg":     "#0a0e17",
    "card":   "#111827",
    "grid":   "#1e293b",
    "text":   "#e2e8f0",
    "dim":    "#64748b",
    "green":  "#10b981",
    "red":    "#ef4444",
    "blue":   "#3b82f6",
    "purple": "#8b5cf6",
    "amber":  "#f59e0b",
    "cyan":   "#06b6d4",
    "white":  "#ffffff",
}

INPUT_LABELS = FEATURE_NAMES  # ['rsi', 'macd', ..., 'sentiment', 'position_flag', 'unrealized_pnl']
OUTPUT_LABELS = ["HOLD", "BUY", "SELL"]


def load_genome(genome_path=None):
    if genome_path is None:
        genome_path = str(PROJECT_ROOT / "checkpoints" / "best_genome.pkl")
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    print(f"📂 Loaded genome from {genome_path}")
    return genome


def analyze_genome(genome, config):
    """Extract topology info and connection weights."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Get all connections from the genome
    connections = {}
    for key, cg in genome.connections.items():
        if cg.enabled:
            connections[key] = cg.weight

    # Get all nodes
    input_nodes = config.genome_config.input_keys    # [-1, -2, ..., -13]
    output_nodes = config.genome_config.output_keys  # [0, 1, 2]
    hidden_nodes = list(genome.nodes.keys())
    hidden_nodes = [n for n in hidden_nodes if n not in output_nodes]

    # Map input node IDs to feature names
    input_map = {}
    for i, node_id in enumerate(input_nodes):
        if i < len(INPUT_LABELS):
            input_map[node_id] = INPUT_LABELS[i]
        else:
            input_map[node_id] = f"input_{i}"

    output_map = {node_id: OUTPUT_LABELS[i] for i, node_id in enumerate(output_nodes)}

    return {
        "connections": connections,
        "input_nodes": input_nodes,
        "output_nodes": output_nodes,
        "hidden_nodes": hidden_nodes,
        "input_map": input_map,
        "output_map": output_map,
        "genome": genome,
    }


def compute_feature_importance(analysis):
    """
    Calculate how important each input feature is by summing the absolute
    weights of all connections originating from that input node.
    """
    connections = analysis["connections"]
    input_nodes = analysis["input_nodes"]
    input_map = analysis["input_map"]

    importance = defaultdict(float)
    connection_count = defaultdict(int)

    for (src, dst), weight in connections.items():
        if src in input_nodes:
            name = input_map.get(src, f"node_{src}")
            importance[name] += abs(weight)
            connection_count[name] += 1

    # Also trace through hidden nodes (one level deep)
    hidden_nodes = set(analysis["hidden_nodes"])
    # Find connections from inputs to hidden
    input_to_hidden = {}
    for (src, dst), weight in connections.items():
        if src in input_nodes and dst in hidden_nodes:
            name = input_map.get(src, f"node_{src}")
            if dst not in input_to_hidden:
                input_to_hidden[dst] = []
            input_to_hidden[dst].append((name, weight))

    # Find connections from hidden to outputs
    for (src, dst), weight in connections.items():
        if src in hidden_nodes and dst in analysis["output_nodes"]:
            if src in input_to_hidden:
                for (input_name, input_weight) in input_to_hidden[src]:
                    # Propagated importance = |input_weight * hidden_to_output_weight|
                    importance[input_name] += abs(input_weight * weight) * 0.5

    return dict(importance), dict(connection_count)


def compute_output_bias(analysis):
    """
    For each output node, compute the net bias from each input feature.
    Positive = this feature pushes toward this action.
    """
    connections = analysis["connections"]
    input_nodes = analysis["input_nodes"]
    input_map = analysis["input_map"]
    output_map = analysis["output_map"]

    # Direct connections only
    bias_matrix = {}
    for output_id, output_name in output_map.items():
        bias_matrix[output_name] = {}
        for input_id in input_nodes:
            input_name = input_map.get(input_id, f"node_{input_id}")
            key = (input_id, output_id)
            if key in connections:
                bias_matrix[output_name][input_name] = connections[key]
            else:
                bias_matrix[output_name][input_name] = 0.0

    return bias_matrix


def plot_genome(analysis, output_path):
    """Create a comprehensive genome visualization."""
    importance, conn_counts = compute_feature_importance(analysis)
    bias_matrix = compute_output_bias(analysis)
    connections = analysis["connections"]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(COLORS["bg"])

    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.96, top=0.92, bottom=0.06)

    # ── Panel 1: Feature Importance Bar Chart ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS["card"])

    if importance:
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]

        # Color code: sentiment in cyan, technical in blue, position in purple
        bar_colors = []
        for name in names:
            if "sentiment" in name:
                bar_colors.append(COLORS["cyan"])
            elif name in ["position_flag", "unrealized_pnl"]:
                bar_colors.append(COLORS["purple"])
            else:
                bar_colors.append(COLORS["blue"])

        bars = ax1.barh(range(len(names)), values, color=bar_colors, alpha=0.85)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=10, color=COLORS["text"])
        ax1.invert_yaxis()
        ax1.set_xlabel("Summed |Weight|", color=COLORS["dim"], fontsize=10)
        ax1.set_title("FEATURE IMPORTANCE", fontsize=13, fontweight="bold",
                      color=COLORS["white"], pad=10)

        # Add value labels
        for bar, val in zip(bars, values):
            ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=9, color=COLORS["dim"])
    else:
        ax1.text(0.5, 0.5, "No direct connections from inputs", transform=ax1.transAxes,
                ha="center", va="center", color=COLORS["dim"], fontsize=12)

    for spine in ax1.spines.values():
        spine.set_color(COLORS["grid"])
    ax1.tick_params(colors=COLORS["dim"])
    ax1.grid(True, alpha=0.1, axis="x", color=COLORS["dim"])

    # ── Panel 2: Connection Heatmap (Input → Output) ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLORS["card"])

    output_names = list(bias_matrix.keys())
    input_names = list(INPUT_LABELS)
    matrix = np.zeros((len(input_names), len(output_names)))

    for j, out_name in enumerate(output_names):
        for i, in_name in enumerate(input_names):
            matrix[i, j] = bias_matrix.get(out_name, {}).get(in_name, 0.0)

    max_abs = max(0.01, np.max(np.abs(matrix)))
    im = ax2.imshow(matrix, cmap="RdYlGn", aspect="auto",
                    vmin=-max_abs, vmax=max_abs, interpolation="nearest")

    ax2.set_xticks(range(len(output_names)))
    ax2.set_xticklabels(output_names, fontsize=11, fontweight="bold", color=COLORS["text"])
    ax2.set_yticks(range(len(input_names)))
    ax2.set_yticklabels(input_names, fontsize=10, color=COLORS["text"])
    ax2.set_title("INPUT → OUTPUT WEIGHT MATRIX", fontsize=13, fontweight="bold",
                  color=COLORS["white"], pad=10)

    # Add weight values in cells
    for i in range(len(input_names)):
        for j in range(len(output_names)):
            val = matrix[i, j]
            if abs(val) > 0.01:
                color = COLORS["white"] if abs(val) > max_abs * 0.5 else COLORS["dim"]
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(colors=COLORS["dim"])
    cbar.set_label("Weight", color=COLORS["dim"], fontsize=10)

    # ── Panel 3: Network Topology ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(COLORS["card"])
    ax3.set_xlim(-0.5, 3.5)
    ax3.set_ylim(-0.5, max(len(INPUT_LABELS), 5) + 0.5)
    ax3.axis("off")
    ax3.set_title("NETWORK TOPOLOGY", fontsize=13, fontweight="bold",
                  color=COLORS["white"], pad=10)

    # Draw layers
    input_x, hidden_x, output_x = 0.3, 1.7, 3.0
    num_inputs = len(INPUT_LABELS)
    num_hidden = len(analysis["hidden_nodes"])
    num_outputs = len(OUTPUT_LABELS)

    # Position nodes
    input_positions = {}
    for i, (node_id, name) in enumerate(zip(analysis["input_nodes"], INPUT_LABELS)):
        y = (num_inputs - 1 - i) * (num_inputs / (num_inputs + 1)) + 0.5
        input_positions[node_id] = (input_x, y)
        ax3.add_patch(plt.Circle((input_x, y), 0.18, color=COLORS["blue"], zorder=5))
        ax3.text(input_x - 0.35, y, name, ha="right", va="center",
                fontsize=7, color=COLORS["text"])

    output_positions = {}
    output_y_start = (num_inputs - num_outputs) / 2
    for i, (node_id, name) in enumerate(zip(analysis["output_nodes"], OUTPUT_LABELS)):
        y = output_y_start + i * 2.5 + 1
        output_positions[node_id] = (output_x, y)
        color = COLORS["green"] if name == "BUY" else COLORS["red"] if name == "SELL" else COLORS["amber"]
        ax3.add_patch(plt.Circle((output_x, y), 0.22, color=color, zorder=5))
        ax3.text(output_x + 0.35, y, name, ha="left", va="center",
                fontsize=10, color=color, fontweight="bold")

    hidden_positions = {}
    if num_hidden > 0:
        for i, node_id in enumerate(analysis["hidden_nodes"]):
            y = (num_inputs / (num_hidden + 1)) * (i + 1)
            hidden_positions[node_id] = (hidden_x, y)
            ax3.add_patch(plt.Circle((hidden_x, y), 0.15, color=COLORS["purple"], zorder=5))
            ax3.text(hidden_x, y, str(node_id), ha="center", va="center",
                    fontsize=7, color=COLORS["white"], zorder=6)

    all_positions = {**input_positions, **output_positions, **hidden_positions}

    # Draw connections
    for (src, dst), weight in connections.items():
        if src in all_positions and dst in all_positions:
            x1, y1 = all_positions[src]
            x2, y2 = all_positions[dst]
            color = COLORS["green"] if weight > 0 else COLORS["red"]
            alpha = min(1.0, abs(weight) / max(1, max(abs(w) for w in connections.values())) + 0.15)
            linewidth = max(0.5, min(3.0, abs(weight)))
            ax3.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                    linewidth=linewidth, zorder=1)

    # Layer labels
    ax3.text(input_x, -0.3, f"INPUTS ({num_inputs})", ha="center",
            fontsize=9, color=COLORS["dim"], fontweight="bold")
    if num_hidden > 0:
        ax3.text(hidden_x, -0.3, f"HIDDEN ({num_hidden})", ha="center",
                fontsize=9, color=COLORS["dim"], fontweight="bold")
    ax3.text(output_x, -0.3, f"OUTPUTS ({num_outputs})", ha="center",
            fontsize=9, color=COLORS["dim"], fontweight="bold")

    # ── Panel 4: Stats Summary ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(COLORS["card"])
    ax4.axis("off")
    ax4.set_title("GENOME STATS", fontsize=13, fontweight="bold",
                  color=COLORS["white"], pad=10)

    genome = analysis["genome"]
    num_connections = len([c for c in genome.connections.values() if c.enabled])
    num_disabled = len([c for c in genome.connections.values() if not c.enabled])
    weights = [c.weight for c in genome.connections.values() if c.enabled]

    stats = [
        ("Input Nodes", str(len(analysis["input_nodes"]))),
        ("Hidden Nodes", str(len(analysis["hidden_nodes"]))),
        ("Output Nodes", str(len(analysis["output_nodes"]))),
        ("Active Connections", str(num_connections)),
        ("Disabled Connections", str(num_disabled)),
        ("Total Complexity", f"{len(analysis['hidden_nodes'])} + {num_connections}"),
        ("", ""),
        ("Weight Range", f"[{min(weights):.2f}, {max(weights):.2f}]" if weights else "N/A"),
        ("Mean |Weight|", f"{np.mean(np.abs(weights)):.3f}" if weights else "N/A"),
        ("Std Weight", f"{np.std(weights):.3f}" if weights else "N/A"),
    ]

    # Top feature
    if importance:
        top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[0]
        stats.append(("", ""))
        stats.append(("Top Feature", f"{top[0]} ({top[1]:.2f})"))
        # Is sentiment important?
        sent_imp = importance.get("sentiment", 0)
        total_imp = sum(importance.values())
        sent_pct = (sent_imp / total_imp * 100) if total_imp > 0 else 0
        stats.append(("Sentiment Weight%", f"{sent_pct:.1f}%"))

    for i, (label, value) in enumerate(stats):
        y = 0.92 - i * 0.065
        if label:
            ax4.text(0.05, y, label, transform=ax4.transAxes, fontsize=11,
                    color=COLORS["dim"], va="center")
            ax4.text(0.95, y, value, transform=ax4.transAxes, fontsize=11,
                    color=COLORS["text"], va="center", ha="right", fontweight="bold")
        else:
            ax4.axhline(y=y, xmin=0.05, xmax=0.95, color=COLORS["grid"],
                        linewidth=0.5)

    # Title
    fig.suptitle("EVOLVED NEURAL NETWORK — BEST GENOME",
                fontsize=18, fontweight="bold", color=COLORS["white"], y=0.97)

    output_path = str(PROJECT_ROOT / "backtest_results" / "genome_visualization.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"📊 Visualization saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", type=str, default=None)
    args = parser.parse_args()

    genome = load_genome(args.genome)

    config_path = str(PROJECT_ROOT / "agents" / "config-trader")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    analysis = analyze_genome(genome, config)

    print(f"\n🧬 Genome Topology:")
    print(f"   Inputs:  {len(analysis['input_nodes'])}")
    print(f"   Hidden:  {len(analysis['hidden_nodes'])}")
    print(f"   Outputs: {len(analysis['output_nodes'])}")
    print(f"   Connections: {len(analysis['connections'])}")

    importance, _ = compute_feature_importance(analysis)
    if importance:
        print(f"\n📊 Feature Importance (by connection weight):")
        for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 5)
            print(f"   {name:20s} {imp:6.2f} {bar}")

    output_path = plot_genome(analysis, None)

    # Print sentiment analysis
    sent_imp = importance.get("sentiment", 0)
    total_imp = sum(importance.values()) if importance else 1
    print(f"\n🧠 Sentiment Analysis:")
    print(f"   Sentiment importance: {sent_imp:.2f} ({sent_imp/total_imp*100:.1f}% of total)")
    if sent_imp / total_imp > 0.15:
        print(f"   ✅ The AI IS using news sentiment significantly!")
    elif sent_imp / total_imp > 0.05:
        print(f"   ⚠️  The AI uses sentiment moderately")
    else:
        print(f"   ❌ The AI is mostly ignoring sentiment")


if __name__ == "__main__":
    main()
