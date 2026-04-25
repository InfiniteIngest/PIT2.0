"""
nash_analysis.py -- Post-training analysis script.

Run this after training to produce charts and statistics from the CSV logs.
Can also be opened as a Jupyter notebook (each section maps to a cell).

Usage:
    python notebooks/nash_analysis.py --data data/runs/

Produces:
    - Win/loss/mutual_loss rates per moral weight
    - Square 24 targeting rate vs. moral weight
    - Q-value heatmaps (strategy maps)
    - Action distribution over time (does strategy stabilize?)
    - Nash Equilibrium detection: does the strategy converge?
"""

from __future__ import annotations
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for cmd redirect
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -- Style --------------------------------------------------------------------
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.dpi":      150,
    "figure.figsize":  (12, 6),
    "font.family":     "monospace",
    "axes.titlesize":  13,
    "axes.labelsize":  11,
})


# -- Data loading -------------------------------------------------------------

def load_all_episodes(data_dir: str) -> pd.DataFrame:
    """Load all *_episodes.csv files from a run directory into one DataFrame."""
    files = glob.glob(os.path.join(data_dir, "*_episodes.csv"))
    if not files:
        raise FileNotFoundError(f"No episode CSV files found in {data_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} episode records from {len(files)} run(s).")
    return combined


def load_all_decisions(data_dir: str) -> pd.DataFrame:
    """Load all *_decisions.csv files."""
    files = glob.glob(os.path.join(data_dir, "*_decisions.csv"))
    if not files:
        raise FileNotFoundError(f"No decision CSV files found in {data_dir}")
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} decision records from {len(files)} run(s).")
    return combined


def load_meta_files(data_dir: str) -> list[dict]:
    files = glob.glob(os.path.join(data_dir, "*_meta.json"))
    metas = []
    for f in files:
        with open(f) as fp:
            metas.append(json.load(fp))
    return metas


# -- Analysis functions --------------------------------------------------------

def plot_outcome_rates(ep_df: pd.DataFrame, output_dir: str):
    """Line chart: Win/Loss/Mutual_Loss rates for each moral weight.
    Filtered to self_play mode only so all runs are comparable.
    vs_random and vs_uniform runs are Phase 5 comparison data and
    must not be mixed into the moral weight sweep chart.
    """
    # Filter to self_play only -- the controlled experiment
    sp_df = ep_df[ep_df["run_id"].str.contains("self_play")]
    if sp_df.empty:
        print("  No self_play runs found for outcome_rates chart -- skipping.")
        return
    last_20 = _last_n_pct(sp_df)

    rates = (
        last_20.groupby("moral_weight")["outcome"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("rate_pct")
        .reset_index()
    )

    fig, ax = plt.subplots()
    outcome_colors = {"win": "#4caf50", "loss": "#f44336", "mutual_loss": "#ff9800"}
    for outcome, color in outcome_colors.items():
        subset = rates[rates["outcome"] == outcome]
        ax.plot(subset["moral_weight"], subset["rate_pct"], "o-",
                label=outcome.replace("_", " ").title(), color=color, lw=2, ms=8)

    ax.set_xlabel("Moral Weight")
    ax.set_ylabel("Rate (%) -- last 20% of training")
    ax.set_title("Outcome Rates vs. Moral Weight\n(higher moral weight = more guilt about reaching square 24)")
    ax.legend()
    xticks = sorted(rates["moral_weight"].unique().tolist())
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(round(t, 2)) for t in xticks])
    try:
        plt.tight_layout()
    except Exception:
        pass
    path = os.path.join(output_dir, "outcome_rates.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved -> {path}")


def plot_square24_targeting(ep_df: pd.DataFrame, output_dir: str):
    # Filter to self_play only for the controlled moral weight comparison
    ep_df = ep_df[ep_df["run_id"].str.contains("self_play")].copy()
    """
    Key research chart: Does the agent deliberately target square 24 when losing?

    We filter to episodes where the agent was losing (opponent_final_pos > agent_final_pos)
    and count how often it triggered square 24.
    """
    losing_ep = ep_df[ep_df["opponent_final_pos"] > ep_df["agent_final_pos"]].copy()
    if losing_ep.empty:
        print("No losing episodes found -- skipping square 24 targeting chart.")
        return

    # Rolling 24-trigger rate grouped by moral weight, over training
    fig, ax = plt.subplots()
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, ep_df["moral_weight"].nunique()))

    for (mw, color) in zip(sorted(ep_df["moral_weight"].unique()), colors):
        subset = losing_ep[losing_ep["moral_weight"] == mw].copy()
        subset = subset.sort_values("episode")
        window = max(1, min(1000, len(subset) // 5))
        subset["hit_24_rolling"] = subset["hit_square_24"].rolling(
            window=window, min_periods=min(50, window)
        ).mean() * 100

        ax.plot(subset["episode"], subset["hit_24_rolling"],
                label=f"MW={mw}", color=color, lw=1.5, alpha=0.85)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Square 24 Trigger Rate (%) -- rolling avg, when losing")
    ax.set_title("Nash Equilibrium Detection:\nDoes the Agent Sabotage When Losing?")
    ax.legend(title="Moral Weight", loc="upper right")
    ax.axhline(0, color="white", lw=0.5, ls="--")
    try:
        plt.tight_layout()
    except Exception:
        pass
    path = os.path.join(output_dir, "square24_targeting.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved -> {path}")


def plot_action_distribution(ep_df: pd.DataFrame, output_dir: str):
    """Stacked area: How do action choices evolve over training?"""
    fig, axes = plt.subplots(1, len(ep_df["moral_weight"].unique()),
                             figsize=(16, 5), sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]

    for ax, mw in zip(axes, sorted(ep_df["moral_weight"].unique())):
        subset = ep_df[ep_df["moral_weight"] == mw].sort_values("episode").copy()
        w = max(1000, len(subset) // 10)

        for col, color, label in [
            ("random_rolls",   "#2196f3", "Random Roll"),
            ("choose_2_rolls", "#ff9800", "Choose 2"),
            ("choose_3_rolls", "#9c27b0", "Choose 3"),
        ]:
            rolled = subset[col].rolling(window=w, min_periods=100).mean()
            ax.plot(subset["episode"], rolled, label=label, color=color, lw=1.5)

        ax.set_title(f"MW = {mw}")
        ax.set_xlabel("Episode")
        if ax == axes[0]:
            ax.set_ylabel("Avg actions per episode")
        ax.legend(fontsize=7)

    plt.suptitle("Action Distribution Over Training (Rolling Average)", y=1.02)
    try:
        plt.tight_layout()
    except Exception:
        pass
    path = os.path.join(output_dir, "action_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved -> {path}")


def plot_strategy_heatmap(data_dir: str, output_dir: str):
    """
    Q-value heatmap: For each board position (agent vs opponent), what
    action does the trained agent prefer?

    Loads the FINAL Q-tables and visualizes the greedy policy.
    """
    q_files = glob.glob(os.path.join(data_dir, "*_FINAL.npy"))
    if not q_files:
        print("No FINAL Q-tables found for heatmap.")
        return

    for q_path in q_files:
        run_id  = Path(q_path).stem.replace("_FINAL", "")
        q_table = np.load(q_path)   # shape: [65, 65, 2, 2, 3]

        # Use no-skip state (most common)
        q_noskip = q_table[:, :, 0, 0, :]   # shape: [65, 65, 3]
        best_action = np.argmax(q_noskip, axis=2)   # 0=random, 1=choose2, 2=choose3

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Heatmap of best action
        cmap   = mcolors.ListedColormap(["#2196f3", "#ff9800", "#9c27b0"])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm   = mcolors.BoundaryNorm(bounds, cmap.N)

        im = ax1.imshow(best_action, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        cbar = plt.colorbar(im, ax=ax1, ticks=[0, 1, 2])
        cbar.set_ticklabels(["Random", "Choose 2", "Choose 3"])
        ax1.set_xlabel("Opponent Position")
        ax1.set_ylabel("Agent Position")
        ax1.set_title(f"Greedy Policy Map\n{run_id}")

        # Highlight the "danger zone" near square 24
        ax1.axhline(y=20, color="red", lw=1, ls="--", alpha=0.5, label="~sq 24 range")
        ax1.axvline(x=20, color="orange", lw=1, ls="--", alpha=0.5)
        ax1.legend(fontsize=8)

        # Heatmap of max Q-value (confidence)
        max_q = np.max(q_noskip, axis=2)
        im2   = ax2.imshow(max_q, cmap="viridis", origin="lower", aspect="auto")
        plt.colorbar(im2, ax=ax2, label="Max Q-value")
        ax2.set_xlabel("Opponent Position")
        ax2.set_ylabel("Agent Position")
        ax2.set_title(f"Q-value Confidence\n{run_id}")

        try:
            plt.tight_layout()
        except Exception:
            pass
        path = os.path.join(output_dir, f"heatmap_{run_id}.png")
        plt.savefig(path)
        plt.close()
        print(f"Saved -> {path}")


def _last_n_pct(ep_df: pd.DataFrame, pct: float = 0.2) -> pd.DataFrame:
    parts = []
    for (run_id, mw), g in ep_df.groupby(["run_id", "moral_weight"]):
        tail = g.tail(max(1, int(len(g) * pct))).copy()
        tail["run_id"] = run_id
        tail["moral_weight"] = mw
        parts.append(tail)
    return pd.concat(parts, ignore_index=True) if parts else ep_df.iloc[:0]


def print_summary_table(ep_df: pd.DataFrame):
    """Print a summary table to the terminal."""
    last_20 = _last_n_pct(ep_df)

    print("\n" + "="*70)
    print("  SUMMARY TABLE (last 20% of training per run)")
    print("="*70)
    print(f"  {'MW':>6}  {'Win%':>7}  {'Loss%':>7}  {'M24%':>7}  {'Avg Steps':>10}  {'24 Rate (losing)':>18}")
    print("-"*70)

    for mw in sorted(last_20["moral_weight"].unique()):
        g = last_20[last_20["moral_weight"] == mw]
        win_pct  = (g["outcome"] == "win").mean()         * 100
        los_pct  = (g["outcome"] == "loss").mean()        * 100
        m24_pct  = (g["outcome"] == "mutual_loss").mean() * 100
        avg_step = g["steps"].mean()

        losing   = g[g["opponent_final_pos"] > g["agent_final_pos"]]
        rate_24  = losing["hit_square_24"].mean() * 100 if len(losing) > 0 else float("nan")

        print(f"  {mw:>6.2f}  {win_pct:>6.1f}%  {los_pct:>6.1f}%  {m24_pct:>6.1f}%  {avg_step:>10.1f}  {rate_24:>16.1f}%")

    print("="*70 + "\n")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Piticot Nash Analysis")
    parser.add_argument("--data",   default="data/runs", help="Directory with CSV/npy files")
    parser.add_argument("--output", default="data/plots", help="Where to save charts")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\nLoading data from: {args.data}")
    ep_df  = load_all_episodes(args.data)
    metas  = load_meta_files(args.data)
    if metas:
        print(f"Found {len(metas)} run metadata file(s).")
        for m in metas:
            # Phase 4 dynamic runs store 'schedule' instead of 'moral_weight'
            if 'moral_weight' in m:
                label = f"moral_weight={m['moral_weight']}"
            else:
                label = f"schedule={m.get('schedule', 'unknown')}"
            print(f"  run_id={m['run_id']}  {label}  episodes={m['episodes']}  duration={m.get('duration_s', 0):.0f}s")

    print_summary_table(ep_df)
    plot_outcome_rates(ep_df, args.output)
    plot_square24_targeting(ep_df, args.output)
    plot_action_distribution(ep_df, args.output)
    plot_strategy_heatmap(args.data, args.output)

    print(f"\nAll charts saved to: {args.output}/")
    print("Open them to study the Nash Equilibrium emergence.")


if __name__ == "__main__":
    main()