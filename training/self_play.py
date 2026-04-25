"""
self_play.py -- Main training loop for the Piticot Nash experiment.

Supports three training modes:
  1. self_play:   Agent plays against a copy of itself (two Q-agents)
  2. vs_random:   Agent plays against a RandomAgent baseline
  3. vs_uniform:  Agent plays against a UniformRandomAgent

The training loop sweeps over MORAL_WEIGHT values and saves separate
Q-tables and logs for each value -- this is the core experimental design.
"""

from __future__ import annotations
import os
import sys
import time
import json
import numpy as np
from datetime import datetime

# Make sure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.piticot_env  import PiticotEnv, Outcome
from env.dice         import DiceAction
from agents.rl_agent  import QAgent
from agents.random_agent import RandomAgent, UniformRandomAgent
from reward.reward_fn import RewardConfig, compute_reward
from analysis.strategy_logger import StrategyLogger


def run_experiment(
    moral_weight:   float,
    episodes:       int       = 200_000,
    mode:           str       = "self_play",   # "self_play" | "vs_random" | "vs_uniform"
    seed:           int       = 42,
    output_dir:     str       = "data/runs",
    log_every_n:    int       = 200,           # log step-data every N episodes
    save_every_n:   int       = 50_000,        # save Q-table checkpoint every N episodes
    alpha:          float     = 0.1,
    gamma:          float     = 0.95,
    epsilon_start:  float     = 1.0,
    epsilon_end:    float     = 0.02,
    epsilon_decay:  float     = 0.9995,
) -> QAgent:
    """
    Train a Q-agent for a given moral_weight and return the trained agent.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id    = f"mw{moral_weight:.2f}_{mode}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"  Run ID:       {run_id}")
    print(f"  Moral weight: {moral_weight}")
    print(f"  Mode:         {mode}")
    print(f"  Episodes:     {episodes:,}")
    print(f"{'='*60}")

    config  = RewardConfig(MORAL_WEIGHT=moral_weight)
    env     = PiticotEnv(seed=seed)
    agent   = QAgent(alpha=alpha, gamma=gamma, epsilon_start=epsilon_start,
                     epsilon_end=epsilon_end, epsilon_decay=epsilon_decay, seed=seed)
    logger  = StrategyLogger(run_id=run_id, output_dir=output_dir, log_every_n=log_every_n)

    # Opponent setup
    if mode == "self_play":
        opponent = agent   # same agent object plays both sides (shared Q-table)
    elif mode == "vs_random":
        opponent = RandomAgent(seed=seed + 1)
    elif mode == "vs_uniform":
        opponent = UniformRandomAgent(seed=seed + 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # -- Stats tracking --------------------------------------------------
    outcomes = {Outcome.WIN: 0, Outcome.LOSS: 0, Outcome.MUTUAL_LOSS: 0}
    t_start  = time.time()

    # -- Training loop ---------------------------------------------------
    for ep in range(episodes):
        state = env.reset()
        agent_pos, opp_pos, agent_skip, opp_skip = state
        done  = False
        logger.on_episode_start(ep)

        while not done:
            # Agent chooses action
            agent_action = agent.choose_action(agent_pos, opp_pos, agent_skip, opp_skip)

            # Opponent chooses action.
            # In all modes the opponent sees the board from its own perspective
            # (positions swapped so its "agent_pos" is actually opp_pos, etc.)
            opp_action = opponent.choose_action(opp_pos, agent_pos, opp_skip, agent_skip)

            # Step environment
            result = env.step(agent_action, opp_action)

            # Compute reward
            reward, breakdown = compute_reward(
                outcome=result.outcome,
                agent_pos=result.agent_pos,
                opponent_pos=result.opponent_pos,
                agent_dice_action=agent_action,
                config=config,
            )

            # Get Q-values for logging
            q_vals = agent.q_values_at(agent_pos, opp_pos).tolist()

            # Log this step
            logger.log_step(
                moral_weight=moral_weight,
                agent_pos=agent_pos,
                opponent_pos=opp_pos,
                agent_skip=agent_skip,
                opponent_skip=opp_skip,
                action=agent_action,
                roll_result=result.agent_roll,
                reward=reward,
                outcome=result.outcome,
                q_values=q_vals,
            )

            # Update Q-table
            agent.on_step_end(result, reward)

            # Update state
            agent_pos  = result.agent_pos
            opp_pos    = result.opponent_pos
            agent_skip = result.agent_skip
            opp_skip   = result.opponent_skip
            done       = result.done

        # -- Episode end -------------------------------------------------
        outcome_key = result.outcome
        outcomes[outcome_key] = outcomes.get(outcome_key, 0) + 1

        logger.on_episode_end(
            moral_weight=moral_weight,
            outcome=result.outcome,
            agent_final_pos=result.agent_pos,
            opp_final_pos=result.opponent_pos,
            epsilon=agent.epsilon,
        )
        agent.on_episode_end()

        # -- Periodic checkpoint + progress report ------------------------
        if (ep + 1) % save_every_n == 0:
            ckpt_path = os.path.join(output_dir, f"{run_id}_ep{ep+1}.npy")
            agent.save(ckpt_path)

        if (ep + 1) % 10_000 == 0:
            elapsed = time.time() - t_start
            total   = ep + 1
            win_pct = outcomes[Outcome.WIN]         / total * 100
            los_pct = outcomes[Outcome.LOSS]        / total * 100
            mut_pct = outcomes[Outcome.MUTUAL_LOSS] / total * 100
            print(
                f"  Ep {total:>7,} | eps={agent.epsilon:.4f} | "
                f"W={win_pct:5.1f}% L={los_pct:5.1f}% M24={mut_pct:5.1f}% | "
                f"{elapsed:.0f}s elapsed"
            )

    # -- Final save ------------------------------------------------------
    final_path = os.path.join(output_dir, f"{run_id}_FINAL.npy")
    agent.save(final_path)
    logger.close()

    # Save run metadata
    meta = {
        "run_id":       run_id,
        "moral_weight": moral_weight,
        "mode":         mode,
        "episodes":     episodes,
        "seed":         seed,
        "alpha":        alpha,
        "gamma":        gamma,
        "outcomes":     outcomes,
        "duration_s":   time.time() - t_start,
        "q_table_path": final_path,
    }
    meta_path = os.path.join(output_dir, f"{run_id}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Run complete. Metadata -> {meta_path}")
    return agent


def sweep_moral_weights(
    weights:     list[float] = [0.0, 0.2, 0.5, 1.0, 2.0],
    episodes:    int         = 200_000,
    mode:        str         = "self_play",
    seed:        int         = 42,
    output_dir:  str         = "data/runs",
):
    """
    Run the full experimental sweep across moral_weight values.
    This is the main experiment -- call this to produce all research data.
    """
    print(f"\n{'#'*60}")
    print(f"  PITICOT NASH EXPERIMENT -- MORAL WEIGHT SWEEP")
    print(f"  Weights:  {weights}")
    print(f"  Episodes: {episodes:,} per weight")
    print(f"  Mode:     {mode}")
    print(f"{'#'*60}")

    results = {}
    for w in weights:
        agent = run_experiment(
            moral_weight=w,
            episodes=episodes,
            mode=mode,
            seed=seed,
            output_dir=output_dir,
        )
        results[w] = agent

    print(f"\n{'#'*60}")
    print(f"  ALL RUNS COMPLETE. Data in: {output_dir}/")
    print(f"{'#'*60}")
    return results


# -- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Piticot Nash Experiment Trainer")
    parser.add_argument("--mode",      default="self_play",
                        choices=["self_play", "vs_random", "vs_uniform"])
    parser.add_argument("--episodes",  type=int,   default=200_000)
    parser.add_argument("--sweep",     action="store_true",
                        help="Run full moral weight sweep (recommended)")
    parser.add_argument("--weight",    type=float, default=0.0,
                        help="Single moral weight (if not sweeping)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--output",    default="data/runs")
    args = parser.parse_args()

    if args.sweep:
        sweep_moral_weights(
            weights=[0.0, 0.2, 0.5, 1.0, 2.0],
            episodes=args.episodes,
            mode=args.mode,
            seed=args.seed,
            output_dir=args.output,
        )
    else:
        run_experiment(
            moral_weight=args.weight,
            episodes=args.episodes,
            mode=args.mode,
            seed=args.seed,
            output_dir=args.output,
        )
