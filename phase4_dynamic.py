"""
phase4_dynamic.py -- Dynamic moral weight training (Phase 4).

Runs 4 training schedules where moral_weight changes during the run:
  growth   -- starts amoral (0.0), linearly increases to moral (1.0)
  decay    -- starts moral (1.0), linearly decreases to amoral (0.0)
  shock    -- amoral for first half, suddenly moral at episode 150,000
  cynicism -- moral for first half, suddenly amoral at episode 150,000

Run from the pit2.0\ directory:
    python phase4_dynamic.py
"""

import sys, os, json, time
sys.path.insert(0, '.')

from datetime import datetime
from env.piticot_env import PiticotEnv, Outcome
from env.dice import DiceAction
from agents.rl_agent import QAgent
from agents.random_agent import RandomAgent
from reward.reward_fn import RewardConfig, compute_reward
from analysis.strategy_logger import StrategyLogger


def get_dynamic_weight(episode: int, total_episodes: int, schedule: str) -> float:
    progress = episode / total_episodes
    if schedule == "growth":
        return progress * 1.0
    elif schedule == "decay":
        return (1.0 - progress) * 1.0
    elif schedule == "shock":
        return 0.0 if progress < 0.5 else 1.0
    elif schedule == "cynicism":
        return 1.0 if progress < 0.5 else 0.0
    return 0.0


def run_dynamic(schedule: str, episodes: int = 300_000, seed: int = 42, output_dir: str = "data/runs"):
    # Use timestamp in run_id so re-runs never overwrite previous data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"dynamic_{schedule}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"  Run ID:   {run_id}")
    print(f"  Schedule: {schedule}")
    print(f"  Episodes: {episodes:,}")
    print(f"{'='*60}")

    env      = PiticotEnv(seed=seed)
    agent    = QAgent(seed=seed)
    opponent = RandomAgent(seed=seed + 1)
    logger   = StrategyLogger(run_id=run_id, output_dir=output_dir, log_every_n=300)
    outcomes = {Outcome.WIN: 0, Outcome.LOSS: 0, Outcome.MUTUAL_LOSS: 0}
    t_start  = time.time()

    for ep in range(episodes):
        moral_weight = get_dynamic_weight(ep, episodes, schedule)
        config       = RewardConfig(MORAL_WEIGHT=moral_weight)

        state = env.reset()
        agent_pos, opp_pos, agent_skip, opp_skip = state
        done = False
        logger.on_episode_start(ep)

        while not done:
            agent_action = agent.choose_action(agent_pos, opp_pos, agent_skip, opp_skip)
            opp_action   = opponent.choose_action(opp_pos, agent_pos, opp_skip, agent_skip)
            result       = env.step(agent_action, opp_action)

            reward, _ = compute_reward(
                outcome=result.outcome,
                agent_pos=result.agent_pos,
                opponent_pos=result.opponent_pos,
                agent_dice_action=agent_action,
                config=config,
            )

            q_vals = agent.q_values_at(agent_pos, opp_pos).tolist()
            logger.log_step(
                moral_weight=moral_weight,
                agent_pos=agent_pos, opponent_pos=opp_pos,
                agent_skip=agent_skip, opponent_skip=opp_skip,
                action=agent_action, roll_result=result.agent_roll,
                reward=reward, outcome=result.outcome, q_values=q_vals,
            )

            agent.on_step_end(result, reward)
            agent_pos  = result.agent_pos
            opp_pos    = result.opponent_pos
            agent_skip = result.agent_skip
            opp_skip   = result.opponent_skip
            done       = result.done

        outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
        logger.on_episode_end(
            moral_weight=moral_weight, outcome=result.outcome,
            agent_final_pos=result.agent_pos, opp_final_pos=result.opponent_pos,
            epsilon=agent.epsilon,
        )
        agent.on_episode_end()

        if (ep + 1) % 10_000 == 0:
            elapsed = time.time() - t_start
            print(f"  Ep {ep+1:>7,} | mw={moral_weight:.3f} | eps={agent.epsilon:.4f} | {elapsed:.0f}s")

    final_path = os.path.join(output_dir, f"{run_id}_FINAL.npy")
    agent.save(final_path)
    logger.close()

    meta = {
        "run_id":     run_id,
        "schedule":   schedule,
        "moral_weight": None,   # dynamic -- see schedule field
        "episodes":   episodes,
        "seed":       seed,
        "outcomes":   outcomes,
        "duration_s": time.time() - t_start,
    }
    with open(os.path.join(output_dir, f"{run_id}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Done. Outcomes: {outcomes}")


if __name__ == "__main__":
    os.makedirs("data/runs", exist_ok=True)
    for schedule in ["growth", "decay", "shock", "cynicism"]:
        print(f"\nRunning schedule: {schedule}")
        run_dynamic(schedule=schedule, episodes=300_000, seed=42, output_dir="data/runs")
    print("\nPhase 4 complete.")
