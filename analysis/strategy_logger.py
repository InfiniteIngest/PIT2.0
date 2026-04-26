"""
strategy_logger.py -- Records every decision and outcome for post-hoc analysis.

Two levels of logging:
  1. Episode log (one row per game):
     - Who won, how many steps, was square 24 triggered, moral_weight, etc.

  2. Decision log (one row per step):
     - Exact state, action chosen, reward received, Q-values, etc.

Both are saved as CSV for easy analysis in pandas / Excel / R.
Logs are written incrementally.
"""

from __future__ import annotations
import csv
import os
import time
from dataclasses import dataclass, field, asdict
from env.dice import DiceAction, DICE_ACTION_NAMES
from env.piticot_env import Outcome


# -- Data structures --------------------------------------------------------

@dataclass
class EpisodeRecord:
    run_id:             str
    episode:            int
    moral_weight:       float
    outcome:            str      # win / loss / mutual_loss
    steps:              int
    agent_final_pos:    int
    opponent_final_pos: int
    hit_square_24:      bool     # did game end at 24?
    was_losing_at_24:   bool     # was agent behind when 24 was triggered?
    epsilon:            float    # exploration rate at this episode
    random_rolls:       int      # count of RANDOM_ROLL actions this episode
    choose_2_rolls:     int
    choose_3_rolls:     int
    timestamp:          float    = field(default_factory=time.time)


@dataclass
class DecisionRecord:
    run_id:             str
    episode:            int
    step:               int
    moral_weight:       float
    agent_pos:          int
    opponent_pos:       int
    gap:                int      # opponent_pos - agent_pos (positive = agent losing)
    agent_skip:         bool
    opponent_skip:      bool
    action:             str      # "random" / "choose_2" / "choose_3"
    roll_result:        int
    reward:             float
    outcome:            str
    q_random:           float    # Q-value for RANDOM_ROLL at this state
    q_choose2:          float    # Q-value for CHOOSE_2
    q_choose3:          float    # Q-value for CHOOSE_3
    near_24:            bool     # agent could reach 24 in 1-3 moves
    near_65:            bool     # agent could reach 65 in 1-3 moves
    opponent_near_65:   bool     # opponent could reach 65 in 1-3 moves


# -- Logger class ------------------------------------------------------------

class StrategyLogger:
    """
    Incrementally writes episode and decision records to CSV files.

    Parameters
    ----------
    run_id:       Unique identifier for this training run (used in filenames)
    output_dir:   Directory to write CSV files into
    log_every_n:  Only log step-level decisions every N episodes (reduces file size)
    """

    def __init__(
        self,
        run_id:      str,
        output_dir:  str = "data/runs",
        log_every_n: int = 100,
    ):
        self.run_id      = run_id
        self.output_dir  = output_dir
        self.log_every_n = log_every_n

        os.makedirs(output_dir, exist_ok=True)

        ep_path   = os.path.join(output_dir, f"{run_id}_episodes.csv")
        dec_path  = os.path.join(output_dir, f"{run_id}_decisions.csv")

        self._ep_file:  TextIO = open(ep_path,  "w", newline="")
        self._dec_file: TextIO = open(dec_path, "w", newline="")

        self._ep_writer  = csv.DictWriter(self._ep_file,  fieldnames=EpisodeRecord.__dataclass_fields__.keys())
        self._dec_writer = csv.DictWriter(self._dec_file, fieldnames=DecisionRecord.__dataclass_fields__.keys())

        self._ep_writer.writeheader()
        self._dec_writer.writeheader()

        # State for current episode
        self._current_episode:   int = 0
        self._step:              int = 0
        self._action_counts:     dict[str, int] = {"random": 0, "choose_2": 0, "choose_3": 0}
        self._hit_24:            bool = False
        self._was_losing_at_24:  bool = False

        print(f"Logger initialised. Writing to:")
        print(f"  Episodes  -> {ep_path}")
        print(f"  Decisions -> {dec_path}")

    # -- Public API ---------------------------------------------------------

    def on_episode_start(self, episode: int):
        self._current_episode  = episode
        self._step             = 0
        self._action_counts    = {"random": 0, "choose_2": 0, "choose_3": 0}
        self._hit_24           = False
        self._was_losing_at_24 = False

    def log_step(
        self,
        moral_weight:  float,
        agent_pos:     int,
        opponent_pos:  int,
        agent_skip:    bool,
        opponent_skip: bool,
        action:        DiceAction,
        roll_result:   int,
        reward:        float,
        outcome:       str,
        q_values:      list[float],   # [q_random, q_choose2, q_choose3]
    ):
        self._step += 1
        action_name = DICE_ACTION_NAMES[action]
        self._action_counts[action_name] += 1

        # Track square 24 events
        if outcome == Outcome.MUTUAL_LOSS:
            self._hit_24           = True
            self._was_losing_at_24 = opponent_pos > agent_pos

        # Proximity flags
        near_24        = 1 <= (24 - agent_pos) <= 3 if agent_pos < 24 else False
        near_65        = 1 <= (65 - agent_pos) <= 6
        opp_near_65    = 1 <= (65 - opponent_pos) <= 6

        record = DecisionRecord(
            run_id=self.run_id, episode=self._current_episode, step=self._step,
            moral_weight=moral_weight, agent_pos=agent_pos, opponent_pos=opponent_pos,
            gap=opponent_pos - agent_pos, agent_skip=agent_skip, opponent_skip=opponent_skip,
            action=action_name, roll_result=roll_result, reward=reward, outcome=outcome,
            q_random=q_values[0], q_choose2=q_values[1], q_choose3=q_values[2],
            near_24=near_24, near_65=near_65, opponent_near_65=opp_near_65,
        )

        # Only write step-level data every N episodes to keep file sizes reasonable
        if self._current_episode % self.log_every_n == 0:
            self._dec_writer.writerow(asdict(record))

    def on_episode_end(
        self,
        moral_weight:    float,
        outcome:         str,
        agent_final_pos: int,
        opp_final_pos:   int,
        epsilon:         float,
    ):
        rec = EpisodeRecord(
            run_id=self.run_id, episode=self._current_episode,
            moral_weight=moral_weight, outcome=outcome,
            steps=self._step,
            agent_final_pos=agent_final_pos, opponent_final_pos=opp_final_pos,
            hit_square_24=self._hit_24, was_losing_at_24=self._was_losing_at_24,
            epsilon=epsilon,
            random_rolls=self._action_counts["random"],
            choose_2_rolls=self._action_counts["choose_2"],
            choose_3_rolls=self._action_counts["choose_3"],
        )
        self._ep_writer.writerow(asdict(rec))
        # Flush periodically so data survives crashes
        if self._current_episode % 500 == 0:
            self._ep_file.flush()
            self._dec_file.flush()

    def close(self):
        self._ep_file.flush()
        self._dec_file.flush()
        self._ep_file.close()
        self._dec_file.close()
        print(f"Logger closed. All data written.")
