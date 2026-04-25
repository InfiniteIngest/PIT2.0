"""
rl_agent.py -- Tabular Q-learning agent for Piticot.

State representation:
    (agent_pos, opponent_pos, agent_skip, opponent_skip)
    agent_pos, opponent_pos: 0-indexed [0, 64]
    agent_skip, opponent_skip: 0 or 1

    Total states: 65 x 65 x 2 x 2 = 16,900

Action space: 3 (RANDOM_ROLL, CHOOSE_2, CHOOSE_3)

Q-table shape: [65, 65, 2, 2, 3]

The agent uses eps-greedy exploration with exponential decay:
  - High eps early -> explore freely
  - Low eps later  -> exploit learned strategy

A "proximity heuristic" is tracked for logging: when the agent is
within striking distance of square 24 AND is losing, does it steer
toward 24? This is the core Nash Equilibrium behavior we're studying.
"""

from __future__ import annotations
import numpy as np
from agents.base_agent import BaseAgent
from env.dice import DiceAction


class QAgent(BaseAgent):
    """
    Tabular Q-learning agent.

    Parameters
    ----------
    alpha:          Learning rate (how fast Q-values update)
    gamma:          Discount factor (how much future rewards matter)
    epsilon_start:  Initial exploration rate
    epsilon_end:    Minimum exploration rate
    epsilon_decay:  Multiplicative decay per episode
    seed:           RNG seed for reproducibility
    """

    def __init__(
        self,
        alpha:          float = 0.1,
        gamma:          float = 0.95,
        epsilon_start:  float = 1.0,
        epsilon_end:    float = 0.02,
        epsilon_decay:  float = 0.9995,
        seed:           int | None = None,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)

        # Q-table: [pos_agent, pos_opp, skip_agent, skip_opp, action]
        self.q_table = np.zeros((65, 65, 2, 2, 3), dtype=np.float32)

        # -- Tracking ------------------------------------------------------
        self.episode           = 0
        self.total_steps       = 0
        self._last_state_idx:  tuple | None = None
        self._last_action:     DiceAction | None = None

    # -- BaseAgent interface ------------------------------------------------

    def choose_action(
        self,
        agent_pos:     int,
        opponent_pos:  int,
        agent_skip:    bool,
        opponent_skip: bool,
    ) -> DiceAction:
        state_idx = self._to_idx(agent_pos, opponent_pos, agent_skip, opponent_skip)
        self._last_state_idx = state_idx

        if self.rng.random() < self.epsilon:
            action = DiceAction(int(self.rng.integers(0, 3)))
        else:
            action = DiceAction(int(np.argmax(self.q_table[state_idx])))

        self._last_action = action
        self.total_steps += 1
        return action

    def on_step_end(self, result, reward: float):
        """Update Q-table using the Bellman equation."""
        if self._last_state_idx is None or self._last_action is None:
            return

        next_idx = self._to_idx(
            result.agent_pos, result.opponent_pos,
            result.agent_skip, result.opponent_skip,
        )

        old_q    = self.q_table[self._last_state_idx][int(self._last_action)]
        next_max = 0.0 if result.done else float(np.max(self.q_table[next_idx]))
        target   = reward + self.gamma * next_max
        new_q    = old_q + self.alpha * (target - old_q)

        self.q_table[self._last_state_idx][int(self._last_action)] = new_q

    def on_episode_end(self):
        """Decay epsilon after each episode."""
        self.episode += 1
        self.epsilon  = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # -- Q-table utilities -------------------------------------------------

    def best_action_at(self, agent_pos: int, opponent_pos: int) -> DiceAction:
        """Return the greedy best action (no exploration) for a given state."""
        idx = self._to_idx(agent_pos, opponent_pos, False, False)
        return DiceAction(int(np.argmax(self.q_table[idx])))

    def q_values_at(self, agent_pos: int, opponent_pos: int) -> np.ndarray:
        """Return all three Q-values for a given state (no skip flags)."""
        idx = self._to_idx(agent_pos, opponent_pos, False, False)
        return self.q_table[idx].copy()

    def save(self, path: str):
        """Save Q-table to disk."""
        np.save(path, self.q_table)
        print(f"Q-table saved -> {path}")

    def load(self, path: str):
        """Load Q-table from disk."""
        self.q_table = np.load(path)
        print(f"Q-table loaded <- {path}")

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _to_idx(
        agent_pos: int, opponent_pos: int,
        agent_skip: bool, opponent_skip: bool,
    ) -> tuple[int, int, int, int]:
        return (
            agent_pos    - 1,   # 1-indexed -> 0-indexed
            opponent_pos - 1,
            int(agent_skip),
            int(opponent_skip),
        )
