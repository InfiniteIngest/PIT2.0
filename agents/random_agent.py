"""
random_agent.py -- Baseline agent that always picks RANDOM_ROLL.

Used as a fixed opponent to test the learning agent against a
non-strategic player.
"""

import numpy as np
from agents.base_agent import BaseAgent
from env.dice import DiceAction


class RandomAgent(BaseAgent):
    """Always rolls the dice randomly. No strategy."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def choose_action(self, agent_pos, opponent_pos, agent_skip, opponent_skip) -> DiceAction:
        return DiceAction.RANDOM_ROLL


class UniformRandomAgent(BaseAgent):
    """
    Picks uniformly at random among all three dice actions.
    Slightly more interesting than RandomAgent as a baseline because it will occasionally use the 2/3 choice mechanic.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def choose_action(self, agent_pos, opponent_pos, agent_skip, opponent_skip) -> DiceAction:
        return DiceAction(int(self.rng.integers(0, 3)))
