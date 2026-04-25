"""
dice.py -- Dice rolling mechanics.

Each turn the agent has three possible dice actions:
  Action 0: RANDOM_ROLL  -- roll the dice normally (uniform 1-6)
  Action 1: CHOOSE_2     -- declare a roll of exactly 2
  Action 2: CHOOSE_3     -- declare a roll of exactly 3

The 2/3 choice is the agent's "safe move" -- it trades randomness for
control, at the cost of potentially slower progress.

We use numpy's PCG64 generator (superior statistical properties vs
Python's Mersenne Twister) for all random rolls.
"""

from enum import IntEnum
import numpy as np


class DiceAction(IntEnum):
    RANDOM_ROLL = 0
    CHOOSE_2    = 1
    CHOOSE_3    = 2


# Human-readable names for logging
DICE_ACTION_NAMES = {
    DiceAction.RANDOM_ROLL: "random",
    DiceAction.CHOOSE_2:    "choose_2",
    DiceAction.CHOOSE_3:    "choose_3",
}


class DiceEngine:
    """
    Handles all dice rolling with a seeded, high-quality RNG.
    Seed can be fixed for reproducible experiments.
    """

    def __init__(self, seed: int | None = None):
        # PCG64 is the default BitGenerator in modern numpy.
        # It has excellent statistical properties and passes
        # TestU01's BigCrush suite -- far better than Mersenne Twister.
        self.rng = np.random.default_rng(seed)
        self._seed = seed

    def roll(self, action: DiceAction) -> int:
        """Execute a dice action and return the resulting value (1-6)."""
        if action == DiceAction.RANDOM_ROLL:
            return int(self.rng.integers(1, 7))  # uniform [1, 6]
        elif action == DiceAction.CHOOSE_2:
            return 2
        elif action == DiceAction.CHOOSE_3:
            return 3
        else:
            raise ValueError(f"Unknown DiceAction: {action}")

    def reset(self, seed: int | None = None):
        """Re-seed the RNG. Call between experiment runs for reproducibility."""
        seed = seed if seed is not None else self._seed
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def action_name(action: DiceAction) -> str:
        return DICE_ACTION_NAMES[action]
