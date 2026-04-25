"""
base_agent.py -- Abstract base class for all agents.
"""

from abc import ABC, abstractmethod
from env.dice import DiceAction


class BaseAgent(ABC):

    @abstractmethod
    def choose_action(
        self,
        agent_pos:     int,
        opponent_pos:  int,
        agent_skip:    bool,
        opponent_skip: bool,
    ) -> DiceAction:
        """Choose a dice action given the current game state."""
        ...

    def on_step_end(self, result, reward: float):
        """Called after each step. Override in learning agents to update."""
        pass

    def on_episode_end(self):
        """Called at episode end. Override for episode-level bookkeeping."""
        pass
