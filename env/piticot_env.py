"""
piticot_env.py -- Gymnasium-compatible two-player Piticot environment.

State space:
  (agent_pos, opponent_pos, agent_skip_turn, opponent_skip_turn)
  Positions: 1-65 (we use 0 as "not yet started" internally but expose 1-based)
  Skip flags: 0 or 1

Action space (per turn, agent only):
  0 = RANDOM_ROLL
  1 = CHOOSE_2
  2 = CHOOSE_3

The opponent is controlled externally (pass an opponent agent to step()).
This allows self-play, random opponent, or fixed-strategy opponent.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from env.board import resolve_landing, SquareEffect, BOARD_MIN, BOARD_MAX
from env.dice import DiceEngine, DiceAction


# -- Outcome codes (used by reward_fn and loggers) --------------------------
class Outcome:
    WIN         = "win"
    LOSS        = "loss"
    MUTUAL_LOSS = "mutual_loss"
    ONGOING     = "ongoing"


@dataclass
class StepResult:
    """Everything that happened in one full round (both players moved)."""
    agent_pos:          int
    opponent_pos:       int
    agent_skip:         bool
    opponent_skip:      bool
    agent_dice_action:  DiceAction
    agent_roll:         int
    opponent_roll:      int
    outcome:            str          # Outcome.*
    done:               bool
    info:               dict = field(default_factory=dict)


class PiticotEnv:
    """
    Two-player Piticot environment.

    Usage:
        env = PiticotEnv(seed=42)
        state = env.reset()
        while not done:
            agent_action = agent.choose_action(state)
            result = env.step(agent_action, opponent_action)
            done = result.done
    """

    def __init__(self, seed: int | None = None):
        self.dice       = DiceEngine(seed=seed)
        self.seed       = seed
        self._reset_state()

    # -- Public API ---------------------------------------------------------

    def reset(self) -> tuple[int, int, bool, bool]:
        """Reset the board for a new episode. Returns initial state tuple."""
        self._reset_state()
        return self._state()

    def hard_reset(self, seed: int | None = None) -> tuple[int, int, bool, bool]:
        """Reset the board AND re-seed the RNG. Use only between separate
        experiment runs, never between episodes within the same run."""
        self.dice.reset(seed=seed if seed is not None else self.seed)
        self._reset_state()
        return self._state()

    def step(
        self,
        agent_action:    DiceAction,
        opponent_action: DiceAction,
    ) -> StepResult:
        """
        Advance one full round (agent moves, then opponent moves, unless
        the game ends mid-round).

        Returns a StepResult with full information about what happened.
        """
        assert not self.done, "Game is over. Call reset() first."

        info: dict = {}

        # -- Agent's turn --------------------------------------------------
        agent_roll = 0
        if self.agent_skip:
            # Agent is waiting this turn
            self.agent_skip = False
            info["agent_waited"] = True
        else:
            agent_roll = self.dice.roll(agent_action)
            raw         = self.agent_pos + agent_roll
            resolved    = resolve_landing(raw)
            self.agent_pos  = resolved.final_pos
            self.agent_skip = resolved.skip_turn

            if resolved.effect == SquareEffect.WIN:
                self.done = True
                return StepResult(
                    agent_pos=self.agent_pos, opponent_pos=self.opp_pos,
                    agent_skip=self.agent_skip, opponent_skip=self.opp_skip,
                    agent_dice_action=agent_action, agent_roll=agent_roll,
                    opponent_roll=0, outcome=Outcome.WIN, done=True, info=info,
                )

            if resolved.effect == SquareEffect.MUTUAL_LOSS:
                self.done = True
                return StepResult(
                    agent_pos=self.agent_pos, opponent_pos=self.opp_pos,
                    agent_skip=self.agent_skip, opponent_skip=self.opp_skip,
                    agent_dice_action=agent_action, agent_roll=agent_roll,
                    opponent_roll=0, outcome=Outcome.MUTUAL_LOSS, done=True, info=info,
                )

        # -- Opponent's turn -----------------------------------------------
        opp_roll = 0
        if self.opp_skip:
            self.opp_skip = False
            info["opponent_waited"] = True
        else:
            opp_roll   = self.dice.roll(opponent_action)
            raw        = self.opp_pos + opp_roll
            resolved   = resolve_landing(raw)
            self.opp_pos  = resolved.final_pos
            self.opp_skip = resolved.skip_turn

            if resolved.effect == SquareEffect.WIN:
                self.done = True
                return StepResult(
                    agent_pos=self.agent_pos, opponent_pos=self.opp_pos,
                    agent_skip=self.agent_skip, opponent_skip=self.opp_skip,
                    agent_dice_action=agent_action, agent_roll=agent_roll,
                    opponent_roll=opp_roll, outcome=Outcome.LOSS, done=True, info=info,
                )

            if resolved.effect == SquareEffect.MUTUAL_LOSS:
                self.done = True
                return StepResult(
                    agent_pos=self.agent_pos, opponent_pos=self.opp_pos,
                    agent_skip=self.agent_skip, opponent_skip=self.opp_skip,
                    agent_dice_action=agent_action, agent_roll=agent_roll,
                    opponent_roll=opp_roll, outcome=Outcome.MUTUAL_LOSS, done=True, info=info,
                )

        return StepResult(
            agent_pos=self.agent_pos, opponent_pos=self.opp_pos,
            agent_skip=self.agent_skip, opponent_skip=self.opp_skip,
            agent_dice_action=agent_action, agent_roll=agent_roll,
            opponent_roll=opp_roll, outcome=Outcome.ONGOING, done=False, info=info,
        )

    def state_as_index(self) -> tuple[int, int, int, int]:
        """Return state as indices suitable for Q-table lookup."""
        return (
            self.agent_pos - 1,      # 0-indexed
            self.opp_pos   - 1,      # 0-indexed
            int(self.agent_skip),
            int(self.opp_skip),
        )

    # -- Internal -----------------------------------------------------------

    def _reset_state(self):
        self.agent_pos  = BOARD_MIN
        self.opp_pos    = BOARD_MIN
        self.agent_skip = False
        self.opp_skip   = False
        self.done       = False

    def _state(self) -> tuple[int, int, bool, bool]:
        return (self.agent_pos, self.opp_pos, self.agent_skip, self.opp_skip)
