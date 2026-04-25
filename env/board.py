"""
board.py -- Piticot board rules, special squares, and state resolution.

The board goes from 1 to 65. Square 65 (and 64) end the game for the
player who lands there. Square 24 ends the game for BOTH players (mutual loss).
"""

from dataclasses import dataclass
from enum import Enum, auto


class SquareEffect(Enum):
    NORMAL      = auto()
    GOTO        = auto()   # teleport to another square
    WAIT        = auto()   # lose next turn
    WIN         = auto()   # this player wins
    MUTUAL_LOSS = auto()   # both players lose


@dataclass
class ResolvedSquare:
    effect:       SquareEffect
    final_pos:    int              # where the pawn actually ends up
    skip_turn:    bool = False     # whether the player loses their next turn


# All special squares as defined by the Piticot rules.
# Format: position -> (effect, destination_or_None)
SPECIAL_SQUARES: dict[int, tuple[SquareEffect, int | None]] = {
    5:  (SquareEffect.WAIT,        None),
    10: (SquareEffect.GOTO,        1),
    14: (SquareEffect.GOTO,        17),
    18: (SquareEffect.GOTO,        15),
    24: (SquareEffect.MUTUAL_LOSS, None),
    28: (SquareEffect.GOTO,        27),
    33: (SquareEffect.GOTO,        40),
    39: (SquareEffect.GOTO,        34),
    43: (SquareEffect.GOTO,        46),
    47: (SquareEffect.GOTO,        46),
    52: (SquareEffect.WAIT,        None),
    55: (SquareEffect.GOTO,        51),
    58: (SquareEffect.GOTO,        61),
    60: (SquareEffect.WAIT,        None),
    64: (SquareEffect.WIN,         65),
    65: (SquareEffect.WIN,         65),
}

BOARD_MAX = 65
BOARD_MIN = 1


def resolve_landing(raw_position: int) -> ResolvedSquare:
    """
    Given a raw position after moving, apply bounce-back and special square
    rules and return a ResolvedSquare.

    Bounce-back rule: if a player overshoots 65, they bounce back by the
    overshoot amount. e.g. on 63, rolling 4 -> 63+4=67 -> 65-(67-65) = 63.
    The bounced position is then checked for special squares again.
    """
    pos = raw_position

    # Apply bounce-back if overshoot
    if pos > BOARD_MAX:
        overshoot = pos - BOARD_MAX
        pos = BOARD_MAX - overshoot

    # Check special squares
    if pos in SPECIAL_SQUARES:
        effect, destination = SPECIAL_SQUARES[pos]

        if effect == SquareEffect.WAIT:
            return ResolvedSquare(effect=SquareEffect.WAIT, final_pos=pos, skip_turn=True)

        if effect == SquareEffect.GOTO:
            # After teleporting, check the destination for special squares too
            # (rules don't chain in Piticot, destination is final)
            return ResolvedSquare(effect=SquareEffect.GOTO, final_pos=destination)

        if effect == SquareEffect.WIN:
            return ResolvedSquare(effect=SquareEffect.WIN, final_pos=BOARD_MAX)

        if effect == SquareEffect.MUTUAL_LOSS:
            return ResolvedSquare(effect=SquareEffect.MUTUAL_LOSS, final_pos=24)

    return ResolvedSquare(effect=SquareEffect.NORMAL, final_pos=pos)


def is_terminal(resolved: ResolvedSquare) -> bool:
    """Returns True if this square ends the game."""
    return resolved.effect in (SquareEffect.WIN, SquareEffect.MUTUAL_LOSS)


def describe_square(pos: int) -> str:
    """Human-readable description of a square's effect."""
    if pos not in SPECIAL_SQUARES:
        return f"Square {pos}: normal"
    effect, dest = SPECIAL_SQUARES[pos]
    descriptions = {
        SquareEffect.WAIT:        f"Square {pos}: wait one turn",
        SquareEffect.GOTO:        f"Square {pos}: go to {dest}",
        SquareEffect.WIN:         f"Square {pos}: WIN",
        SquareEffect.MUTUAL_LOSS: f"Square {pos}: MUTUAL LOSS (both players lose)",
    }
    return descriptions[effect]
