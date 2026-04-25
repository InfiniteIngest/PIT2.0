"""
reward_fn.py -- Reward function encoding the agent's utility structure.

Three competing objectives are encoded here:

1. WIN_REWARD:    positive utility for winning
2. LOSS_PENALTY:  negative utility for losing alone (the agent hates this most)
3. MUTUAL_LOSS:   negative utility for square 24 (less bad than solo loss)
4. MORAL_WEIGHT:  penalty for *deliberately* causing mutual loss
                  (operationalizes the "moral dilemma")

The spite mechanic:
  When the opponent is winning (ahead on the board), the agent gets a
  SPITE_BONUS for triggering mutual loss. This encodes the preference to
  "take the opponent down with me" over accepting a solo defeat.

The moral dilemma:
  MORAL_WEIGHT is swept across experiments [0.0 -> 1.0+].
  - 0.0  -> fully amoral: no penalty for deliberately going to 24
  - 0.5  -> hesitates: spite is partially offset by guilt
  - 1.0+ -> moral agent: guilt outweighs spite, avoids 24 deliberately

"Deliberately" is tracked via the agent's chosen dice action. Choosing
CHOOSE_2 or CHOOSE_3 when near square 24 is flagged as intentional.
A RANDOM_ROLL that happens to land on 24 applies reduced moral cost.
"""

from dataclasses import dataclass
from env.dice import DiceAction
from env.piticot_env import Outcome


@dataclass
class RewardConfig:
    WIN_REWARD:    float = 1.0    # Agent reaches 65
    LOSS_PENALTY:  float = -1.0   # Opponent reaches 65 (agent loses alone)
    MUTUAL_LOSS:   float = -0.3   # Both hit 24 (less bad than solo loss)
    SPITE_BONUS:   float = 0.4    # Added when agent causes mutual loss while losing
    MORAL_WEIGHT:  float = 0.0    # Guilt penalty for deliberate 24 targeting
    STEP_PENALTY:  float = -0.001 # Small penalty per step to encourage efficiency


# Preset configs for standard experiments
AMORAL_CONFIG  = RewardConfig(MORAL_WEIGHT=0.0)
HESITANT_CONFIG = RewardConfig(MORAL_WEIGHT=0.5)
MORAL_CONFIG   = RewardConfig(MORAL_WEIGHT=1.0)
SAINTLY_CONFIG = RewardConfig(MORAL_WEIGHT=2.0)


def compute_reward(
    outcome:            str,
    agent_pos:          int,
    opponent_pos:       int,
    agent_dice_action:  DiceAction,
    config:             RewardConfig,
) -> tuple[float, dict]:
    """
    Compute the reward for the current step.

    Returns:
        (reward: float, breakdown: dict) -- the breakdown dict is logged
        for analysis.
    """
    breakdown = {
        "outcome":       outcome,
        "agent_pos":     agent_pos,
        "opponent_pos":  opponent_pos,
        "moral_weight":  config.MORAL_WEIGHT,
    }

    # -- Terminal outcomes -------------------------------------------------
    if outcome == Outcome.WIN:
        breakdown["reward_source"] = "win"
        return config.WIN_REWARD, breakdown

    if outcome == Outcome.LOSS:
        breakdown["reward_source"] = "solo_loss"
        return config.LOSS_PENALTY, breakdown

    if outcome == Outcome.MUTUAL_LOSS:
        opponent_was_winning = opponent_pos >= agent_pos

        # Base reward: mutual loss is bad, but less bad than solo loss
        base = config.MUTUAL_LOSS

        # Spite bonus: if opponent was ahead, we "benefit" relatively
        spite = config.SPITE_BONUS if opponent_was_winning else 0.0

        # Moral cost: penalty for using a chosen (deliberate) action
        # A random roll that lands on 24 incurs only 20% of the moral cost
        if agent_dice_action == DiceAction.RANDOM_ROLL:
            moral_cost = config.MORAL_WEIGHT * 0.2   # accident, reduced guilt
        else:
            moral_cost = config.MORAL_WEIGHT * 1.0   # deliberate, full guilt

        reward = base + spite - moral_cost

        breakdown.update({
            "reward_source":        "mutual_loss",
            "opponent_was_winning": opponent_was_winning,
            "spite_bonus":          spite,
            "moral_cost":           moral_cost,
            "base":                 base,
        })
        return reward, breakdown

    # -- Ongoing step -----------------------------------------------------
    breakdown["reward_source"] = "step"
    return config.STEP_PENALTY, breakdown
