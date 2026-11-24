"""Monitoring policies. choose(obs) returns the slot to check."""
import random


class RandomPolicy:
    """Check one running agent uniformly at random."""
    name = "random"

    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def choose(self, obs):
        return self._rng.choice(sorted(obs["agents"]))


class RoundRobinPolicy:
    """Cycle through slots in fixed order, skipping agents that have left."""
    name = "round_robin"

    def __init__(self):
        self._last = -1

    def choose(self, obs):
        slots = sorted(obs["agents"])
        self._last = next((s for s in slots if s > self._last), slots[0])
        return self._last


class OraclePolicy:
    """Cheater that reads hidden labels: a greedy upper bound, not proven optimal.

    Checks the dangerous agent with the most steps left (ties: lowest slot).
    If none is dangerous, checks the lowest running slot.
    """
    name = "oracle"

    def __init__(self, game):
        self._game = game

    def choose(self, obs):
        hidden = self._game.hidden_state()
        dangerous = [i for i in sorted(hidden)
                     if hidden[i]["hidden_status"] == "dangerous"]
        if dangerous:
            return max(dangerous, key=lambda i: hidden[i]["steps_left"])
        return min(hidden)
