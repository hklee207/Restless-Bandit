"""One deadline-scheduling arm (one charging spot), Sec. 3.3 / Sec. 5.2.

State s_t = (D, B): D = rounds left including this one, B = units still
needed. The empty spot is (0, 0). See SPEC.md, "Round dynamics".

Assumed constants (SPEC.md (b)): c = 0.5, F(B) = 0.2 B^2, arrival prob 0.7,
new vehicle D ~ U{1..12}, B ~ U{1..9}.
"""
import numpy as np

c = 0.5
P_ARRIVE = 0.7
D_MAX, B_MAX = 12, 9


def F(B):
    """Penalty for B units left unfulfilled when the vehicle leaves."""
    return 0.2 * B ** 2


def round_dynamics(D, B, a_t, arrive, new_D, new_B):
    """One round for one arm or for arrays of arms (works on ints or numpy arrays).

    Vehicle present (D >= 1):
      1. a_t = 1 and B > 0: deliver 1 unit, earn 1 - c.
      2. D <- D - 1.
      3. D == 0: vehicle leaves at the end of THIS round; pay F(B) in THIS
         round's reward (matches the beta^(D-1) factor in the index).
    Spot empty at the end of the round: with the pre-drawn coin `arrive`
    (prob 0.7) the next state is the new vehicle (new_D, new_B), else (0, 0).
    Returns (D_next, B_next, r_t).
    """
    D = np.asarray(D); B = np.asarray(B); a_t = np.asarray(a_t)
    present = D >= 1
    charge = present & (a_t == 1) & (B > 0)
    r_t = np.where(charge, 1.0 - c, 0.0)
    B = B - charge.astype(B.dtype)
    D = np.where(present, D - 1, D)
    leaves = present & (D == 0)
    r_t = r_t - np.where(leaves, F(B), 0.0)
    B = np.where(D == 0, 0, B)                 # spot now (or already) empty
    empty = D == 0
    take = empty & np.asarray(arrive)
    D_next = np.where(take, new_D, D)
    B_next = np.where(take, new_B, B)
    return D_next, B_next, r_t


class ArrivalStream:
    """Pre-drawn arrival randomness for one arm: per round, the 0.7 coin and
    the (D, B) of the vehicle that would arrive. SHARED reuses one stream for
    the R episodes of a mini-batch; INDEPENDENT draws a fresh one per episode."""

    def __init__(self, rng, horizon):
        self.arrive = rng.random(horizon) < P_ARRIVE
        self.new_D = rng.integers(1, D_MAX + 1, size=horizon)
        self.new_B = rng.integers(1, B_MAX + 1, size=horizon)


class Arm:
    """The Sec. 3.3 black box: set any state, then step(a_t) -> (s_{t+1}, r_t)."""

    def __init__(self, arrivals: ArrivalStream):
        self.arrivals = arrivals
        self.D, self.B, self.t = 0, 0, 0

    def set_state(self, s):
        self.D, self.B = int(s[0]), int(s[1])
        self.t = 0

    @property
    def s_t(self):
        return (self.D, self.B)

    def step(self, a_t):
        """Fast scalar version of round_dynamics (same rules)."""
        st = self.arrivals
        D, B, r_t = self.D, self.B, 0.0
        if D >= 1:
            if a_t == 1 and B > 0:
                B -= 1
                r_t += 1.0 - c
            D -= 1
            if D == 0:
                r_t -= F(B)
                B = 0
        if D == 0 and st.arrive[self.t]:
            D, B = int(st.new_D[self.t]), int(st.new_B[self.t])
        self.D, self.B = D, B
        self.t += 1
        return (D, B), r_t


def whittle_index(D, B, beta=0.99):
    """Closed-form deadline Whittle index from ref [34] (SPEC.md (b) item 8).
    Empty spot (0, 0) and B = 0 give 0."""
    D = np.asarray(D, dtype=float); B = np.asarray(B, dtype=float)
    W = np.where(B <= D - 1, 1.0 - c, 0.0)
    big = beta ** (D - 1) * (F(B - D + 1) - F(B - D)) + 1.0 - c
    W = np.where((B >= D) & (B >= 1), big, W)
    return np.where(B == 0, 0.0, W)
