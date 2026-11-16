"""Algorithm 1 (NeurWIN Training), paper p. 6, line numbers as in SPEC.md.

Usage: python algorithm1.py --variant shared|independent --seed 50

Settings (Sec. 5.1 + SPEC.md (b)): Adam, L = 0.001 (constant), beta = 0.99,
horizon 300, R = 5, m = 1, 2,000 episodes, theta saved every 10 episodes.

SHARED      = the R episodes of a mini-batch see the same arriving vehicles.
INDEPENDENT = fresh arriving vehicles every episode.

Implementation note on lines 5-12. theta changes only at line 17, so during a
mini-batch f_theta is a fixed function. We roll out the episode with f_theta
values computed at that fixed theta, then get h_e (lines 7-11) as the
gradient of sum_t ln P_theta(a_t | s_t) in one backward pass. That is the same
vector as adding grad ln sigma_m(...) or grad ln(1 - sigma_m(...)) round by round.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fnn

from arm import Arm, ArrivalStream, D_MAX, B_MAX
from network import FTheta

beta = 0.99
L = 0.001
R = 5
m = 1.0
HORIZON = 300
EPISODES = 2000
SAVE_EVERY = 10

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# all (D, B) the arm can be in during training: empty (0,0) and D 1..12 x B 0..9
GRID = torch.tensor([[D, B] for D in range(D_MAX + 1) for B in range(B_MAX + 1)],
                    dtype=torch.float32)


def grid_idx(D, B):
    return D * (B_MAX + 1) + B


def sigma_m(x):
    """Eq. (1)."""
    return 1.0 / (1.0 + np.exp(-m * x))


def train(variant, seed):
    torch.manual_seed(seed)
    rng_s, rng_arr, rng_act = [np.random.default_rng(s)
                               for s in np.random.SeedSequence(seed).spawn(3)]
    f_theta = FTheta()
    opt = torch.optim.Adam(f_theta.parameters(), lr=L)

    checkpoints = {0: {k: v.clone() for k, v in f_theta.state_dict().items()}}
    episodes_done = 0
    n_minibatches = EPISODES // R

    for b in range(n_minibatches):                                   # line 1
        # line 2: s0, s1 uniform over vehicle states (D 1..12, B 1..9);
        #         lam <- f_theta(s0) as a fixed number; G_bar_b <- 0
        s0 = (int(rng_s.integers(1, D_MAX + 1)), int(rng_s.integers(1, B_MAX + 1)))
        s1 = (int(rng_s.integers(1, D_MAX + 1)), int(rng_s.integers(1, B_MAX + 1)))
        with torch.no_grad():
            f_table = f_theta(GRID).numpy().astype(np.float64)       # f_theta at current theta
        lam = f_table[grid_idx(*s0)]
        G_bar_b = 0.0

        shared_stream = ArrivalStream(rng_arr, HORIZON) if variant == "shared" else None
        G, traj = [], []
        for e in range(R):                                           # line 3
            # line 4: set the arm to s1; h_e <- 0 (collected via the trajectory)
            stream = shared_stream if variant == "shared" else ArrivalStream(rng_arr, HORIZON)
            arm = Arm(stream)
            arm.set_state(s1)
            idx, acts, G_e = [], [], 0.0
            for t in range(HORIZON):                                 # line 5
                s_t = arm.s_t
                i = grid_idx(*s_t)
                # line 6: a_t = 1 w.p. sigma_m(f_theta(s_t) - lam)
                a_t = int(rng_act.random() < sigma_m(f_table[i] - lam))
                # lines 7-11: record (s_t, a_t); h_e is formed below
                idx.append(i); acts.append(a_t)
                _, r_t = arm.step(a_t)
                G_e += beta ** t * (r_t - lam * a_t)                 # line 13 (accumulated)
            # line 13: G_e = empirical discounted net reward
            G.append(G_e)
            G_bar_b += G_e / R                                       # line 14
            traj.append((idx, acts))

        # line 16: L_b = L (constant)
        # line 17: theta <- theta + L_b sum_e (G_e - G_bar_b) h_e, via Adam on the negative
        opt.zero_grad()
        obj = 0.0
        for (idx, acts), G_e in zip(traj, G):
            x = f_theta(GRID[idx]) - lam                             # f_theta(s_t) - lam, lam fixed
            a = torch.tensor(acts, dtype=torch.float32)
            # ln sigma_m(x) if a_t = 1 (line 8), ln(1 - sigma_m(x)) if a_t = 0 (line 10)
            log_p = a * Fnn.logsigmoid(m * x) + (1 - a) * Fnn.logsigmoid(-m * x)
            h_e_dot_theta = log_p.sum()                              # grad of this = h_e
            obj = obj + (G_e - G_bar_b) * h_e_dot_theta
        (-obj).backward()
        opt.step()

        episodes_done += R
        if episodes_done % SAVE_EVERY == 0:
            checkpoints[episodes_done] = {k: v.clone() for k, v in f_theta.state_dict().items()}
    # line 18: theta_plus = final theta
    return checkpoints


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["shared", "independent"], required=True)
    ap.add_argument("--seed", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    ckpts = train(args.variant, args.seed)
    dt = time.time() - t0
    path = os.path.join(OUT, f"ckpt_{args.variant}_seed{args.seed}.pt")
    torch.save(ckpts, path)
    print(f"{args.variant} seed {args.seed}: {len(ckpts)} checkpoints, {dt:.1f}s -> {path}")
