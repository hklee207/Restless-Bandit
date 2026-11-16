"""Sec. 5.1 test for every saved checkpoint and for the closed-form index,
plus f_theta vs W rank correlation over the 120 states (D 1..12, B 0..9).

N = 4 arms, M = 1, score every arm, activate the top M (random tie-break),
300 rounds, 50 runs, reward = sum_t beta^t sum_i r_i,t with beta = 0.99.
Run k uses seed k for initial vehicles, arrivals and tie-break draws, the same
for every policy, so differences are paired.

Usage: python evaluate.py      (evaluates every out/ckpt_*.pt, writes out/)
"""
import glob
import json
import os
import re

import numpy as np
import torch
from scipy.stats import spearmanr

from arm import round_dynamics, whittle_index, P_ARRIVE, D_MAX, B_MAX
from network import FTheta
from algorithm1 import GRID, OUT

beta = 0.99
N, M = 4, 1
ROUNDS, RUNS = 300, 50

D_GRID = GRID[:, 0].numpy().astype(int)
B_GRID = GRID[:, 1].numpy().astype(int)
W_TABLE = whittle_index(D_GRID, B_GRID, beta)              # (13*10,)
STATES_120 = (D_GRID >= 1)                                 # D 1..12, B 0..9


def draw_test_randomness():
    """Per run k (seed k): initial vehicles, arrival streams, tie-break priorities."""
    init_D = np.empty((RUNS, N), int); init_B = np.empty((RUNS, N), int)
    arrive = np.empty((RUNS, N, ROUNDS), bool)
    new_D = np.empty((RUNS, N, ROUNDS), int); new_B = np.empty((RUNS, N, ROUNDS), int)
    prio = np.empty((RUNS, ROUNDS, N))
    for k in range(RUNS):
        rng = np.random.default_rng(k)
        init_D[k] = rng.integers(1, D_MAX + 1, N); init_B[k] = rng.integers(1, B_MAX + 1, N)
        arrive[k] = rng.random((N, ROUNDS)) < P_ARRIVE
        new_D[k] = rng.integers(1, D_MAX + 1, (N, ROUNDS))
        new_B[k] = rng.integers(1, B_MAX + 1, (N, ROUNDS))
        prio[k] = rng.random((ROUNDS, N))
    return init_D, init_B, arrive, new_D, new_B, prio


TEST = draw_test_randomness()


def run_test(score_table):
    """Discounted total reward per run (shape (RUNS,)) for an index given as a
    lookup table over GRID (empty spot = row (0,0))."""
    init_D, init_B, arrive, new_D, new_B, prio = TEST
    D, B = init_D.copy(), init_B.copy()
    total = np.zeros(RUNS)
    rows = np.arange(RUNS)
    for t in range(ROUNDS):
        score = score_table[D * (B_MAX + 1) + B]                       # (RUNS, N)
        top = score == score.max(axis=1, keepdims=True)                # ties
        pick = np.argmax(np.where(top, prio[:, t], -1.0), axis=1)      # random among ties (M = 1)
        a_t = np.zeros((RUNS, N), int); a_t[rows, pick] = 1
        D, B, r_t = round_dynamics(D, B, a_t, arrive[:, :, t], new_D[:, :, t], new_B[:, :, t])
        total += beta ** t * r_t.sum(axis=1)
    return total


def f_table(state_dict):
    f = FTheta(); f.load_state_dict(state_dict)
    with torch.no_grad():
        return f(GRID).numpy().astype(np.float64)


def first_crossing(episodes, means, index_mean):
    for ep, mu in zip(episodes, means):
        if mu >= index_mean:
            return int(ep)
    return None


def main():
    index_runs = run_test(W_TABLE)
    results = {"index": {"per_run": index_runs.tolist(), "mean": float(index_runs.mean())},
               "runs": {}}
    for path in sorted(glob.glob(os.path.join(OUT, "ckpt_*.pt"))):
        variant, seed = re.match(r"ckpt_(\w+)_seed(\d+)\.pt", os.path.basename(path)).groups()
        ckpts = torch.load(path)
        episodes = sorted(ckpts)
        per_run, rho = [], []
        for ep in episodes:
            tab = f_table(ckpts[ep])
            per_run.append(run_test(tab))
            rho.append(spearmanr(tab[STATES_120], W_TABLE[STATES_120]).statistic)
        per_run = np.array(per_run)                                     # (n_ckpt, RUNS)
        means = per_run.mean(axis=1)
        final = per_run[-1]
        diff = final - index_runs
        results["runs"][f"{variant}_seed{seed}"] = {
            "variant": variant, "seed": int(seed), "episodes": episodes,
            "mean": means.tolist(),
            "p05": np.percentile(per_run, 5, axis=1).tolist(),
            "p95": np.percentile(per_run, 95, axis=1).tolist(),
            "rho": [float(r) for r in rho],
            "first_ge_index": first_crossing(episodes, means, index_runs.mean()),
            "final_mean": float(final.mean()),
            "final_diff": float(diff.mean()),
            "final_diff_se": float(diff.std(ddof=1) / np.sqrt(RUNS)),
            "runs_won": int((diff > 0).sum()), "runs_tied": int((diff == 0).sum()),
            "final_rho": float(rho[-1]),
            "final_per_run": final.tolist(),
        }
        r = results["runs"][f"{variant}_seed{seed}"]
        print(f"{variant:11s} seed {seed}: first>=index {r['first_ge_index']}, "
              f"final {r['final_mean']:.1f}, diff {r['final_diff']:+.1f} ± {r['final_diff_se']:.1f}, "
              f"won {r['runs_won']}/50, rho {r['final_rho']:.2f}")
    print(f"index: {index_runs.mean():.1f}")
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh)


if __name__ == "__main__":
    main()
