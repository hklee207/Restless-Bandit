"""
Fast re-computation of the authors' NeurWIN test (same env, same seeds, same tie rule), used to
(1) get the full training curve quickly and (2) run the N-scaling check.
Each checkpoint's network is evaluated once on every (T,B) state and turned into a lookup table,
which gives the same choices as calling the network inside the loop.

Usage:
  python curve.py curve  <repo> <out.csv>                 # N=4, M=1, all checkpoints 0..2000 step 10
  python curve.py scale  <repo> <out.csv>                 # N in {4,8,20,100}, M=N/4, final model vs index
"""
import sys, os, random
import numpy as np, torch
from multiprocessing import Pool

MODE, REPO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, REPO)
from neurwin import fcnn
from envs.deadlineSchedulingEnv import deadlineSchedulingEnv
TL, BETA, C, RUNS = 300, 0.99, 0.5, 50
MDIR = os.path.join(REPO, "trainResults/neurwin/deadline_env/")

def cf(T, B):
    if B == 0: return 0.0
    if 1 <= B <= T - 1: return 1 - C
    return BETA**(T-1)*0.2*((B-T+1)**2 - (B-T)**2) + 1 - C

def table_for(ep):
    net = fcnn(2)
    net.load_state_dict(torch.load(MDIR + f"seed_50_lr_0.001_batchSize_5_trainedNumEpisodes_{ep}/trained_model.pt"))
    net.eval()
    g = np.array([[T, B] for T in range(13) for B in range(10)], dtype=np.float32)
    with torch.no_grad():
        # same float32 arithmetic as the authors' per-state forward()
        v = [float(net.forward(s)[0]) for s in g]
    return {(int(T), int(B)): x for (T, B), x in zip(g, v)}

CF = {(T, B): cf(T, B) for T in range(13) for B in range(10)}

def rollout(score, N, M):
    random.seed(30); np.random.seed(30); torch.manual_seed(30)
    out = []
    for _ in range(RUNS):
        seeds = np.random.randint(0, 10000, size=N)
        envs = [deadlineSchedulingEnv(seed=int(s), numEpisodes=1, episodeLimit=TL, maxDeadline=12, maxLoad=9,
                newJobProb=0.7, processingCost=C, train=False, batchSize=2000, noiseVar=0) for s in seeds]
        st = [(int(a), int(b)) for a, b in (e.reset() for e in envs)]
        tot = 0.0
        for t in range(TL):
            # authors' getSelection(): repeated max, ties -> lowest arm id
            chosen = set(sorted(range(N), key=lambda k: (-score[st[k]], k))[:M])
            r = 0.0
            for k, e in enumerate(envs):
                s, rew, _, _ = e.step(1 if k in chosen else 0)
                st[k] = (int(s[0]), int(s[1])); r += rew
            tot += BETA**t * r
        out.append(tot)
    return out

def curve_job(ep):
    return ep, rollout(table_for(ep), 4, 1)

def scale_job(nm):
    N, M = nm
    return N, M, rollout(table_for(2000), N, M), rollout(CF, N, M)

if __name__ == "__main__":
    import pandas as pd
    with Pool(os.cpu_count()) as p:
        if MODE == "curve":
            res = p.map(curve_job, list(range(0, 2001, 10)))
            rows = [dict(episode=ep, run=i, reward=r) for ep, rs in res for i, r in enumerate(rs)]
            pd.DataFrame(rows).to_csv(OUT, index=False)
            idx = rollout(CF, 4, 1)
            pd.DataFrame(dict(run=range(RUNS), reward=idx)).to_csv(OUT.replace(".csv", "_index.csv"), index=False)
        else:
            res = p.map(scale_job, [(4, 1), (8, 2), (20, 5), (100, 25)])
            rows = []
            for N, M, nw, ix in res:
                d = np.array(nw) - np.array(ix)
                rows.append(dict(N=N, M=M, neurwin=np.mean(nw), index=np.mean(ix), gap=d.mean(),
                                 gap_per_arm=d.mean() / N, gap_pct=100 * d.mean() / abs(np.mean(ix)),
                                 paired_se=d.std(ddof=1) / np.sqrt(RUNS), neurwin_better_runs=int((d > 0).sum())))
            df = pd.DataFrame(rows); df.to_csv(OUT, index=False); print(df.to_string(index=False))
