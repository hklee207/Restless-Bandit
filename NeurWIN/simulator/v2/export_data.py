"""Export everything the v2 explainer page needs, from the authors' code and my trained checkpoints.
Usage: python export_data.py <NeurWIN repo> <out.json>
"""
import sys, json, random
import numpy as np, torch, pandas as pd

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
from neurwin import fcnn
from envs.deadlineSchedulingEnv import deadlineSchedulingEnv

MDIR = REPO + "/trainResults/neurwin/deadline_env/"
BETA, C = 0.99, 0.5


def cf(T, B):  # exact copy of selectDeadlineIndex()
    if B == 0:
        return 0.0
    if 1 <= B <= T - 1:
        return 1 - C
    return (BETA ** (T - 1)) * (0.2 * (B - T + 1) ** 2) - (BETA ** (T - 1)) * (0.2 * (B - T) ** 2) + 1 - C


def table(ep):
    net = fcnn(2)
    net.load_state_dict(torch.load(MDIR + f"seed_50_lr_0.001_batchSize_5_trainedNumEpisodes_{ep}/trained_model.pt"))
    net.eval()
    with torch.no_grad():
        return [[float(net.forward(np.array([T, B], dtype=np.float32))[0]) for B in range(10)] for T in range(13)]


ckpts = list(range(0, 2001, 10))
tables = {ep: table(ep) for ep in ckpts}
closed = [[cf(T, B) for B in range(10)] for T in range(13)]

c = pd.read_csv(REPO + "/replication/out/curve_N4.csv")
g = c.groupby("episode").reward
curve = [dict(ep=int(e), mean=round(float(g.mean()[e]), 2), p5=round(float(g.quantile(.05)[e]), 2),
              p95=round(float(g.quantile(.95)[e]), 2)) for e in ckpts]
idx_mean = float(pd.read_csv(REPO + "/replication/out/curve_N4_index.csv").reward.mean())
scale = pd.read_csv(REPO + "/replication/out/scale.csv").to_dict(orient="records")

# ---- trace: 3 runs, N=4, M=1, authors' env + test-script seeding, both policies
def run_trace(policy_tab, seeds):
    envs = [deadlineSchedulingEnv(seed=int(s), numEpisodes=1, episodeLimit=300, maxDeadline=12, maxLoad=9,
                                  newJobProb=0.7, processingCost=C, train=False, batchSize=2000, noiseVar=0)
            for s in seeds]
    st = [tuple(int(v) for v in e.reset()) for e in envs]
    draws = [dict(job=[int(x) for x in e.jobList], T=[int(x) for x in e.timeUntilDeadline],
                  B=[int(x) for x in e.load]) for e in envs]
    rewards, picks = [], []
    for t in range(300):
        a = max(range(4), key=lambda k: (policy_tab[st[k][0]][st[k][1]], -k))
        r = 0.0
        for k, e in enumerate(envs):
            s, rew, _, _ = e.step(1 if k == a else 0)
            st[k] = (int(s[0]), int(s[1]))
            r += rew
        rewards.append(r); picks.append(a)
    return draws, rewards, picks

random.seed(30); np.random.seed(30); torch.manual_seed(30)
seed_list = [np.random.randint(0, 10000, size=4) for _ in range(3)]
trace = []
for seeds in seed_list:
    d, r_cf, p_cf = run_trace(closed, seeds)
    _, r_nw, p_nw = run_trace(tables[2000], seeds)
    trace.append(dict(draws=d, closed=dict(rewards=r_cf, picks=p_cf), neurwin=dict(rewards=r_nw, picks=p_nw)))
    print("run total cf %.4f nw %.4f" % (sum(BETA**t*x for t, x in enumerate(r_cf)), sum(BETA**t*x for t, x in enumerate(r_nw))))

data = dict(ckpts=ckpts, tables={str(k): [[round(v, 6) for v in row] for row in t] for k, t in tables.items()},
            closed=closed, curve=curve, indexMean=round(idx_mean, 4), scale=scale)
json.dump(data, open(OUT, "w"), separators=(",", ":"))
json.dump(trace, open(OUT.replace(".json", "_trace.json"), "w"), separators=(",", ":"))
print("ok", len(json.dumps(data)) // 1024, "KB")
