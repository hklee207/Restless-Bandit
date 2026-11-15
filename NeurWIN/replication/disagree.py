"""
Where do NeurWIN and the closed-form index pick different cars?
Roll out the NeurWIN policy (same seeds as the authors' test) and, at every step, also ask
which car the closed-form index would pick. Tally the (NeurWIN pick, index pick) state pairs.
Also test variant E: closed-form index with ties broken by larger remaining charge B.
Usage: python disagree.py <repo> <model.pt>
"""
import sys, random
from collections import Counter
import numpy as np, torch
REPO, MODEL = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
from neurwin import fcnn
from envs.deadlineSchedulingEnv import deadlineSchedulingEnv
ARMS, TL, BETA, C, RUNS = 4, 300, 0.99, 0.5, 50

def cf(T, B):
    if B == 0: return 0.0
    if 1 <= B <= T - 1: return 1 - C
    return BETA**(T-1)*0.2*((B-T+1)**2 - (B-T)**2) + 1 - C

net = fcnn(2); net.load_state_dict(torch.load(MODEL)); net.eval()
def nw(T, B):
    with torch.no_grad(): return float(net.forward(np.array([T, B], dtype=np.float32))[0])

def rollout(policy, log=None):
    random.seed(30); np.random.seed(30); torch.manual_seed(30)
    out = []
    for _ in range(RUNS):
        seeds = np.random.randint(0, 10000, size=ARMS)
        envs = [deadlineSchedulingEnv(seed=int(s), numEpisodes=1, episodeLimit=TL, maxDeadline=12, maxLoad=9,
                newJobProb=0.7, processingCost=C, train=False, batchSize=2000, noiseVar=0) for s in seeds]
        st = [tuple(int(v) for v in e.reset()) for e in envs]
        tot = 0.0
        for t in range(TL):
            a = policy(st)
            if log is not None:
                b = max(range(ARMS), key=lambda k: (cf(*st[k]), -k))
                log["steps"] += 1
                if cf(*st[a]) != cf(*st[b]):          # a real disagreement, not a tie
                    log["pairs"][(st[a], st[b])] += 1
            r = 0.0
            for k, e in enumerate(envs):
                s, rew, _, _ = e.step(1 if k == a else 0); st[k] = tuple(int(v) for v in s); r += rew
            tot += BETA**t * r
        out.append(tot)
    return np.array(out)

log = {"steps": 0, "pairs": Counter()}
d = rollout(lambda st: max(range(ARMS), key=lambda k: (nw(*st[k]), -k)), log)
e = rollout(lambda st: max(range(ARMS), key=lambda k: (cf(*st[k]), st[k][1], -k)))
a = rollout(lambda st: max(range(ARMS), key=lambda k: (cf(*st[k]), -k)))
print(f"A index {a.mean():.2f} | E index+larger-B tiebreak {e.mean():.2f} | D NeurWIN {d.mean():.2f}")
n = sum(log["pairs"].values())
print(f"NeurWIN rollout: {log['steps']} decisions, {n} real disagreements ({100*n/log['steps']:.1f}%)")
cat = Counter()
for (sa, sb), c in log["pairs"].items():
    # sa = (T,B) NeurWIN charged; sb = (T,B) index would have charged
    if sb[1] >= sb[0] and sa[1] < sa[0]:
        cat["index wanted an already-late car (B>=T); NeurWIN charged an on-track car"] += c
    elif sa[1] >= sa[0] and sb[1] >= sb[0]:
        cat["both late cars, different ones"] += c
    elif sa[1] >= sa[0] and sb[1] < sb[0]:
        cat["NeurWIN charged a late car; index wanted an on-track car"] += c
    else:
        cat["other"] += c
for k, v in cat.most_common(): print(f"  {v:5d}  {k}")
print("Top 12 pairs  NeurWIN-charged (T,B) [idx]  vs  index-charged (T,B) [idx]:")
for (sa, sb), c in log["pairs"].most_common(12):
    print(f"  {c:4d}  {sa} [{cf(*sa):.3f} | nw {nw(*sa):.2f}]  vs  {sb} [{cf(*sb):.3f} | nw {nw(*sb):.2f}]")
