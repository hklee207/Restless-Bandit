"""
Gap check: why does NeurWIN end above the closed-form deadline Whittle index at N=4, M=1?

Hypothesis: tie-breaking. The closed-form index gives the SAME value (1 - c = 0.5) to every
state with 1 <= B <= T-1, so with 4 arms there are often ties. The authors' test script breaks
ties by arm number (lowest key wins), which is arbitrary. NeurWIN outputs a smooth function of
(T, B), so it effectively breaks those ties by urgency.

This script reuses the authors' environment and the authors' exact index formula and seeds,
so variant A must reproduce their test script's closed-form numbers exactly.

Usage: python gap_check.py <path to NeurWIN repo> <path to trained_model.pt> <out dir>
"""
import sys, os, random, json
import numpy as np
import torch

REPO, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, REPO)
from neurwin import fcnn
from envs.deadlineSchedulingEnv import deadlineSchedulingEnv

ARMS, SCHEDULE, TIMELIMIT, BETA, C, RUNS, SEED = 4, 1, 300, 0.99, 0.5, 50, 30


def closed_form(T, B):
    """Exact copy of selectDeadlineIndex() in the authors' test script."""
    if B == 0:
        return 0.0
    if 1 <= B <= T - 1:
        return 1 - C
    first = (BETA ** (T - 1)) * (0.2 * (B - T + 1) ** 2)
    second = (BETA ** (T - 1)) * (0.2 * (B - T) ** 2)
    return first - second + 1 - C


net = fcnn(stateSize=2)
net.load_state_dict(torch.load(MODEL))
net.eval()


def neurwin(T, B):
    with torch.no_grad():
        return float(net.forward(np.array([T, B], dtype=np.float32))[0])


def make_rule(name, rng):
    """Return a function states -> list of chosen arms."""
    def pick(scores):
        # max with ties broken by lowest arm id, like the authors' getSelection()
        order = sorted(range(ARMS), key=lambda k: (-scores[k][0], scores[k][1], k))
        return order[:SCHEDULE]

    def rule(states):
        sc = {}
        for k, (T, B) in states.items():
            if name == "A_closedform_tie_by_arm_id":
                sc[k] = (closed_form(T, B), 0)
            elif name == "B_closedform_tie_by_least_slack":
                # slack = T - B (time to spare); smaller slack = more urgent
                sc[k] = (closed_form(T, B), T - B if B > 0 else 99)
            elif name == "C_closedform_tie_random":
                sc[k] = (closed_form(T, B), rng.random())
            elif name == "D_neurwin_2000":
                sc[k] = (neurwin(T, B), 0)
        return pick(sc)
    return rule


def run_policy(name):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rng = random.Random(123)
    rule = make_rule(name, rng)
    totals = []
    for _ in range(RUNS):
        envSeeds = np.random.randint(0, 10000, size=ARMS)
        envs = {i: deadlineSchedulingEnv(seed=int(envSeeds[i]), numEpisodes=1, episodeLimit=TIMELIMIT,
                                         maxDeadline=12, maxLoad=9, newJobProb=0.7, processingCost=C,
                                         train=False, batchSize=2000, noiseVar=0) for i in range(ARMS)}
        states = {k: tuple(e.reset()) for k, e in envs.items()}
        total = 0.0
        for t in range(TIMELIMIT):
            chosen = rule(states)
            r = 0.0
            for k, e in envs.items():
                s, rew, _, _ = e.step(1 if k in chosen else 0)
                states[k] = tuple(s)
                r += rew
            total += (BETA ** t) * r
        totals.append(total)
    return np.array(totals)


os.makedirs(OUT, exist_ok=True)
names = ["A_closedform_tie_by_arm_id", "B_closedform_tie_by_least_slack",
         "C_closedform_tie_random", "D_neurwin_2000"]
res = {n: run_policy(n) for n in names}
base = res["A_closedform_tie_by_arm_id"]
summary = {}
for n in names:
    d = res[n] - base
    summary[n] = dict(mean=float(res[n].mean()),
                      diff_vs_A=float(d.mean()),
                      paired_se=float(d.std(ddof=1) / np.sqrt(RUNS)) if n != names[0] else 0.0,
                      runs_better_than_A=int((d > 1e-9).sum()))
    print(f"{n:35s} mean {res[n].mean():8.2f}  diff vs A {d.mean():+6.2f}  "
          f"(paired SE {summary[n]['paired_se']:.2f}, better in {summary[n]['runs_better_than_A']}/50)")

# NeurWIN-vs-index gap left after better tie-breaking
d = res["D_neurwin_2000"] - res["B_closedform_tie_by_least_slack"]
summary["D_minus_B"] = dict(mean=float(d.mean()), paired_se=float(d.std(ddof=1) / np.sqrt(RUNS)))
print(f"NeurWIN minus (index + least-slack tie-break): {d.mean():+.2f} (paired SE {summary['D_minus_B']['paired_se']:.2f})")

# How often is the top index tied at a decision? (closed form, variant A rollout)
# and the learned vs closed-form index over all valid states
rows = []
for T in range(1, 13):
    for B in range(0, 10):
        rows.append((T, B, closed_form(T, B), neurwin(T, B)))
rows = np.array(rows)
from scipy.stats import pearsonr, spearmanr
summary["index_corr"] = dict(pearson=float(pearsonr(rows[:, 2], rows[:, 3])[0]),
                             spearman=float(spearmanr(rows[:, 2], rows[:, 3])[0]))
tie = rows[(rows[:, 1] >= 1) & (rows[:, 1] <= rows[:, 0] - 1)]
summary["tie_region"] = dict(n_states=int(len(tie)),
                             spearman_neurwin_vs_minus_slack=float(spearmanr(tie[:, 3], -(tie[:, 0] - tie[:, 1]))[0]),
                             spearman_neurwin_vs_B=float(spearmanr(tie[:, 3], tie[:, 1])[0]),
                             spearman_neurwin_vs_minus_T=float(spearmanr(tie[:, 3], -tie[:, 0])[0]))
print(json.dumps(summary["index_corr"]), json.dumps(summary["tie_region"]))
np.savetxt(os.path.join(OUT, "index_table.csv"), rows, delimiter=",", header="T,B,closed_form,neurwin", comments="")
json.dump(summary, open(os.path.join(OUT, "gap_summary.json"), "w"), indent=2)
