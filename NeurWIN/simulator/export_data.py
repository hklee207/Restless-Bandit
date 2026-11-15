"""
Export everything the browser simulator needs. Nothing here changes the authors' code,
the environment, the index formula or the trained networks: it only reads them.

Writes (next to this file):
  data.js     window.SIM_DATA = {...}  index tables + chart data, loaded by index.html
  trace.json  3 reference runs from the authors' env class, used by test_sim.mjs

Usage (from anywhere):
  ../../.venv/bin/python export_data.py
"""
import os, sys, json, random, operator
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from neurwin import fcnn                                      # the authors' network class
from envs.deadlineSchedulingEnv import deadlineSchedulingEnv  # the authors' env class

# Same constants as testingScripts/testDeadlineScheduling.py
STATESIZE = 2
SEED = 30
PROCESSINGCOST = 0.5
BETA = 0.99
TIMELIMIT = 300
MAXT, MAXB = 12, 9
EPISODES = list(range(0, 2001, 10))
MODELDIR = os.path.join(REPO, "trainResults/neurwin/deadline_env/")
OUTDIR = os.path.join(REPO, "replication/out")


def model_path(ep):
    return MODELDIR + f"seed_50_lr_0.001_batchSize_5_trainedNumEpisodes_{ep}/trained_model.pt"


def load_net(ep):
    net = fcnn(stateSize=STATESIZE)
    net.load_state_dict(torch.load(model_path(ep)))
    net.eval()
    return net


def neurwin_index(net, state):
    # Same call as calculateIndexNeuralNetwork() in the test script.
    return net.forward(state).detach().numpy()[0]


def closed_form_index(state):
    """Body of selectDeadlineIndex() in the test script, copied expression for expression.
    `state` is the float32 array [T, B] the env returns, exactly what the script passes in
    (so under numpy >= 2 the arithmetic is float32, as it is when the script runs)."""
    if state[1] == 0:
        return 0
    elif (state[1] >= 1) and (state[1] <= (state[0] - 1)):
        return 1 - PROCESSINGCOST
    elif (state[0] <= state[1]):
        firstVal = (BETA**(state[0]-1))*(0.2*(state[1] - state[0] + 1)**2)
        secondVal = (BETA**(state[0]-1))*(0.2*(state[1] - state[0])**2)
        return firstVal - secondVal + 1 - PROCESSINGCOST


def table(fn):
    """13 x 10 table, table[T][B], of an index function over every state."""
    return [[float(fn(np.array([T, B], dtype=np.float32))) for B in range(MAXB + 1)]
            for T in range(MAXT + 1)]


def getSelection(index, SCHEDULE):
    # Copied from the test script: repeated max, so ties go to the lowest arm number.
    result = []
    copyIndex = index.copy()
    for i in range(SCHEDULE):
        result.append(max(copyIndex.items(), key=operator.itemgetter(1))[0])
        del copyIndex[result[i]]
    return result


def rollout(envSeeds, index_fn, SCHEDULE):
    """One 300-step test run with the authors' env class, as in the test script's loops."""
    envs, states = {}, {}
    for i in range(len(envSeeds)):
        envs[i] = deadlineSchedulingEnv(seed=int(envSeeds[i]), numEpisodes=1, episodeLimit=TIMELIMIT,
                                        maxDeadline=12, maxLoad=9, newJobProb=0.7,
                                        processingCost=PROCESSINGCOST, train=False, batchSize=2000, noiseVar=0)
    for key in envs:
        states[key] = envs[key].reset()
    draws = [dict(seed=int(envSeeds[k]), jobList=envs[k].jobList.tolist(),
                  timeUntilDeadline=envs[k].timeUntilDeadline.tolist(), load=envs[k].load.tolist())
             for k in envs]
    rewards, chosen, seen, discounted = [], [], [], []
    for time in range(TIMELIMIT):
        seen.append([[int(states[k][0]), int(states[k][1])] for k in envs])
        index = {key: index_fn(states[key]) for key in envs}
        arms = getSelection(index, SCHEDULE)
        cumReward = 0
        for arm in arms:
            nextState, reward, done, info = envs[arm].step(1)
            cumReward += reward
            states[arm] = nextState
        for key in envs:
            if key in arms:
                pass
            else:
                nextState, reward, done, info = envs[key].step(0)
                cumReward += reward
                states[key] = nextState
        rewards.append(float(cumReward))
        chosen.append([int(a) for a in arms])
        discounted.append((BETA**time)*cumReward)
    return draws, dict(rewards=rewards, chosen=chosen, states=seen,
                       discountedTotal=float(np.cumsum(discounted)[-1]))


def main():
    # ---------- index tables ----------
    print("evaluating", len(EPISODES), "checkpoints on 13 x 10 states ...")
    nets = {ep: load_net(ep) for ep in EPISODES}
    neurwin_tables = [table(lambda s, n=nets[ep]: neurwin_index(n, s)) for ep in EPISODES]
    closed = table(closed_form_index)

    # ---------- chart data ----------
    c = pd.read_csv(os.path.join(OUTDIR, "curve_N4.csv"))
    g = c.groupby("episode").reward
    mean, p05, p95 = g.mean(), g.quantile(0.05), g.quantile(0.95)
    assert list(mean.index) == EPISODES
    index_mean = float(pd.read_csv(os.path.join(OUTDIR, "curve_N4_index.csv")).reward.mean())
    scale = pd.read_csv(os.path.join(OUTDIR, "scale.csv")).to_dict(orient="records")

    data = dict(
        meta=dict(beta=BETA, c=PROCESSINGCOST, maxT=MAXT, maxB=MAXB, episodeLimit=TIMELIMIT,
                  newJobProb=0.7, tableLayout="table[T][B], T = 0..12, B = 0..9"),
        closedForm=closed,
        checkpoints=EPISODES,
        neurwin=neurwin_tables,
        curve=dict(episode=EPISODES, mean=mean.tolist(), p05=p05.tolist(), p95=p95.tolist()),
        indexMean=index_mean,
        scale=scale,
    )
    with open(os.path.join(HERE, "data.js"), "w") as f:
        f.write("// Generated by export_data.py. Do not edit by hand.\n")
        f.write("window.SIM_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")

    # ---------- reference trace: N = 4, M = 1, 3 runs, the test script's seeding ----------
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    ARMS, SCHEDULE, RUNS = 4, 1, 3
    net2000 = nets[2000]
    ref_index = pd.read_csv(os.path.join(OUTDIR, "curve_N4_index.csv")).set_index("run").reward
    ref_nn = c[c.episode == 2000].set_index("run").reward
    runs = []
    for i in range(RUNS):
        envSeeds = np.random.randint(0, 10000, size=ARMS)
        draws, a = rollout(envSeeds, closed_form_index, SCHEDULE)
        draws2, b = rollout(envSeeds, lambda s: neurwin_index(net2000, s), SCHEDULE)
        assert draws == draws2, "both policies must see the same arrivals"
        # Cross-check against the replication's saved results for the same runs.
        for name, got, ref in (("index", a, ref_index[i]), ("neurwin", b, ref_nn[i])):
            diff = abs(got["discountedTotal"] - ref)
            print(f"run {i} {name:8s} discounted total {got['discountedTotal']:.6f}  saved {ref:.6f}  diff {diff:.1e}")
            assert diff < 1e-9, "trace does not reproduce the saved replication result"
        runs.append(dict(run=i, envSeeds=[int(s) for s in envSeeds], spots=draws, index=a, neurwin=b))
    trace = dict(N=ARMS, M=SCHEDULE, seed=SEED, checkpoint=2000, beta=BETA, c=PROCESSINGCOST,
                 steps=TIMELIMIT, runs=runs)
    with open(os.path.join(HERE, "trace.json"), "w") as f:
        json.dump(trace, f, separators=(",", ":"))

    print("wrote data.js  (%d KB)" % (os.path.getsize(os.path.join(HERE, "data.js")) // 1024))
    print("wrote trace.json (%d KB)" % (os.path.getsize(os.path.join(HERE, "trace.json")) // 1024))


if __name__ == "__main__":
    main()
