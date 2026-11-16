"""
Helper functions for explore.ipynb: Algorithm 1's inputs -> its output theta+ -> how theta+ is
used, for the NeurWIN deadline-scheduling (EV charging) example.

Nothing here changes the authors' code. Training runs algorithm1.train(), which reproduces the
authors' NEURWIN training exactly; testing follows their test loop the same way
replication/curve.py does (same env class, same seed, activate the M arms with the highest
score, ties to the lowest arm). States are written (D, B) as in the paper; the authors' code
calls D "T".
"""
import os, sys, io, re, glob, json, time, random, contextlib, warnings

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the NeurWIN folder
if REPO not in sys.path:
    sys.path.insert(0, REPO)

with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
    warnings.simplefilter("ignore")            # gym prints a long "unmaintained" notice on import
    from neurwin import fcnn
    from envs.deadlineSchedulingEnv import deadlineSchedulingEnv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import algorithm1                              # Algorithm 1 in the paper's notation (same folder)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- constants, as in main.py and testingScripts/testDeadlineScheduling.py ----
STATESIZE = 2
BATCHSIZE = 5            # gradient step every 5 episodes
DISCOUNT = 0.99          # beta
EPISODELIMIT = 300       # rounds per episode
PROCESSINGCOST = 0.5     # c
MAXDEADLINE, MAXLOAD, NEWJOBPROB = 12, 9, 0.7
TEST_SEED = 30           # SEED in the test script

DEFAULT_MODEL_DIR = os.path.join(REPO, "trainResults", "neurwin", "deadline_env")
EXPERIMENT_DIR = os.path.join(REPO, "trainResults", "experiments")

# ---- chart style (same colors as replication/plot.py: NeurWIN blue, index orange) ----
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
INDEX_COLOR = "#eb6834"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]      # experiments, in fixed order
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _show_table(df):
    print(df.to_string(index=False))


# ───────────────────────────── training ─────────────────────────────

REPLICATION_INPUTS = dict(episodes=2000, m=1.0, L=1e-3, R=5, beta=0.99, seed=50)   # main.py's deadline block


def read_inputs(folder):
    """The Algorithm 1 inputs an experiment folder was trained with (from its settings.json)."""
    with open(os.path.join(folder, "settings.json")) as f:
        s = json.load(f)
    if "sigmoid_m" in s:      # written by the first version of this file (authors' NEURWIN class, R=5, beta=0.99)
        s = dict(episodes=s["episodes"], m=s["sigmoid_m"], L=s["lr"], R=BATCHSIZE, beta=DISCOUNT, seed=s["seed"])
    return dict(episodes=int(s["episodes"]), m=float(s["m"]), L=float(s["L"]), R=int(s["R"]),
                beta=float(s["beta"]), seed=int(s["seed"]))


def _inputs(episodes, m, L, R, beta, seed):
    return dict(episodes=int(episodes), m=float(m), L=float(L), R=int(R), beta=float(beta), seed=int(seed))


def train(name, episodes=2000, m=1, L=1e-3, R=5, beta=0.99, seed=50, save_every=10):
    """Run Algorithm 1 (algorithm1.train) and save theta+ and its checkpoints in
    trainResults/experiments/<name>/. Never overwrites: if <name> exists with the same inputs it
    is reused, with other inputs you get an error (pick another name). Returns the folder."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(name)):
        raise ValueError("name may only contain letters, digits, '_', '-' and '.'")
    folder = os.path.join(EXPERIMENT_DIR, str(name)) + "/"
    inputs = _inputs(episodes, m, L, R, beta, seed)
    if os.path.exists(folder):
        old = read_inputs(folder) if os.path.exists(folder + "settings.json") else None
        if old == inputs and inputs["episodes"] in checkpoints(folder):
            print(f"'{name}' was already trained with these inputs; reusing {folder}")
            return folder
        raise ValueError(f"experiment '{name}' already exists with inputs {old}; "
                         f"it is never overwritten, so pick another name.")
    os.makedirs(folder)
    print(f"running Algorithm 1 for '{name}': " + ", ".join(f"{k}={v}" for k, v in inputs.items()) + " ...")
    start = time.time()
    algorithm1.train(inputs["episodes"], m=inputs["m"], L=inputs["L"], beta=inputs["beta"], R=inputs["R"],
                     seed=inputs["seed"], save_dir=folder, save_every=save_every)
    with open(folder + "settings.json", "w") as f:
        json.dump(dict(inputs, save_every=int(save_every)), f, indent=1)
    print(f"done in {time.time() - start:.0f} s, {len(checkpoints(folder))} checkpoints saved in {folder}")
    return folder


def theta_plus_dir(episodes=2000, m=1, L=1e-3, R=5, beta=0.99, seed=50):
    """Folder holding theta+ for these Algorithm 1 inputs. The replication's inputs give the
    authors' checkpoints (algorithm1.train reproduces them exactly); other inputs are trained
    once into trainResults/experiments/<inputs>/ and reused after that."""
    inputs = _inputs(episodes, m, L, R, beta, seed)
    if inputs == REPLICATION_INPUTS:
        return DEFAULT_MODEL_DIR
    name = "ep{episodes}_m{m:g}_L{L:g}_R{R}_beta{beta:g}_seed{seed}".format(**inputs)
    return train(name, **inputs)


def describe(folder):
    """Short label for a model folder, e.g. 'm5 (m = 5)'."""
    if os.path.normpath(folder) == os.path.normpath(DEFAULT_MODEL_DIR):
        return "Replication (m = 1)"
    name = os.path.basename(os.path.normpath(folder))
    try:
        inputs = read_inputs(folder)
    except (OSError, KeyError, ValueError):
        return name
    changed = [f"{k} = {v:g}" for k, v in inputs.items() if v != REPLICATION_INPUTS[k]]
    return f"{name} ({', '.join(changed)})" if changed else name


# ───────────────────────────── testing ─────────────────────────────

def checkpoints(model_dir=None):
    """{training episodes: path of trained_model.pt} for every checkpoint in a folder."""
    model_dir = DEFAULT_MODEL_DIR if model_dir is None else model_dir
    found = {}
    for path in glob.glob(os.path.join(model_dir, "*_trainedNumEpisodes_*", "trained_model.pt")):
        ep = int(re.search(r"_trainedNumEpisodes_(\d+)$", os.path.dirname(path)).group(1))
        if ep in found:
            raise ValueError(f"{model_dir} holds more than one model for episode {ep}")
        found[ep] = path
    if not found:
        raise FileNotFoundError(f"no checkpoints found in {model_dir}")
    return dict(sorted(found.items()))


def closed_form(D, B):
    """Closed-form deadline Whittle index, as selectDeadlineIndex() in the test script."""
    if B == 0:
        return 0.0
    if 1 <= B <= D - 1:
        return 1 - PROCESSINGCOST
    return DISCOUNT**(D - 1)*0.2*((B - D + 1)**2 - (B - D)**2) + 1 - PROCESSINGCOST


def index_table():
    """Closed-form index for every state: {(D, B): score}."""
    return {(D, B): closed_form(D, B) for D in range(MAXDEADLINE + 1) for B in range(MAXLOAD + 1)}


def score_table(model_path):
    """A trained network's score for every state: {(D, B): score}. The network is only
    ever asked about these 130 states, so a lookup table gives the same choices."""
    net = fcnn(STATESIZE)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    grid = np.array([[D, B] for D in range(MAXDEADLINE + 1) for B in range(MAXLOAD + 1)], dtype=np.float32)
    with torch.no_grad():
        values = [float(net.forward(s)[0]) for s in grid]
    return {(int(D), int(B)): v for (D, B), v in zip(grid, values)}


def rollout(score, N, M, runs):
    """The authors' test loop: `runs` episodes of 300 rounds with N arms, charging the M
    highest scores each round. Returns the total discounted reward of each run."""
    random.seed(TEST_SEED); np.random.seed(TEST_SEED); torch.manual_seed(TEST_SEED)
    totals = []
    for _ in range(runs):
        seeds = np.random.randint(0, 10000, size=N)
        envs = [deadlineSchedulingEnv(seed=int(s), numEpisodes=1, episodeLimit=EPISODELIMIT,
                                      maxDeadline=MAXDEADLINE, maxLoad=MAXLOAD, newJobProb=NEWJOBPROB,
                                      processingCost=PROCESSINGCOST, train=False, batchSize=2000, noiseVar=0)
                for s in seeds]
        state = [(int(a), int(b)) for a, b in (e.reset() for e in envs)]
        total = 0.0
        for t in range(EPISODELIMIT):
            # authors' getSelection(): repeated max, so ties go to the lowest arm number
            chosen = set(sorted(range(N), key=lambda k: (-score[state[k]], k))[:M])
            r = 0.0
            for k, e in enumerate(envs):
                s, reward, _, _ = e.step(1 if k in chosen else 0)
                state[k] = (int(s[0]), int(s[1]))
                r += reward
            total += DISCOUNT**t*r
        totals.append(total)
    return totals


def evaluate(model_dir=None, N=4, M=1, runs=50, stride=50):
    """Test every `stride`-th checkpoint in model_dir (0, stride, 2*stride, ...).

    Returns a DataFrame with columns episode, run, reward (total discounted reward).
    The closed-form index is tested on the same N, M and runs; its per-run rewards are in
    result.attrs["index"] (also attrs N, M, runs, model_dir)."""
    if not 1 <= M <= N:
        raise ValueError("need 1 <= M <= N")
    model_dir = DEFAULT_MODEL_DIR if model_dir is None else model_dir
    found = checkpoints(model_dir)
    episodes = [ep for ep in found if ep % stride == 0]
    if not episodes:
        raise ValueError(f"no checkpoint is a multiple of stride={stride}; saved: {list(found)[:6]} ...")
    start = time.time()
    rows = []
    for ep in episodes:
        for run, reward in enumerate(rollout(score_table(found[ep]), N, M, runs)):
            rows.append(dict(episode=ep, run=run, reward=reward))
    result = pd.DataFrame(rows)
    result.attrs.update(index=rollout(index_table(), N, M, runs), N=N, M=M, runs=runs, model_dir=str(model_dir))
    print(f"tested {len(episodes)} checkpoints (episodes {episodes[0]} to {episodes[-1]}) and the index: "
          f"N={N}, M={M}, {runs} runs each, {time.time() - start:.0f} s")
    return result


def summary(result, label=""):
    """One row of numbers for a result from evaluate()."""
    mean = result.groupby("episode").reward.mean()
    last = int(mean.index.max())
    final = result[result.episode == last].sort_values("run").reward.to_numpy()
    index = np.array(result.attrs["index"])
    d = final - index
    reached = mean[mean >= index.mean()]
    return {
        "experiment": label,
        "N": result.attrs["N"], "M": result.attrs["M"],
        "last episode": last,
        "NeurWIN": round(final.mean(), 2),
        "index": round(index.mean(), 2),
        "NeurWIN - index": round(d.mean(), 2),
        "std. error": round(d.std(ddof=1)/np.sqrt(len(d)), 2) if len(d) > 1 else float("nan"),
        "NeurWIN better in": f"{int((d > 0).sum())}/{len(d)}",
        "first tested episode >= index": int(reached.index.min()) if len(reached) else "never",
    }


# ───────────────────────────── plots ─────────────────────────────

def plot_curve(results, labels=None, title=None):
    """Reward against training episodes: mean line with a 5th to 95th percentile band for each
    result, and a dashed line for the closed-form index. Prints the numbers underneath."""
    if isinstance(results, pd.DataFrame):
        results = [results]
    if labels is None:
        labels = [f"experiment {i + 1}" for i in range(len(results))]
    if isinstance(labels, str):
        labels = [labels]
    if len(results) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} experiments in one chart")
    setups = {(r.attrs["N"], r.attrs["M"], r.attrs["runs"]) for r in results}

    fig, ax = plt.subplots(figsize=(9.5, 5), facecolor=SURFACE)
    _style(ax)
    for result, label, color in zip(results, labels, SERIES):
        g = result.groupby("episode").reward
        mean, lo, hi = g.mean(), g.quantile(0.05), g.quantile(0.95)
        ax.fill_between(mean.index, lo, hi, color=color, alpha=0.13, linewidth=0)
        ax.plot(mean.index, mean.values, color=color, lw=2, label=f"{label} (mean, 5-95% band)")
        if len(setups) > 1:      # different N, M per result: each gets its own index line
            ax.axhline(np.mean(result.attrs["index"]), color=color, lw=2, ls="--",
                       label=f"index for {label}")
    if len(setups) == 1:
        N, M, runs = next(iter(setups))
        ax.axhline(np.mean(results[0].attrs["index"]), color=INDEX_COLOR, lw=2, ls="--",
                   label="Closed-form Whittle index")
        default_title = f"Deadline scheduling, N = {N}, M = {M}, {runs} test runs"
    else:
        default_title = "Deadline scheduling"
    ax.set_title(title or default_title, loc="left", fontsize=13)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Total discounted reward")
    ax.set_xlim(0, max(r.episode.max() for r in results))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=10, labelcolor=INK)
    plt.tight_layout()
    plt.show()
    _show_table(pd.DataFrame([summary(r, l) for r, l in zip(results, labels)]))


def plot_scaling(model_dir=None, sizes=((4, 1), (8, 2), (20, 5)), runs=50):
    """Bar chart of (NeurWIN - index) as a percentage of the index, for several station
    sizes, using the last checkpoint in model_dir. Prints the numbers underneath."""
    model_dir = DEFAULT_MODEL_DIR if model_dir is None else model_dir
    found = checkpoints(model_dir)
    last = max(found)
    score, cf = score_table(found[last]), index_table()
    rows = []
    for N, M in sizes:
        nw, ix = np.array(rollout(score, N, M, runs)), np.array(rollout(cf, N, M, runs))
        d = nw - ix
        rows.append({"N": N, "M": M, "NeurWIN": round(nw.mean(), 2), "index": round(ix.mean(), 2),
                     "NeurWIN - index": round(d.mean(), 2),
                     "% of index": round(100*d.mean()/abs(ix.mean()), 2),
                     "std. error": round(d.std(ddof=1)/np.sqrt(runs), 2) if runs > 1 else float("nan"),
                     "NeurWIN better in": f"{int((d > 0).sum())}/{runs}"})
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(1.6*len(rows) + 3, 4.2), facecolor=SURFACE)
    _style(ax)
    x = np.arange(len(rows))
    pct = table["% of index"].to_numpy()
    ax.bar(x, pct, width=0.5, color=SERIES[0])
    ax.axhline(0, color=AXIS, lw=1)
    pad = 0.03*max(1e-9, np.abs(pct).max())
    for xi, p in zip(x, pct):
        ax.text(xi, p + (pad if p >= 0 else -pad), f"{p:+.1f}%", ha="center",
                va="bottom" if p >= 0 else "top", color=INK, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N = {r['N']}, M = {r['M']}" for r in rows])
    ax.set_ylabel("NeurWIN - index, % of index")
    ax.set_title(f"NeurWIN's edge over the index by station size (checkpoint {last}, {runs} runs)",
                 loc="left", fontsize=13)
    ax.margins(y=0.18)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()
    _show_table(table)


def whittle_accuracy(score):
    """How close a score table is to being Whittle-accurate (Def. 4) on the 120 states with a
    vehicle slot in use (D = 1..12, B = 0..9). An index only matters through its order: f is
    Whittle-accurate when, for every activation cost lam, f(s) > lam exactly where W(s) > lam,
    i.e. f orders the states like W. Returns (rank correlation, share of pairs with W(s) > W(s')
    that f also puts in that order)."""
    W = index_table()
    states = [(D, B) for D in range(1, MAXDEADLINE + 1) for B in range(MAXLOAD + 1)]
    f = np.array([score[s] for s in states]); w = np.array([W[s] for s in states])
    rank = lambda v: pd.Series(v).rank().to_numpy()
    rho = float(np.corrcoef(rank(f), rank(w))[0, 1])
    i, j = np.where(w[:, None] > w[None, :] + 1e-12)
    return rho, float((f[i] > f[j]).mean())


def plot_theta_plus(model_dirs, episode=None):
    """Heatmaps of the output theta+, f_theta(D, B), for each model folder, next to the
    closed-form Whittle index W(D, B), over the 120 states D = 1..12, B = 0..9. Each grid has
    its own color scale (an index only matters through its order). Prints how Whittle-accurate
    each f_theta is."""
    if isinstance(model_dirs, (str, os.PathLike)):
        model_dirs = [model_dirs]
    Ds, Bs = range(1, MAXDEADLINE + 1), range(MAXLOAD + 1)
    W = index_table()
    panels, rows = [], []
    for folder in model_dirs:
        found = checkpoints(folder)
        ep = max(found) if episode is None else episode
        score = score_table(found[ep])
        panels.append((f"f_theta+  {describe(folder)}, episode {ep}", score))
        rho, pairs = whittle_accuracy(score)
        rows.append({"model": describe(folder), "episode": ep, "rank correlation with W": round(rho, 3),
                     "pairs ordered like W": f"{100 * pairs:.1f}%"})
    panels.append(("W(D, B)  closed-form Whittle index", W))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 6.6), facecolor=SURFACE, sharey=True)
    for ax, (name, table) in zip(np.atleast_1d(axes), panels):
        grid = np.array([[table[(D, B)] for B in Bs] for D in Ds])
        lo, hi = grid.min(), grid.max()
        ax.imshow(grid, cmap=BLUES, vmin=lo, vmax=hi, aspect="auto")
        for r, D in enumerate(Ds):
            for k, B in enumerate(Bs):
                dark = (grid[r, k] - lo) / max(hi - lo, 1e-12) > 0.55
                ax.text(k, r, f"{grid[r, k]:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if dark else INK)
        ax.set_xticks(range(len(Bs))); ax.set_xticklabels(list(Bs))
        ax.set_yticks(range(len(Ds))); ax.set_yticklabels(list(Ds))
        ax.set_xticks(np.arange(-0.5, len(Bs)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(Ds)), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)          # thin gaps between cells
        ax.tick_params(which="both", length=0, colors=INK2)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel("B: charge still needed", color=INK2)
        ax.set_title(name, loc="left", fontsize=11.5, color=INK)
    np.atleast_1d(axes)[0].set_ylabel("D: rounds until the vehicle leaves", color=INK2)
    fig.suptitle("Darker = higher score = activate first (each grid has its own color scale)",
                 x=0.01, ha="left", fontsize=10.5, color=INK2)
    plt.tight_layout()
    plt.show()
    print("How Whittle-accurate theta+ is (Def. 4), over the 120 states D = 1..12, B = 0..9:")
    _show_table(pd.DataFrame(rows))
