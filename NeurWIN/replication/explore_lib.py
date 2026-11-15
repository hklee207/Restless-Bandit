"""
Helper functions for explore.ipynb: train, evaluate and plot the NeurWIN deadline-scheduling
(EV charging) example with different settings.

Nothing here changes the authors' code. Training uses their NEURWIN class and env exactly as
the deadline block of main.py does; testing follows their test loop the same way
replication/curve.py does (same env class, same seed, top-M by score, ties to the lowest arm).
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
    from neurwin import NEURWIN, fcnn
    from envs.deadlineSchedulingEnv import deadlineSchedulingEnv

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

def train(name, episodes=2000, sigmoid_m=1, seed=50, lr=1e-3, save_every=50, force=False, verbose=False):
    """Train a new NeurWIN network on the deadline env, set up and seeded like main.py.

    Checkpoints go to trainResults/experiments/<name>/, never to the replication's folder.
    If <name> was already trained with the same settings it is reused; with different settings
    you get an error unless force=True. Returns the folder.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(name)):
        raise ValueError("name may only contain letters, digits, '_', '-' and '.'")
    folder = os.path.join(EXPERIMENT_DIR, str(name)) + "/"
    settings = dict(episodes=int(episodes), sigmoid_m=float(sigmoid_m), seed=int(seed), lr=float(lr),
                    save_every=int(save_every))
    settings_file = folder + "settings.json"
    if os.path.exists(settings_file) and not force:
        with open(settings_file) as f:
            old = json.load(f)
        if old == settings and int(episodes) in checkpoints(folder):
            print(f"'{name}' is already trained with these settings; reusing {folder}")
            return folder
        raise ValueError(f"experiment '{name}' exists with other settings {old}. "
                         f"Pick a new name, or pass force=True to train over it.")
    os.makedirs(folder, exist_ok=True)

    # --- from here: main.py, deadline block ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = deadlineSchedulingEnv(seed=seed, numEpisodes=episodes, episodeLimit=EPISODELIMIT,
                                maxDeadline=MAXDEADLINE, maxLoad=MAXLOAD, newJobProb=NEWJOBPROB,
                                processingCost=PROCESSINGCOST, train=True, batchSize=BATCHSIZE, noiseVar=0.0)
    agent = NEURWIN(stateSize=STATESIZE, lr=lr, env=env, sigmoidParam=sigmoid_m, numEpisodes=episodes,
                    noiseVar=0.0, seed=seed, batchSize=BATCHSIZE, discountFactor=DISCOUNT,
                    saveDir=folder, episodeSaveInterval=save_every)
    print(f"training '{name}': {episodes} episodes, sigmoid m = {sigmoid_m}, seed {seed}, lr {lr} ...")
    start = time.time()
    if verbose:
        agent.learn()
    else:                                        # learn() prints a line per episode
        with contextlib.redirect_stdout(io.StringIO()):
            agent.learn()
    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=1)
    print(f"done in {time.time() - start:.0f} s, {len(checkpoints(folder))} checkpoints saved in {folder}")
    return folder


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


def closed_form(T, B):
    """Closed-form deadline Whittle index, as selectDeadlineIndex() in the test script."""
    if B == 0:
        return 0.0
    if 1 <= B <= T - 1:
        return 1 - PROCESSINGCOST
    return DISCOUNT**(T - 1)*0.2*((B - T + 1)**2 - (B - T)**2) + 1 - PROCESSINGCOST


def index_table():
    """Closed-form index for every state: {(T, B): score}."""
    return {(T, B): closed_form(T, B) for T in range(MAXDEADLINE + 1) for B in range(MAXLOAD + 1)}


def score_table(model_path):
    """A trained network's score for every state: {(T, B): score}. The network is only
    ever asked about these 130 states, so a lookup table gives the same choices."""
    net = fcnn(STATESIZE)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    grid = np.array([[T, B] for T in range(MAXDEADLINE + 1) for B in range(MAXLOAD + 1)], dtype=np.float32)
    with torch.no_grad():
        values = [float(net.forward(s)[0]) for s in grid]
    return {(int(T), int(B)): v for (T, B), v in zip(grid, values)}


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


def plot_score_tables(model_dir=None, episode=None):
    """Two heatmaps side by side on one color scale: NeurWIN's learned score and the
    closed-form index, for every state (T, B). Prints a short comparison underneath."""
    model_dir = DEFAULT_MODEL_DIR if model_dir is None else model_dir
    found = checkpoints(model_dir)
    episode = max(found) if episode is None else episode
    nw, cf = score_table(found[episode]), index_table()
    Ts, Bs = range(MAXDEADLINE + 1), range(MAXLOAD + 1)
    A = np.array([[nw[(T, B)] for B in Bs] for T in Ts])
    C = np.array([[cf[(T, B)] for B in Bs] for T in Ts])
    lo, hi = min(A.min(), C.min()), max(A.max(), C.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=SURFACE, sharey=True)
    for ax, grid, name in ((axes[0], A, f"NeurWIN score (checkpoint {episode})"),
                           (axes[1], C, "Closed-form Whittle index")):
        im = ax.imshow(grid, cmap=BLUES, vmin=lo, vmax=hi, aspect="auto")
        for T in Ts:
            for B in Bs:
                dark = (grid[T, B] - lo)/(hi - lo) > 0.55
                ax.text(B, T, f"{grid[T, B]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if dark else INK)
        ax.set_xticks(list(Bs)); ax.set_yticks(list(Ts))
        ax.set_xticks(np.arange(-0.5, len(Bs)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(Ts)), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)          # thin gaps between cells
        ax.tick_params(which="both", length=0, colors=INK2)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel("B: charge still needed", color=INK2)
        ax.set_title(name, loc="left", fontsize=13, color=INK)
    axes[0].set_ylabel("T: rounds until the car leaves", color=INK2)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK2, length=0)
    cbar.set_label("score (higher = charge first)", color=INK2)
    plt.show()

    cars = [(T, B) for T in range(1, MAXDEADLINE + 1) for B in range(1, MAXLOAD + 1)]   # a car that still needs charge
    a = np.array([nw[s] for s in cars]); c = np.array([cf[s] for s in cars])
    rank = lambda v: pd.Series(v).rank().to_numpy()
    print(f"Over the {len(cars)} states with a car that still needs charge (T >= 1, B >= 1):")
    print(f"  correlation of the two scores: {np.corrcoef(a, c)[0, 1]:.3f}   "
          f"rank correlation: {np.corrcoef(rank(a), rank(c))[0, 1]:.3f}")
    print(f"  mean absolute difference: {np.abs(a - c).mean():.2f}")
    worst = sorted(cars, key=lambda s: -abs(nw[s] - cf[s]))[:5]
    print("  largest differences:")
    _show_table(pd.DataFrame([{"T": T, "B": B, "NeurWIN": round(nw[(T, B)], 2), "index": round(cf[(T, B)], 2),
                               "NeurWIN - index": round(nw[(T, B)] - cf[(T, B)], 2)} for T, B in worst]))
