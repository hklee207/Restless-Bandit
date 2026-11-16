"""
Algorithm 1 of NeurWIN (Nakhleh et al., NeurIPS 2021), written with the paper's names, for the
deadline-scheduling (EV charging) arm of Sec. 5.2.

This file re-implements the training loop of the authors' NEURWIN class (neurwin.py) line by line
in the notation of Algorithm 1, so that the code can be read next to the paper. It reuses the
authors' objects without changing them:

  f_theta  = fcnn from neurwin.py (2 -> 16 -> 32 -> 1), built after the same torch seeding, with the
             same Adam optimizer as NEURWIN.__init__;
  the arm  = deadlineSchedulingEnv(train=True, batchSize=R, episodeLimit=300), the simulator of
             one charging spot that the authors train on.

The random draws happen in the same order as in the authors' code, so train(2000) reproduces the
checkpoints in trainResults/neurwin/deadline_env/ (see verify() at the bottom).

State of the arm: s = (D, B). D = rounds until the vehicle leaves (the code calls it T or
timeUntilDeadline), B = charge still needed. Charging a unit earns 1 - c (c = 0.5); a vehicle that
leaves with B units missing costs F(B) = 0.2 B^2.

Name map (paper -> Python): theta -> theta (parameters of f_theta), f_theta -> f_theta, beta -> beta,
L -> L, L_b -> L_b, m -> m, R -> R, s0 -> s0, s1 -> s1, lam -> lam, s[t] -> s_t, a[t] -> a_t,
r[t] -> r_t, sigma_m -> sigma_m, h_e -> h_e, G_e -> G_e, G_bar_b -> G_bar_b, theta+ -> theta_plus.

Run   python algorithm1.py          for the two checkpoint checks (10 episodes, then 2,000).
"""
import os, sys, io, math, random, contextlib, warnings

import numpy as np
import torch

NEURWIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if NEURWIN_DIR not in sys.path:
    sys.path.insert(0, NEURWIN_DIR)
with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
    warnings.simplefilter("ignore")              # gym prints a long "unmaintained" notice on import
    from neurwin import fcnn
    from envs.deadlineSchedulingEnv import deadlineSchedulingEnv

ROUNDS = 300                     # rounds per episode (EPISODELIMIT in main.py)
D_MAX, B_MAX = 12, 9             # largest D and B a vehicle can arrive with
ARRIVAL_PROB = 0.7               # a new vehicle arrives at an empty spot with this probability
c = 0.5                          # processing cost per unit charged
WATCH = [(2, 4), (1, 3), (6, 2)]  # states (D, B) whose f_theta the trace follows
CHECKPOINTS = os.path.join(NEURWIN_DIR, "trainResults", "neurwin", "deadline_env")


def sigma_m(x, m):
    """sigma_m(x) = 1 / (1 + exp(-m x)), the soft step that turns Env(lam) into Env*(lam)."""
    return torch.sigmoid(m * x)                  # neurwin.py: torch.sigmoid(self.sigmoidParam*(index - self.cost))


def f_theta_at(f_theta, D, B):
    """f_theta(D, B) as a plain number."""
    with torch.no_grad():
        return float(f_theta.forward(np.array([D, B], dtype=np.float32))[0])


def checkpoint_name(seed, L, R, episodes):
    """Folder name the authors' NEURWIN.close() uses for a checkpoint."""
    return f"seed_{seed}_lr_{L}_batchSize_{R}_trainedNumEpisodes_{episodes}"


def _save(f_theta, save_dir, seed, L, R, episodes):
    folder = os.path.join(save_dir, checkpoint_name(seed, L, R, episodes))
    os.makedirs(folder, exist_ok=True)
    torch.save(f_theta.state_dict(), os.path.join(folder, "trained_model.pt"))


def _flat(theta):
    return torch.cat([p.detach().reshape(-1) for p in theta])


def train(episodes, m=1, L=1e-3, beta=0.99, R=5, seed=50, save_dir=None, save_every=10, trace=False):
    """Algorithm 1: learn theta so that f_theta is a Whittle index of the deadline arm.

    episodes   number of episodes (must be a multiple of R; mini-batches = episodes / R)
    m          sensitivity of sigma_m
    L          learning rate (Adam's step size; L_b = L in every mini-batch)
    beta       discount factor
    R          episodes per mini-batch
    seed       seed for random, numpy, torch and both random generators (main.py uses 50)
    save_dir   if given, save f_theta every save_every episodes, named like the authors' checkpoints
    trace      if True, also return one record per mini-batch (see the end of the loop)

    Returns theta_plus (a state_dict of f_theta), or (theta_plus, trace_records) if trace=True.
    """
    if episodes % R:
        raise ValueError(f"episodes ({episodes}) must be a multiple of R ({R})")

    # ---- Inputs: theta, beta, L, m, R. Set up exactly like main.py + NEURWIN.__init__ ----------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    arm = deadlineSchedulingEnv(seed=seed, numEpisodes=episodes, episodeLimit=ROUNDS, maxDeadline=D_MAX,
                                maxLoad=B_MAX, newJobProb=ARRIVAL_PROB, processingCost=c, train=True,
                                batchSize=R, noiseVar=0.0)   # its own RandomState(seed) draws the vehicles
    torch.manual_seed(seed)                      # NEURWIN.__init__ seeds torch again, then builds the network
    G = np.random.RandomState(seed)              # the agent's generator (NEURWIN.G): draws s0 and every a_t
    f_theta = fcnn(2)                            # theta = the weights and biases of this network
    theta = list(f_theta.parameters())
    optimizer = torch.optim.Adam(theta, lr=L)    # L_b = L for every mini-batch (L15)

    records = []
    for b in range(1, episodes // R + 1):        # for each mini-batch b
        theta_before = _flat(theta) if trace else None
        watch_before = [f_theta_at(f_theta, D, B) for D, B in WATCH] if trace else None
        G_e_list, h_e_list, charged, first_rounds = [], [], [], []
        G_bar_b = 0.0                            # L2: G_bar_b <- 0

        for e in range(1, R + 1):                # for each episode e in the mini-batch
            done_so_far = (b - 1) * R + (e - 1)
            if save_dir is not None and done_so_far % save_every == 0:
                _save(f_theta, save_dir, seed, L, R, done_so_far)   # NEURWIN.learn -> close()

            # L4: set the arm to its initial state s1; h_e <- 0.            [authors: env.reset()]
            # In the code s1 is the first vehicle of an arrival sequence that env.reset() draws once
            # per mini-batch: all R episodes of a mini-batch see the SAME vehicles arriving at the
            # same rounds (the env replays its jobList). Algorithm 1 does not say this.
            s_t = arm.reset()
            if e == 1:
                s1 = (int(s_t[0]), int(s_t[1]))
            h_e = [torch.zeros_like(p) for p in theta]
            net_rewards, n_charged = [], 0

            for t in range(ROUNDS):              # for each round t
                f_s_t = f_theta.forward(s_t)     # f_theta(s[t])

                if e == 1 and t == 0:
                    # L2: choose s0 at random; lam <- f_theta(s0).          [authors: newMiniBatchReset]
                    # The code does this at the first round of the mini-batch (inside takeAction),
                    # with the agent's generator: B first, then D, uniform on 1..9 and 1..12.
                    B0 = G.randint(1, B_MAX + 1)
                    D0 = G.randint(1, D_MAX + 1)
                    s0 = np.array([D0, B0], dtype=np.float32)
                    lam = f_theta.forward(s0).detach().numpy()[0]   # activation cost, fixed for the mini-batch

                # L6: a[t] = 1 with probability sigma_m(f_theta(s[t]) - lam), else 0.   [authors: takeAction]
                sigma = sigma_m(f_s_t - lam, m)
                p = sigma.detach().numpy()[0]
                probs = np.array([p, 1 - p])
                probs /= probs.sum()
                a_t = G.choice([1, 0], 1, p=probs)[0]

                # L7-10: h_e += grad ln sigma_m(...) if a[t] = 1, else grad ln(1 - sigma_m(...)).
                # [authors: takeAction calls logProb.backward(), which adds this round's gradient to
                #  .grad; _saveEpisodeGradients copies the episode's sum out and zeroes .grad.]
                # Every round adds to h_e, whether the vehicle was charged or not.
                log_prob = torch.log(sigma) if a_t == 1 else torch.log(1 - sigma)
                for h, g in zip(h_e, torch.autograd.grad(log_prob, theta)):
                    h += g

                # One round of Env*(lam): the arm moves and pays r[t].
                s_next, r_t, _, _ = arm.step(a_t)
                # Net reward r[t] - lam * a[t].                         [authors: learn(), reward -= self.cost]
                net_t = r_t - lam if a_t == 1 else r_t
                net_rewards.append(net_t)
                n_charged += int(a_t)

                if trace and e == 1 and t < 3:
                    first_rounds.append(dict(t=t, s_t=(int(s_t[0]), int(s_t[1])), f_theta=float(f_s_t.detach()[0]),
                                             sigma=float(p), a_t=int(a_t), r_t=float(r_t), net=float(net_t)))
                s_t = s_next

            # L12: G_e <- discounted net reward of episode e.            [authors: _discountRewards]
            # (_discountRewards returns -G_e; the authors' code carries the minus sign from here on.)
            G_e = sum((beta ** t) * net_t for t, net_t in enumerate(net_rewards))
            # L13: G_bar_b <- G_bar_b + G_e / R  (the bootstrapped baseline: the mini-batch average)
            G_bar_b = G_bar_b + G_e / R
            G_e_list.append(G_e)
            h_e_list.append(h_e)
            charged.append(n_charged)

        # L15: L_b <- learning rate in mini-batch b (constant here: Adam with step size L).
        # L16: theta <- theta + L_b * sum_e (G_e - G_bar_b) * h_e.     [authors: _performBatchStep]
        # Adam takes a step DOWN the gradient stored in .grad, so we store minus the ascent
        # direction: theta moves UP sum_e (G_e - G_bar_b) h_e, i.e. gradient ascent on the
        # discounted net reward. (Adam rescales each coordinate, so the step is L-sized per weight,
        # not exactly L times the sum.)
        offsets = [G_e - G_bar_b for G_e in G_e_list]       # offset net rewards G_e - G_bar_b
        for k, p in enumerate(theta):
            ascent = torch.zeros_like(p)
            for e in range(R):
                ascent += offsets[e] * h_e_list[e][k]
            p.grad = -ascent
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)

        if trace:
            records.append(dict(
                b=b, s0=(int(D0), int(B0)), lam=float(lam), s1=s1,
                G_e=[float(x) for x in G_e_list], charged=charged, G_bar_b=float(G_bar_b),
                offsets=[float(x) for x in offsets],
                h_e_size=[float(torch.sqrt(sum((h ** 2).sum() for h in hs))) for hs in h_e_list],
                first_rounds=first_rounds,
                theta_change=float(torch.linalg.vector_norm(_flat(theta) - theta_before)),
                watch=[dict(D=D, B=B, before=wb, after=f_theta_at(f_theta, D, B))
                       for (D, B), wb in zip(WATCH, watch_before)],
            ))

    if save_dir is not None:
        _save(f_theta, save_dir, seed, L, R, episodes)
    theta_plus = {k: v.detach().clone() for k, v in f_theta.state_dict().items()}
    return (theta_plus, records) if trace else theta_plus


def f_theta_from(theta_plus):
    """An fcnn network loaded with theta_plus, ready to score states."""
    net = fcnn(2)
    net.load_state_dict(theta_plus)
    return net


def max_difference(theta_plus, checkpoint_episodes, folder=CHECKPOINTS, seed=50, L=1e-3, R=5):
    """Largest |difference| between theta_plus and an authors' checkpoint, over all 625 numbers."""
    path = os.path.join(folder, checkpoint_name(seed, L, R, checkpoint_episodes), "trained_model.pt")
    saved = torch.load(path)
    return max(float((theta_plus[k] - saved[k]).abs().max()) for k in saved)


def verify():
    import time
    for episodes in (10, 2000):
        start = time.time()
        theta_plus = train(episodes)
        diff = max_difference(theta_plus, episodes)
        print(f"{episodes:>5} episodes ({episodes // 5} mini-batches): max |theta_plus - checkpoint| = "
              f"{diff:.3g}   ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    verify()
