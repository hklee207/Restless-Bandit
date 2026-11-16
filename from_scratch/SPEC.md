# SPEC — clean-room rebuild of NeurWIN deadline scheduling (Sec. 5.2)

Source: `paper/restless_bandit.pdf` only (12 pages). Page numbers are the
printed page footers. "Alg. 1 line k" counts the lines of the Algorithm 1 box
on p. 6 as numbered below. Nothing in `NeurWIN/` or on GitHub was opened.

Algorithm 1 (p. 6) with line numbers used throughout the code:

```
 1  for each mini-batch b do
 2    choose s0, s1 uniformly at random; lam <- f_theta(s0); G_bar_b <- 0
 3    for each episode e in the mini-batch do
 4      set the arm to initial state s1; h_e <- 0
 5      for each round t in the episode do
 6        a_t = 1 w.p. sigma_m(f_theta(s_t) - lam), else a_t = 0
 7        if a_t = 1 then
 8          h_e <- h_e + grad_theta ln sigma_m(f_theta(s_t) - lam)
 9        else
10          h_e <- h_e + grad_theta ln(1 - sigma_m(f_theta(s_t) - lam))
11        end if
12      end for
13      G_e <- empirical discounted net reward in episode e
14      G_bar_b <- G_bar_b + G_e / R
15    end for
16    L_b <- learning rate in mini-batch b
17    theta <- theta + L_b * sum_e (G_e - G_bar_b) h_e
18  end for
```

---

## (a) Quantities stated in the paper

| Quantity | Code name | Value / definition | Where |
|---|---|---|---|
| Arm simulator: set any state, `step(a) -> (s_{t+1}, r_t)` | `arm.py` | black box, two functions | Sec. 3.3, p. 4, ¶1 |
| Network output | `f_theta(s)` | real number; theta = all weights and biases | Sec. 3.3, p. 4, ¶2 |
| Activation cost | `lam` (λ) | net reward r_t − λ·a_t | Sec. 3.2 p. 3; Sec. 4.1 / Fig. 1 p. 5 |
| Training cost | `lam = f_theta(s0)` | set once per mini-batch | Alg. 1 line 2; Sec. 4.2 p. 6 |
| Sigmoid | `sigma_m(x) = 1/(1+exp(-m x))` | x = f_theta(s_t) − λ | Eq. (1), p. 6 |
| Sensitivity | `m` | "a sensitivity parameter" (no value given) | Eq. (1), p. 6 |
| Random action | `a_t ~ Bernoulli(sigma_m(...))` | the Env*(λ) environment | Alg. 1 line 6; p. 6 below Eq. 1 |
| Score-function sum | `h_e` | sum over t of grad ln P(a_t) | Alg. 1 lines 7–11 |
| Episode return | `G_e` | discounted net reward of episode e | Alg. 1 line 13 |
| Baseline | `G_bar_b` | mean of G_e over the R episodes | Alg. 1 line 14; p. 6 last ¶ |
| Update | `theta += L_b * sum_e (G_e - G_bar_b) h_e` | gradient ascent | Alg. 1 line 17 |
| Output | `theta_plus` | trained parameters | Alg. 1 header |
| Episodes per mini-batch | `R = 5` | "Each mini-batch consists of five episodes" | Sec. 5.1, p. 7, last ¶ |
| Learning rate | `L = 0.001` | "initial learning rate", Adam | Sec. 5.1, p. 7, last ¶ |
| Discount | `beta = 0.99` | | Sec. 5.1, p. 7, last ¶ |
| Episode horizon | 300 rounds | | Sec. 5.1, p. 7, last ¶ |
| Network | 2 → 16 → 32 → 1, ReLU on both hidden layers, fully connected | input size = state dimension = 2 | Sec. 5.1, p. 7, last ¶ |
| State | `(D, B)` integers, B ≤ 9, D ≤ 12, **"size of state space is 120"** | | Sec. 5.2, p. 7, ¶3 |
| Charging reward | `1 - c` per unit delivered | | Sec. 5.2, p. 7, ¶1 |
| Penalty | `F(B)`, B = unfulfilled charge when vehicle leaves | | Sec. 5.2, p. 7, ¶2 |
| Arrivals | an available spot "may" get a new vehicle, which announces D and B | | Sec. 5.2, p. 7, ¶1 |
| Each charging spot is one arm; N spots, power for M | | | Sec. 5.2, p. 7, ¶1–2 |
| Test policy | score each arm, activate top M | | Sec. 5.1, p. 7, ¶1 |
| Test settings | (N, M) = (4, 1) [also (10,1), (100,25) not rebuilt]; 50 independent runs | | Sec. 5.1, p. 7, ¶1 |
| Training length | 2,000 training episodes (x-axis of Fig. 2) | | Fig. 2, p. 8 |
| Claimed result | "converges to that of the deadline Whittle index in 600 training episodes" | | Sec. 5.2, p. 7 last line |
| Test metric | "Total Discounted Rewards" | | Fig. 2 y-axis, p. 8 |

## (b) Not specified by the paper: what we assume

| # | Item | Assumed value | Source |
|---|---|---|---|
| 1 | Charging reward | c = 0.5, so 1 − c = 0.5 | **assumed**: ref [34], from your replication notes |
| 2 | Penalty | F(B) = 0.2·B² | **assumed**: ref [34], from your replication notes |
| 3 | Arrival | an empty spot gets a new vehicle w.p. 0.7 | **assumed**: ref [34], from your replication notes |
| 4 | New vehicle's (D, B) | D uniform on 1..12, B uniform on 1..9, independent | **assumed**: ranges from ref [34] via your notes; "uniform, independent" is **my choice — please confirm** |
| 5 | Sigmoid sensitivity | m = 1 | **assumed**: paper's appendix (not in this PDF), via your notes |
| 6 | s0, s1 | each uniform over vehicle states D 1..12, B 1..9 (108 states), drawn independently | **assumed**: your spec |
| 7 | Round dynamics | see "Round dynamics" below | **assumed**: your spec, with timing fixed by me |
| 8 | Baseline index | W(D,B) = 0 if B = 0; 1 − c if 1 ≤ B ≤ D−1; β^(D−1)·F(B−D+1) − β^(D−1)·F(B−D) + 1 − c if B ≥ D | **assumed**: ref [34], from your replication notes |
| 9 | Empty spot as a state | (D, B) = (0, 0); reward 0 whatever a_t is; W(0,0) = 0 | **my choice — please confirm** |
| 10 | "120 states" | Your ranges give 12 × 9 = 108 vehicle states. 120 = 12 × 10 comes from **D 1..12, B 0..9**: B = 0 is a fully charged vehicle still parked, which charging can reach. I use those 120 states for the f_theta-vs-W rank correlation. s0 and s1 stay on the 108 states with B ≥ 1 (item 6). | **my choice — please confirm** |
| 11 | Network input | raw integers (D, B), no scaling | **my choice — please confirm** (the paper says nothing; I won't tune this) |
| 12 | λ in the gradient | λ = f_theta(s0) is computed once at line 2 and treated as a fixed number: no gradient flows through λ. This is how Alg. 1 reads ("set λ ←"), and line 8 differentiates only through f_theta(s_t). | **my choice — please confirm** |
| 13 | Learning-rate schedule | L_b = 0.001 constant. The paper says "initial" but gives no decay. | **my choice — please confirm** |
| 14 | Adam with ascent | Adam minimizes −Σ_e (G_e − G_bar_b) · Σ_t ln P_theta(a_t), which is the same as the line-17 ascent step | standard; Sec. 5.1 names Adam |
| 15 | Discounting | G_e = Σ_{t=0}^{299} β^t (r_t − λ a_t); the test reward is Σ_{t=0}^{299} β^t Σ_i r_i,t | **my choice**: t starts at 0 as in Sec. 3.1 |
| 16 | Episode count | "training episodes" = individual episodes, so 2,000 episodes = 400 mini-batches; checkpoint every 10 episodes = every 2 mini-batches, plus theta at episode 0 → 201 checkpoints | **my choice — please confirm** |
| 17 | Actions on empty or full arms | During training, line 6 samples a_t and adds to h_e **every** round, including empty or B = 0 rounds (activating costs λ, earns 0). This follows Alg. 1 literally. | **my choice** |
| 18 | Test start state | each of the N arms starts with a fresh vehicle (item 4) | **my choice — please confirm** |
| 19 | Test tie-breaking | ties in the score broken uniformly at random. The closed-form index has many ties (W = 0.5 for every B ≤ D−1). | **my choice — please confirm** |
| 20 | Test randomness | run k (k = 0..49) uses seed k for arrivals, and that same arrival stream is used for every policy. This makes the comparisons paired. | **my choice** |
| 21 | SHARED / INDEPENDENT | SHARED: one arrival stream (arrival coin flips + new vehicles' (D, B), one per round) is drawn per mini-batch and reused in all R episodes. INDEPENDENT: a fresh stream per episode. Action sampling uses a separate RNG in both. | your spec; mechanics **my choice** |

### Round dynamics (item 7), exactly

The state at the start of round t is s_t = (D, B). D = rounds left **including
this one**, B = units still needed.

Vehicle present (D ≥ 1):
1. If a_t = 1 and B > 0: deliver 1 unit, r_t += 1 − c, B ← B − 1.
   If a_t = 1 and B = 0: nothing happens (r_t += 0).
2. D ← D − 1.
3. If D = 0, the vehicle leaves **at the end of round t**. The penalty is
   paid **in this same round**: r_t −= F(B), with B as it is after step 1.
   The spot is now empty.

Spot empty at the end of round t (just emptied, or already empty): with
probability 0.7 a new vehicle arrives and s_{t+1} = new (D, B) (item 4).
Otherwise s_{t+1} = (0, 0). An empty spot earns r_t = 0.

**Why the penalty is paid in round t (the last round, D = 1).** A vehicle
with D rounds left can be charged in rounds t, …, t+D−1. Its last round is
t+D−1, which is discounted by β^(D−1). The closed-form index (item 8)
charges the penalty difference F(B−D+1) − F(B−D) with factor exactly
β^(D−1). So the penalty must be applied in the vehicle's last round, not in
the round after it. Booking it one round later would make the simulator
inconsistent with W by a factor of β.

**Why the arrival shows up at s_{t+1}.** The spot becomes free at the end of
round t, so the earliest a new vehicle can be served is round t+1. This
means a spot never idles a round on purpose: "an empty spot may get a new
vehicle that round" means the vehicle arrives in the round the spot frees
and is present from the next state on. An already-empty spot gets the same
0.7 chance every round.

Side effect worth knowing: departures happen at fixed times (D counts down
regardless of a_t). So under SHARED, all R episodes of a mini-batch have
**identical vehicle timelines**. They differ only in how much each vehicle
is charged and therefore in the penalties.

## (c) Test protocol (Sec. 5.1)

- N = 4 arms (spots), M = 1 activation per round, all arms of one type, so a
  single f_theta scores all four.
- Each round: score each arm (f_theta(s_i,t) for NeurWIN, W(s_i,t) for the
  baseline), activate the top M = 1 (ties: item 19), step all arms.
- 300 rounds per run, 50 independent runs (seeds 0..49, item 20).
- Metric: total discounted reward Σ_t β^t Σ_i r_i,t, β = 0.99.
- Done for every saved checkpoint (201 per variant) and for the closed-form
  index W.
- Index check: Spearman rank correlation between f_theta(D,B) and W(D,B)
  over the 120 states D 1..12, B 0..9 (item 10).

## Training runs (Phase 2)

Two variants (SHARED, INDEPENDENT), each one training run with seed 50:
Adam, L = 0.001, β = 0.99, horizon 300, R = 5, m = 1, 2,000 episodes,
theta saved every 10 episodes. No tuning after seeing results.

## What we compare against (Phase 3)

Your replication of the authors' code, N = 4, M = 1: NeurWIN −99.3, index
−105.1, first checkpoint ≥ index at 570 episodes, rank correlation 0.78.
The paper itself claims convergence to the index by about 600 episodes
(Sec. 5.2). We judge by behavior: crossing between ~400 and ~900 episodes,
ending near or above the index, similar state ranking.
