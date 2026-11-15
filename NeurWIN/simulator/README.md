# Simulator: Whittle index vs NeurWIN on the EV-charging example

A single web page that runs two copies of the same charging station side by side, on the
same car arrivals. One station is run by the closed-form deadline Whittle index, the other
by the trained NeurWIN network. It works offline and needs no libraries.

## What it shows

- **Two stations, same cars.** Each spot card shows the car, the rounds until it leaves (T),
  the charge it still needs (B) and the spot's current index. The spot being charged this
  round is highlighted. A car with B >= T is shown in red.
- **Why each choice was made**, and a count of the rounds where the two stations chose differently.
- **Totals per station:** units charged, penalty paid, net reward, and the discounted total
  (beta = 0.99, the paper's metric).
- **Live chart** of cumulative net reward for both stations.
- **Training chart** (NeurWIN mean with a 5th to 95th percentile band, against the index).
  Click or drag on it to choose which training checkpoint (0 to 2000, step 10) runs the
  NeurWIN station. 0 is the untrained network.
- **Run 50 episodes:** 50 seeds (the seed in the box, and the next 49) at the current N, M
  and checkpoint. Reports both means and the paired difference with its standard error.
- **Scaling table** from the replication (`replication/out/scale.csv`).

## How to run

Double-click `index.html`, or serve the folder:

```bash
cd NeurWIN/simulator
python3 -m http.server 8000      # then open http://localhost:8000
```

`index.html?steps=40` fast-forwards 40 rounds on load (used for testing).

## How the rules are copied

`sim.js` (part 1) is a line-by-line port of `envs/deadlineSchedulingEnv.py` (`reset`, `step`,
`_calRewardAndState`, `_newArrival`) and of `getSelection()` in
`testingScripts/testDeadlineScheduling.py` (charge the M highest indices, ties to the lowest
spot number). No existing file in the repo was changed.

There are only 13 x 10 possible states (T = 0..12, B = 0..9), so each policy is stored as a
lookup table of its index for every state. The page never runs a network; it reads the table.

The only thing that differs from Python is the random generator that draws arrivals. The
distributions are the same (a new car with probability 0.7, T uniform on 1..12, B uniform on
1..9), but a given seed number does not give the same cars as the Python test script.

## How data.js was made

```bash
cd NeurWIN/simulator
../../.venv/bin/python export_data.py
```

`export_data.py` reads, and does not modify:

- every checkpoint in `trainResults/neurwin/deadline_env/` (episodes 0, 10, ..., 2000), loaded
  with the authors' `fcnn` class and evaluated on all 130 states;
- the closed-form index, using the body of `selectDeadlineIndex()` from the test script
  (beta = 0.99, c = 0.5);
- `replication/out/curve_N4.csv`, `curve_N4_index.csv` and `scale.csv` for the charts and table.

It writes `data.js` (`window.SIM_DATA = {...}`; a `.js` file so the page works without a
server) and `trace.json`: 3 runs at N = 4, M = 1 made with the authors' env class and the test
script's seeding (SEED = 30), with each spot's random draws and the reward of every round
under both policies. The exporter also checks those 3 runs against the saved replication
results.

## Tests

```bash
node test_sim.mjs
```

1. Replays the Python draws in `trace.json` through `sim.js` and checks the states, the
   choice and the reward of every round against Python (limit 1e-9) for both policies.
2. Runs 500 seeds at N = 4, M = 1, checkpoint 2000 with the JS generator and checks the
   means against the replication (index about -105, NeurWIN about -99, difference about +5.8).

Result on 2026-11-15: all 1,800 rounds matched exactly; 500 seeds gave index -106.64,
NeurWIN -100.61, paired difference +6.03 (standard error 0.10).

## Files

| File | What it is |
|---|---|
| `index.html`, `style.css` | the page |
| `sim.js` | environment port and policies (part 1), page logic (part 2) |
| `data.js` | generated: index tables and chart data |
| `trace.json` | generated: Python reference runs for the test |
| `export_data.py` | makes `data.js` and `trace.json` |
| `test_sim.mjs` | the tests |
