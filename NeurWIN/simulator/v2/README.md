# EV charging explainer (v2)

One-page research note with interactive figures: the charging problem, Algorithm 1 step by
step (real traced values in the paper's notation), a side-by-side simulation (exact Whittle index vs NeurWIN on the same cars),
results and the link to AI-agent monitoring.

- `ev-charging-explainer.html`: the finished page. Open it directly (double-click) or serve
  the folder with `python3 -m http.server 8001`. Works offline; fonts fall back to system fonts.
- `engine.js`: line-by-line JS port of `envs/deadlineSchedulingEnv.py` + the top-M index policy.
- `export_data.py`: builds `data.json` (score tables for every checkpoint, closed-form index,
  training curve, scaling table, Algorithm 1 trace of mini-batches 1, 2, 50, 200, 400 from
  `replication/algorithm1.py`) and `data_trace.json` (3 reference runs from the authors' env).
  Run from this folder: `python export_data.py ../.. data.json` (path = the NeurWIN folder).
- `test_engine.mjs`: `node test_engine.mjs .` replays the Python reference runs (must match
  to ~1e-15 with identical choices) and runs 500 seeds as a sanity check.
- `page.html` + `build.py`: page template; `python3 build.py` inlines engine and data into
  `ev-charging-explainer.html`.
