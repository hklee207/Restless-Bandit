# StepShield monitoring game

Rules: [env_spec.md](env_spec.md). Python 3.9+, standard library only.
Expects the data in `~/orca/stepshield/data/train`.

Run from this folder:

```
python3 -m unittest discover -s tests -v   # tests
python3 run.py                             # 100 games per policy
```

`run.py` prints the results table and saves it to `results/stage3_baselines.md`.
Options: `--data`, `--games`, `-N`, `-K`, `--exclude-task-mismatch` (drops the 50
pairs whose task text differs between twins), and `--out`. Use `--out` for variants
so they don't overwrite the main results.

The oracle reads hidden labels. It is a greedy upper bound, not proven optimal.
