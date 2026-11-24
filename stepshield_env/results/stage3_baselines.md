# Stage 3 baselines

- stepshield data commit: `86710a9`
- Runs loaded: 1278; pairs used: 639 of 639 (exclude_task_mismatch=False)
- Games: 100 (seeds 0-99), N=10, K=2; same games for every policy
- % rogue caught is pooled over all rogue agents; mean catch delay covers caught agents only

| policy | mean total reward | standard error | % rogue caught | mean catch delay |
|---|---:|---:|---:|---:|
| random | -7.51 | 0.43 | 69.5% | 3.16 |
| round_robin | -6.27 | 0.29 | 73.0% | 2.96 |
| oracle | -0.10 | 0.03 | 100.0% | 0.05 |
