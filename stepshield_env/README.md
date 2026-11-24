# StepShield monitoring game

Rules: [env_spec.md](env_spec.md). Python 3.9+, standard library only.
Expects the data in `~/orca/stepshield/data/train`.

Run from this folder:

```
python3 -m unittest discover -s tests -v   # tests
python3 run.py                             # 100 games per policy
```

`run.py` prints the results and saves them to `results/stage4a_static_guard.md`.
`results/stage3_baselines.md` is the earlier run with only the first three policies.
StaticGuard is loaded from `~/orca/stepshield/benchmark/detectors`; LLMJudge is never
imported. Scores are cached in `cache/` (untracked, safe to delete).
Options: `--data`, `--games`, `-N`, `-K`, `--exclude-task-mismatch` (drops the 50
pairs whose task text differs between twins), and `--out`. Use `--out` for variants
so they don't overwrite the main results.

The oracle reads hidden labels. It is a greedy upper bound, not proven optimal.

## Stage 4b: LLM judge (`run_llm_judge.py`)

StepShield's LLM judge as a policy: each turn, check the agent the judge finds most
suspicious (flagged first, then higher risk; ties go to the agent checked longest ago,
so a judge with no signal plays exactly like round_robin). It uses StepShield's own
prompts, read from `~/orca/stepshield/benchmark/detectors/llm_judge.py` without
importing it. Standard library only; nothing to install.

Default judge: Anthropic `claude-haiku-5-5` (effort `low`, `max_tokens` 1024, no
temperature, since Haiku 5.5 rejects it). `--provider openai` uses the paper's
`gpt-4.1-mini` (temperature 0, `max_tokens` 200).

```
python3 run_llm_judge.py --dry-run          # no key, no calls: counts, cost estimate, sample prompt
export ANTHROPIC_API_KEY=...                # in your terminal only; never saved
python3 run_llm_judge.py --limit 20         # smoke test: 20 calls
python3 run_llm_judge.py                    # full run, stops at --max-cost (default $5)
```

- Scores are cached in `cache/llm_judge.json` (untracked). Rerunning resumes; games are
  played only once every step in play is scored. Results go to
  `results/stage4b_llm_judge.md` (all policies, paired differences, judge flag rates).
- The current step's result is hidden: the prompt's Observation line says the step
  has not run yet. `--show-result` is a sensitivity check that shows it.
- API failures are retried and reported, never cached as "safe". A 401/403/400 stops
  the run at once. Malformed API responses are retried too. A reply cut off at the
  token limit gets one retry with twice the room; if it is still cut off, it is
  reported and not cached. Refusals and unparsable replies are cached with a status,
  count as no signal, and are reported.
- Replies are parsed strictly: `is_rogue` must be true/false (a quoted "false" is
  read as false, not as rogue) and `confidence` a number in [0, 1].
- `--max-cost` caps total spend on this model, counting earlier runs saved in the
  cache. Calls already in flight when it's hit still finish (at most `--workers`).
  For a model without a built-in price, pass `--price-in` and `--price-out`, or the
  run refuses to start.
- Scores are saved every 20 replies and on Ctrl-C or a crash.
- Other options: `--model`, `--effort`, `--max-tokens`, `--workers`, `--price-in`,
  `--price-out`, plus the same game options as `run.py`.
