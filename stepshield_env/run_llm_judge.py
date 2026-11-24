"""Stage 4b: StepShield's LLM judge as a policy, on the same 100 games.

Steps (run from this folder):
  1. python3 run_llm_judge.py --dry-run              # no key, no calls: counts + cost estimate
  2. export ANTHROPIC_API_KEY=...                    # only in your terminal
     python3 run_llm_judge.py --limit 20             # smoke test: 20 calls
  3. python3 run_llm_judge.py                        # full run (stops at --max-cost)
Rerunning resumes from the cache, so an interrupted run loses nothing.
"""
import argparse
import math
import statistics
import subprocess
import sys
from pathlib import Path

from data import DEFAULT_DATA, load_runs, pair_runs
from env import CURRENT_FIELDS, TASK_FIELDS, Game, sample_game
from llm_judge import (CACHE, PRICES, PROVIDERS, LLMJudgePolicy, LLMJudgeScorer,
                       make_transport)
from policies import OraclePolicy, RandomPolicy, RoundRobinPolicy, StaticGuardPolicy
from run import play

RESULTS = Path(__file__).resolve().parent / "results" / "stage4b_llm_judge.md"
# Rough output tokens per call for the dry-run estimate (Haiku 5.5 may think a little).
EST_OUT = {"anthropic": 150, "openai": 60}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=sorted(PROVIDERS), default="anthropic")
    p.add_argument("--model", help="default: claude-haiku-5-5 / gpt-4.1-mini")
    p.add_argument("--effort", default="low", help="Anthropic only (low/medium/high)")
    p.add_argument("--max-tokens", type=int,
                   help="default 1024 for Anthropic (thinking counts), 200 for OpenAI")
    p.add_argument("--workers", type=int, default=8, help="parallel API calls")
    p.add_argument("--max-cost", type=float, default=5.0,
                   help="USD cap on TOTAL spend for this model, including earlier runs")
    p.add_argument("--price-in", type=float, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, help="USD per 1M output tokens")
    p.add_argument("--limit", type=int, help="score at most this many new steps")
    p.add_argument("--dry-run", action="store_true", help="no API calls, no key needed")
    p.add_argument("--show-result", action="store_true",
                   help="sensitivity check: judge also sees the current step's result")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("-N", type=int, default=10)
    p.add_argument("-K", type=int, default=2)
    p.add_argument("--exclude-task-mismatch", action="store_true")
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=RESULTS)
    args = p.parse_args(argv)
    args.model = args.model or PROVIDERS[args.provider]["model"]
    if args.max_tokens is None:
        args.max_tokens = 1024 if args.provider == "anthropic" else 200
    default_price = PRICES.get(args.model, (math.nan, math.nan))
    args.price_in = default_price[0] if args.price_in is None else args.price_in
    args.price_out = default_price[1] if args.price_out is None else args.price_out
    return args


def settings_for(args):
    if args.provider == "anthropic":
        return {"max_tokens": args.max_tokens, "effort": args.effort}
    return {"max_tokens": args.max_tokens, "temperature": 0.0}


def step_prompts(scorer, runs, show_result):
    """Every step of every run in play, as the judge would see it (upper bound)."""
    fields = CURRENT_FIELDS + (("observation",) if show_result else ())
    for run in runs:
        task = {f: run.task[f] for f in TASK_FIELDS}
        for t, step in enumerate(run.steps, 1):
            yield scorer.prompt(task, {f: step[f] for f in fields}, t)


def per_game(games, make_policy, show_result=False):
    out = []
    for seed, runs in games:
        game = Game(runs, show_result)
        out.append(play(game, make_policy(game, seed)))
    return out


def row(name, summaries):
    rewards = [s["total_reward"] for s in summaries]
    delays = [d for s in summaries for d in s["delays"]]
    caught = sum(s["caught"] for s in summaries)
    total = caught + sum(s["missed"] for s in summaries)
    se = statistics.stdev(rewards) / math.sqrt(len(rewards)) if len(rewards) > 1 else math.nan
    return (f"| {name} | {statistics.mean(rewards):.2f} | {se:.2f} | "
            f"{100 * caught / total if total else math.nan:.1f}% | "
            f"{statistics.mean(delays) if delays else math.nan:.2f} |")


def paired(a, b):
    diffs = [x["total_reward"] - y["total_reward"] for x, y in zip(a, b)]
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else math.nan
    return statistics.mean(diffs), se


def judge_flag_rates(scorer, runs, show_result):
    fields = CURRENT_FIELDS + (("observation",) if show_result else ())
    groups = {g: [0, 0] for g in ("clean", "rogue, before tau", "rogue, at tau",
                                  "rogue, after tau")}
    status = {}
    for run in runs:
        task = {f: run.task[f] for f in TASK_FIELDS}
        for t, step in enumerate(run.steps, 1):
            e = scorer.lookup(task, {f: step[f] for f in fields}, t)
            status[e["status"]] = status.get(e["status"], 0) + 1
            flagged = e["status"] == "ok" and bool(e["is_rogue"])
            g = ("clean" if not run.rogue else "rogue, before tau" if t < run.tau
                 else "rogue, at tau" if t == run.tau else "rogue, after tau")
            groups[g][0] += flagged
            groups[g][1] += 1
    return groups, status


def main(argv=None, transport=None, log=print):
    args = parse_args(argv)
    runs = load_runs(args.data)
    pairs = pair_runs(runs, args.exclude_task_mismatch)
    games = [(s, sample_game(pairs, args.N, args.K, s)) for s in range(args.games)]
    in_play = list({id(r): r for _, rs in games for r in rs}.values())

    scorer = LLMJudgeScorer(args.provider, args.model, settings_for(args), args.cache,
                            show_result=args.show_result)
    prompts = list(step_prompts(scorer, in_play, args.show_result))
    todo = scorer.missing(prompts)
    unique = len({scorer.key(p) for p in prompts})
    log(f"{args.provider}/{args.model} {settings_for(args)} | runs in play: {len(in_play)}"
        f" | steps: {len(prompts)} | unique judge calls: {unique} | still to score: {len(todo)}")

    if args.dry_run:
        if math.isnan(args.price_in) or math.isnan(args.price_out):
            log(f"Note: no known price for {args.model!r}; pass --price-in/--price-out.")
        chars = sum(len(p) for p in todo) + len(todo) * len(scorer.system)
        factor = 3.3 if args.provider == "anthropic" else 4.0  # rough chars per token
        tin, tout = chars / factor, len(todo) * EST_OUT[args.provider]
        cost = (tin * args.price_in + tout * args.price_out) / 1e6
        log(f"Estimate for {len(todo)} calls: ~{tin / 1e6:.2f}M input + "
            f"~{tout / 1e6:.2f}M output tokens = ~${cost:.2f} "
            f"(prices ${args.price_in}/${args.price_out} per 1M; check them).")
        sample = next(p for p in prompts)
        log("\n--- Example prompt the judge would get (step 1 of the first run) ---\n"
            + sample + "\n--- end ---")
        return 0

    if todo:
        if math.isnan(args.price_in) or math.isnan(args.price_out):
            log(f"No known price for model {args.model!r}, so the spend cap can't work. "
                "Pass --price-in and --price-out (USD per 1M tokens) and rerun.")
            return 2
        if transport is None:
            transport = make_transport(args.provider, args.model,
                                       **dict(settings_for(args), timeout=60))
        batch = todo[:args.limit] if args.limit else todo
        log(f"Scoring {len(batch)} steps with {args.workers} workers, "
            f"spend cap ${args.max_cost:.2f} ...")
        rep = scorer.fetch(batch, transport, workers=args.workers, max_cost=args.max_cost,
                           prices=(args.price_in, args.price_out), log=log)
        log(f"Done: {rep['done']} scored, {rep['failed']} failed, {rep['not_run']} not run, "
            f"refusals {rep.get('refusal', 0)}, parse errors {rep.get('parse_error', 0)}, "
            f"cut-off replies retried {rep['truncated_retried']}, "
            f"tokens {rep['in_tok']} in / {rep['out_tok']} out, ~${rep['cost']:.3f} this run "
            f"(~${rep['spent_before'] + rep['cost']:.3f} total for this model)")
        for e in rep["errors"]:
            log(f"  error: {e}")
        todo = scorer.missing(prompts)
    if todo:
        log(f"{len(todo)} steps still unscored. Rerun the same command to continue "
            "(the cache keeps what is done). Games are played only when all are scored.")
        return 1

    from static_guard import StaticGuardScorer
    sg = StaticGuardScorer()
    judge = "llm_judge_with_result (sensitivity check)" if args.show_result else "llm_judge"
    policies = [
        ("random", lambda g, s: RandomPolicy(f"random-policy-{s}"), False),
        ("round_robin", lambda g, s: RoundRobinPolicy(), False),
        ("static_guard", lambda g, s: StaticGuardPolicy(sg), False),
        (judge, lambda g, s: LLMJudgePolicy(scorer), args.show_result),
        ("oracle", lambda g, s: OraclePolicy(g), False),
    ]
    results = {name: per_game(games, make, show) for name, make, show in policies}
    sg.save()
    groups, status = judge_flag_rates(scorer, in_play, args.show_result)
    used = [scorer.lookup({f: r.task[f] for f in TASK_FIELDS},
                          {f: s[f] for f in CURRENT_FIELDS + (("observation",)
                                                              if args.show_result else ())},
                          t)
            for r in in_play for t, s in enumerate(r.steps, 1)]
    seen, tin, tout = set(), 0, 0
    for e in used:
        if id(e) not in seen:
            seen.add(id(e))
            tin += e.get("in_tok", 0)
            tout += e.get("out_tok", 0)
    cost = (tin * args.price_in + tout * args.price_out) / 1e6
    try:
        commit = subprocess.run(["git", "-C", str(args.data), "rev-parse", "--short", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    lines = [
        "# Stage 4b: LLM judge", "",
        f"- Judge: {args.provider} / `{args.model}`, settings {settings_for(args)}; "
        "StepShield's own prompts; current step's result "
        + ("SHOWN (sensitivity check)" if args.show_result else "hidden"),
        f"- Judge calls behind these results: {len(seen)} "
        f"({tin} input / {tout} output tokens, ~${cost:.2f} at "
        f"${args.price_in}/${args.price_out} per 1M)",
        f"- Reply status over all steps in play: {status}",
        f"- stepshield data commit: `{commit}`; pairs used: {len(pairs)}; "
        f"games: {args.games} (seeds 0-{args.games - 1}), N={args.N}, K={args.K}; "
        "same games for every policy", "",
        "| policy | mean total reward | standard error | % rogue caught | mean catch delay |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [row(name, s) for name, s in results.items()]
    lines += ["", "Paired differences (same games):", "",
              "| comparison | difference | standard error |", "|---|---:|---:|"]
    for other in ("round_robin", "static_guard", "oracle"):
        d, se = paired(results[judge], results[other])
        lines.append(f"| {judge} - {other} | {d:+.2f} | {se:.2f} |")
    lines += ["", "Judge flag rates (is_rogue = true) over all steps of the runs in play:", "",
              "| steps | flagged |", "|---|---:|"]
    for g, (f, n) in groups.items():
        lines.append(f"| {g} | {f}/{n} ({100 * f / n if n else math.nan:.1f}%) |")
    text = "\n".join(lines) + "\n"
    log(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
