"""Play the baseline policies on the same games and save a results table."""
import argparse
import math
import statistics
import subprocess
from pathlib import Path

from data import DEFAULT_DATA, load_runs, pair_runs
from env import Game, sample_game
from policies import OraclePolicy, RandomPolicy, RoundRobinPolicy

RESULTS = Path(__file__).resolve().parent / "results" / "stage3_baselines.md"

# Fresh policy per game. The random policy's seed string keeps its random
# stream separate from the one that sampled the game.
POLICIES = [
    ("random", lambda game, seed: RandomPolicy(f"random-policy-{seed}")),
    ("round_robin", lambda game, seed: RoundRobinPolicy()),
    ("oracle", lambda game, seed: OraclePolicy(game)),
]


def play(game, policy):
    while not game.done:
        game.step(policy.choose(game.observe()))
    return game.summary()


def evaluate(games, make_policy):
    out = []
    for seed, runs in games:
        game = Game(runs)
        out.append(play(game, make_policy(game, seed)))
    rewards = [s["total_reward"] for s in out]
    delays = [d for s in out for d in s["delays"]]
    caught = sum(s["caught"] for s in out)
    total = caught + sum(s["missed"] for s in out)
    se = statistics.stdev(rewards) / math.sqrt(len(rewards)) if len(rewards) > 1 else math.nan
    return {"reward": statistics.mean(rewards), "se": se,
            "caught_pct": 100 * caught / total if total else math.nan,
            "delay": statistics.mean(delays) if delays else math.nan}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--games", type=int, default=100, help="seeds 0..games-1")
    p.add_argument("-N", type=int, default=10)
    p.add_argument("-K", type=int, default=2)
    p.add_argument("--exclude-task-mismatch", action="store_true")
    p.add_argument("--out", type=Path, default=RESULTS)
    args = p.parse_args()

    runs = load_runs(args.data)
    all_pairs = pair_runs(runs)
    pairs = pair_runs(runs, args.exclude_task_mismatch)
    games = [(s, sample_game(pairs, args.N, args.K, s)) for s in range(args.games)]
    commit = subprocess.run(["git", "-C", str(args.data), "rev-parse", "--short", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()

    lines = [
        "# Stage 3 baselines", "",
        f"- stepshield data commit: `{commit}`",
        f"- Runs loaded: {len(runs)}; pairs used: {len(pairs)} of {len(all_pairs)} "
        f"(exclude_task_mismatch={args.exclude_task_mismatch})",
        f"- Games: {args.games} (seeds 0-{args.games - 1}), N={args.N}, K={args.K}; "
        "same games for every policy",
        "- % rogue caught is pooled over all rogue agents; "
        "mean catch delay covers caught agents only", "",
        "| policy | mean total reward | standard error | % rogue caught | mean catch delay |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, make in POLICIES:
        r = evaluate(games, make)
        lines.append(f"| {name} | {r['reward']:.2f} | {r['se']:.2f} | "
                     f"{r['caught_pct']:.1f}% | {r['delay']:.2f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
