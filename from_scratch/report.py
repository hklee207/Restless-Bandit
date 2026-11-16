"""Phase 3: plots and tables from out/results.json (written by evaluate.py)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithm1 import OUT

COL = {"shared": "#1f77b4", "independent": "#d62728"}


def stays_above(ep, mean, idx):
    for i in range(len(mean)):
        if all(x >= idx for x in mean[i:]):
            return ep[i]
    return None


def main():
    res = json.load(open(os.path.join(OUT, "results.json")))
    I = res["index"]["mean"]
    runs = res["runs"]

    # curve.png: seed 50, both variants, 5-95% band over the 50 test runs
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for v in ("shared", "independent"):
        r = runs[f"{v}_seed50"]
        ep = np.array(r["episodes"])
        ax.fill_between(ep, r["p05"], r["p95"], color=COL[v], alpha=0.15, lw=0)
        ax.plot(ep, r["mean"], color=COL[v], lw=1.8, label=f"NeurWIN {v.upper()} (mean of 50 runs)")
    ax.axhline(I, color="k", ls="--", lw=1.2, label=f"Deadline index ({I:.1f})")
    ax.set_xlabel("Training episodes"); ax.set_ylabel("Total discounted reward")
    ax.set_title("From-scratch NeurWIN, deadline scheduling, N=4, M=1 (seed 50)\nband = 5-95% over 50 test runs")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "curve.png"), dpi=150)

    # curve_seeds.png: mean curves for every seed
    fig, ax = plt.subplots(figsize=(7, 4.2))
    styles = {50: "-", 51: ":", 52: "-."}
    for key, r in sorted(runs.items()):
        ax.plot(r["episodes"], r["mean"], color=COL[r["variant"]], ls=styles.get(r["seed"], "-"),
                lw=1.4, label=f"{r['variant']} seed {r['seed']}")
    ax.axhline(I, color="k", ls="--", lw=1.2, label="Deadline index")
    ax.set_ylim(-160, -90)
    ax.set_xlabel("Training episodes"); ax.set_ylabel("Total discounted reward (mean of 50 runs)")
    ax.set_title("All seeds (y-axis clipped to -160)")
    ax.legend(fontsize=8, ncol=2, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "curve_seeds.png"), dpi=150)

    # table.md
    lines = [f"Deadline index reward (50 runs): **{I:.1f}**\n",
             "| variant | seed | first ckpt >= index | stays >= index from | reward @2000 | index | diff (paired SE) | runs won /50 | rank corr @2000 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for v in ("shared", "independent"):
        rows = [r for r in runs.values() if r["variant"] == v]
        rows.sort(key=lambda r: r["seed"])
        for r in rows:
            sa = stays_above(r["episodes"], r["mean"], I)
            lines.append(f"| {v.upper()} | {r['seed']} | {r['first_ge_index']} | {sa} | {r['final_mean']:.1f} | {I:.1f} | "
                         f"{r['final_diff']:+.1f} ({r['final_diff_se']:.1f}) | {r['runs_won']} | {r['final_rho']:.2f} |")
        fx = [r["first_ge_index"] for r in rows]; sx = [stays_above(r["episodes"], r["mean"], I) for r in rows]
        lines.append(f"| {v.upper()} | **mean** | {np.mean(fx):.0f} | {np.mean(sx):.0f} | "
                     f"{np.mean([r['final_mean'] for r in rows]):.1f} | {I:.1f} | "
                     f"{np.mean([r['final_diff'] for r in rows]):+.1f} | "
                     f"{np.mean([r['runs_won'] for r in rows]):.0f} | {np.mean([r['final_rho'] for r in rows]):.2f} |")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "table.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
