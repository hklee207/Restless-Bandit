"""Plots for the meeting. Usage: python plot.py <out dir with curve_N4.csv, curve_N4_index.csv, scale.csv>"""
import sys, os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
SURF, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})

# ---- Figure 1: replication of Fig. 2 (N=4, M=1)
c = pd.read_csv(os.path.join(D, "curve_N4.csv"))
idx = pd.read_csv(os.path.join(D, "curve_N4_index.csv")).reward.mean()
g = c.groupby("episode").reward
m, lo, hi = g.mean(), g.quantile(0.05), g.quantile(0.95)
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.fill_between(m.index, lo, hi, color=BLUE, alpha=0.15, linewidth=0)
ax.plot(m.index, m.values, color=BLUE, lw=2, label="NeurWIN (mean of 50 test runs, 5th–95th pct band)")
ax.axhline(idx, color=ORANGE, lw=2, ls="--", label="Closed-form deadline Whittle index")
ax.set_xlim(0, 2000); ax.set_ylim(-350, -50)
ax.set_xlabel("Training episodes"); ax.set_ylabel("Total discounted reward")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.text(2000, m.iloc[-1] + 6, f"NeurWIN {m.iloc[-1]:.1f}", ha="right", va="bottom", color=INK, fontsize=10)
ax.text(2000, idx - 6, f"Index {idx:.1f}", ha="right", va="top", color=INK, fontsize=10)
cross = m[m >= idx].index.min()
ax.annotate(f"crosses index at ~{cross} episodes", xy=(cross, idx), xytext=(cross + 150, -260),
            color=INK2, fontsize=10, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_title("Deadline scheduling (EV charging), N = 4 cars, M = 1 charger", loc="left", color=INK, fontsize=12)
ax.legend(loc="lower right", bbox_to_anchor=(1, 0.08), frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout(); fig.savefig(os.path.join(D, "fig2_N4_replication.png"), dpi=200); plt.close(fig)

# ---- Figure 2: does the NeurWIN edge survive as N grows? (M = N/4)
s = pd.read_csv(os.path.join(D, "scale.csv"))
fig, ax = plt.subplots(figsize=(7, 3.8))
labels = [f"N={n}, M={mm}" for n, mm in zip(s.N, s.M)]
cols = [BLUE if v > 0 else ORANGE for v in s.gap_pct]
bars = ax.bar(labels, s.gap_pct, color=cols, width=0.55, edgecolor=SURF, linewidth=2)
ax.axhline(0, color=INK2, lw=1)
for b, v, se, N in zip(bars, s.gap_pct, s.paired_se, s.N):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.25 if v > 0 else -0.25), f"{v:+.1f}%",
            ha="center", va="bottom" if v > 0 else "top", color=INK, fontsize=10)
ax.set_ylabel("NeurWIN vs index (% of index reward)")
ax.set_ylim(-3.5, 7)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.text(0.99, 0.95, "above 0: NeurWIN better\nbelow 0: closed-form index better", transform=ax.transAxes,
        ha="right", va="top", fontsize=9, color=INK2)
ax.set_title("NeurWIN's edge shrinks, then flips, as the system grows (same trained net)", loc="left", color=INK, fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(D, "scaling_gap.png"), dpi=200); plt.close(fig)
print("ok", cross)
