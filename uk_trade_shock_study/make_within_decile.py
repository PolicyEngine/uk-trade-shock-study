"""Within-decile dispersion of energy exposure: the figure the decile
tables cannot produce.  Style matches make_figures.py."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BLUE, GREEN, INK, INK2, MUTED = "#2C6496", "#558B2F", "#1a1a1a", "#4a4a4a", "#9a9a9a"
WIDTH = 6.3

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "figure.facecolor": "white", "axes.facecolor": "white",
})

r = json.loads((HERE.parent / "out" / "second_stage_energy.json").read_text())
dec = [x["decile"] for x in r["by_decile"]]
mean = [x["burden_pct_of_spending"] for x in r["by_decile"]]
p10 = [x["p10_burden_pct_spending"] for x in r["within_decile_dispersion"]]
p90 = [x["p90_burden_pct_spending"] for x in r["within_decile_dispersion"]]
bw = r["variance_decomposition"]["expenditure_share_trimmed_1_99"]["between_pct"]
ww = r["variance_decomposition"]["expenditure_share_trimmed_1_99"]["within_pct"]

fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
fig.subplots_adjust(left=0.145, right=0.975, top=0.885, bottom=0.30)

ax.bar(dec, mean, width=0.62, color=BLUE, alpha=0.85, zorder=2,
       label="Decile mean burden")
for d, lo, hi in zip(dec, p10, p90):
    ax.plot([d, d], [lo, hi], color=GREEN, lw=2.0, solid_capstyle="butt", zorder=3)
    for y in (lo, hi):
        ax.plot([d - 0.16, d + 0.16], [y, y], color=GREEN, lw=1.6, zorder=3)
ax.plot([], [], color=GREEN, lw=2.0,
        label="p10–p90 spread — within the same decile")

ax.set_xticks(dec)
ax.set_xlabel("Equivalised disposable income decile (1 = lowest)")
ax.set_ylabel("Realised energy burden,\n% of household expenditure")
ax.set_title("What the decile lens hides: exposure varies more\nwithin deciles than between them", fontsize=10.0, pad=7)
ax.grid(axis="y", color="#ececec", lw=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=8.2)
ax.annotate(f"Between deciles: {bw:.0f}% of variance\nWithin deciles: {ww:.0f}%",
            xy=(0.34, 0.62), xycoords="axes fraction", fontsize=8.6, color=INK2,
            linespacing=1.45)
fig.text(0.02, 0.055,
         "Bars are each decile's mean realised 2022-23 energy burden as a share of total expenditure; whiskers\n"
         "span the 10th-90th percentile within the same decile. Equivalised deciles; enhanced FRS in PolicyEngine UK;\n"
         "variance shares on the trimmed expenditure-share basis. Consumption is imputed, so this is a bound.",
         fontsize=7.4, color=INK2, linespacing=1.5)

out = HERE / "fig_within_decile.png"
fig.savefig(out, dpi=300)
w_in = fig.get_size_inches()[0]
assert abs(w_in - WIDTH) < 1e-6, w_in
print(f"wrote {out} ({w_in:.2f} in)")
