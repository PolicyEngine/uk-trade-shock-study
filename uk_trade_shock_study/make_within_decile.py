"""Within-decile dispersion of energy exposure: the figure the decile
tables cannot produce.  Style via the shared PolicyEngine figstyle."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import apply_style, save, decile_ax, WIDTH, BLUE, GREEN

HERE = Path(__file__).resolve().parent

r = json.loads((HERE.parent / "results" / "second_stage_energy.json").read_text())
dec = [x["decile"] for x in r["by_decile"]]
mean = [x["burden_pct_of_spending"] for x in r["by_decile"]]
p10 = [x["p10_burden_pct_spending"] for x in r["within_decile_dispersion"]]
p90 = [x["p90_burden_pct_spending"] for x in r["within_decile_dispersion"]]

apply_style()
fig, ax = plt.subplots(figsize=(WIDTH, 4.2))

ax.bar(dec, mean, width=0.62, color=BLUE, zorder=2,
       label="Decile mean burden")
for d, lo, hi in zip(dec, p10, p90):
    ax.plot([d, d], [lo, hi], color=GREEN, lw=2.0, solid_capstyle="butt",
            zorder=3)
    for y in (lo, hi):
        ax.plot([d - 0.16, d + 0.16], [y, y], color=GREEN, lw=1.6, zorder=3)
ax.plot([], [], color=GREEN, lw=2.0,
        label="p10\u2013p90 spread within the decile")

decile_ax(ax, "Realised energy burden,\n% of household expenditure")
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18),
          frameon=False)
save(fig, "fig_within_decile.png")
print("wrote fig_within_decile.png")
