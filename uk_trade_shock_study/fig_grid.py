"""Heatmap for the two-axis grid: discretionary cushioning rate by
rebase factor (fine sweep, 0.55-1.05 in 0.05 steps) x discretionary
stack.  The table prints four rebases (one below the paper's); this figure
uses the 66-cell fine sweep.  House style via make_figures/figstyle."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from make_figures import apply_style, save, WIDTH, BLUE, INK  # noqa: E402
from second_stage_energy import RESULTS_DIR

data = json.load(open(RESULTS_DIR / "grid_energy.json"))
cells = data["cells_fine"]
paper_rb = data["meta"]["paper_rebase"]
stacks = ["epg", "epg_ebss", "ctr", "col_mt", "col_pen", "col_dis"]
stack_lab = ["Energy Price Guarantee only", "+ Bills Support (\u00a3400)",
             "+ Council Tax rebate (\u00a3150)\n= energy-specific stack",
             "+ Cost of Living means-tested (\u00a3650)", "+ Cost of Living pensioner (\u00a3300)",
             "+ Cost of Living disability (\u00a3150)\n= full stack"]
rebases = sorted({c["rebase"] for c in cells})

M = np.zeros((len(stacks), len(rebases)))
for c in cells:
    if c["stack"] in stacks:
        M[stacks.index(c["stack"]), rebases.index(c["rebase"])] = c["rate_pct"]

apply_style()
fig, ax = plt.subplots(figsize=(WIDTH, 5.2))
ax.grid(False)
im = ax.imshow(M, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "pe", ["white", BLUE]), vmin=0, vmax=100, aspect="auto")
for i in range(len(stacks)):
    for j in range(len(rebases)):
        col = "white" if M[i, j] > 55 else INK
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                fontsize=8, color=col)
# Outline the column nearest the paper's computed rebase (a rule through the
# column would sit on top of its numbers).
jp = int(np.argmin([abs(r - paper_rb) for r in rebases]))
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((jp - 0.5, -0.5), 1, len(stacks), fill=False,
                       edgecolor=INK, lw=1.4, ls=(0, (4, 2)), zorder=4))
ax.annotate(f"paper ({paper_rb:.3f})", xy=(jp, -0.5), xytext=(jp, -0.62),
            ha="center", va="bottom", fontsize=8, color=INK,
            annotation_clip=False)
ax.set_xticks(range(len(rebases)), [f"{r:.2f}" for r in rebases])
ax.set_yticks(range(len(stacks)), stack_lab)
ax.set_xlabel("Rebase factor")
ax.set_title("")
ax.set_ylabel("Discretionary stack")
for spine in ax.spines.values():
    spine.set_visible(False)
save(fig, "fig_grid.png")
print("written")
