"""Heatmap for the two-axis grid: discretionary cushioning rate by
rebase factor (fine sweep, 0.55-1.05 in 0.05 steps) x discretionary
stack.  The table keeps the three canonical rebase columns; this figure
uses the 44-cell fine sweep.  House style via make_figures/figstyle."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from make_figures import apply_style, save, WIDTH, BLUE, INK  # noqa: E402

data = json.load(open("out/grid_energy.json"))
cells = data["cells_fine"]
paper_rb = data["meta"]["paper_rebase"]
stacks = ["epg", "epg_ebss", "col_mt", "col_pen", "col_dis"]
stack_lab = ["EPG only", "+ Bills Support (\u00a3400)",
             "+ CoL means-tested (\u00a3650)", "+ CoL pensioner (\u00a3300)",
             "+ CoL disability (\u00a3150)\n= full modelled stack"]
rebases = sorted({c["rebase"] for c in cells})

M = np.zeros((len(stacks), len(rebases)))
for c in cells:
    if c["stack"] in stacks:
        M[stacks.index(c["stack"]), rebases.index(c["rebase"])] = c["rate_pct"]

apply_style()
fig, ax = plt.subplots(figsize=(WIDTH, 4.6))
ax.grid(False)
im = ax.imshow(M, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "pe", ["white", BLUE]), vmin=0, vmax=100, aspect="auto")
for i in range(len(stacks)):
    for j in range(len(rebases)):
        col = "white" if M[i, j] > 55 else INK
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                fontsize=8, color=col)
# Mark the paper's computed rebase with a dashed rule between columns.
jp = np.interp(paper_rb, rebases, range(len(rebases)))
ax.axvline(jp, color=INK, lw=1.0, ls=(0, (4, 2)))
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
import shutil
shutil.copy("fig_grid.png", "../paper_multishock/figures/fig_grid.png")
print("written")
