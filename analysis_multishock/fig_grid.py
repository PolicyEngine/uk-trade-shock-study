"""Heatmap for the two-axis grid (Table tab:grid): cushioning rate by
rebase factor x discretionary stack, annotated with rate and D1/D10.
House style follows make_figures.py."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from make_figures import apply_style, BLUE, INK  # noqa: E402

cells = json.load(open("out/grid_energy.json"))["cells"]
stacks = ["none", "epg", "epg_ebss", "full"]
stack_lab = ["None", "EPG only", "EPG + EBSS", "Full stack"]
rebases = sorted({c["rebase"] for c in cells})
reb_lab = [f"{r:.3f}\n(paper)" if abs(r - min(rebases)) < 1e-9 else f"{r:.1f}"
           for r in rebases]

M = np.zeros((len(stacks), len(rebases)))
A = [[None] * len(rebases) for _ in stacks]
for c in cells:
    i, j = stacks.index(c["stack"]), rebases.index(c["rebase"])
    M[i, j] = c["rate_pct"]
    A[i][j] = c

apply_style()
fig, ax = plt.subplots(figsize=(7.2, 4.4))
fig.subplots_adjust(left=0.16, right=0.88, top=0.90, bottom=0.12)
im = ax.imshow(M, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "pe", ["white", BLUE]), vmin=0, vmax=100, aspect="auto")
for i in range(len(stacks)):
    for j in range(len(rebases)):
        c = A[i][j]
        col = "white" if M[i, j] > 55 else INK
        ax.text(j, i - 0.13, f"{c['rate_pct']:.1f}%", ha="center",
                va="center", fontsize=13, fontweight="bold", color=col)
        ax.text(j, i + 0.22,
                f"D1 {c['cushion_pct_d1']:.0f} / D10 {c['cushion_pct_d10']:.0f}",
                ha="center", va="center", fontsize=8.5, color=col)
ax.set_xticks(range(len(rebases)), reb_lab)
ax.set_yticks(range(len(stacks)), stack_lab)
ax.set_xlabel("Rebase factor")
ax.set_ylabel("Discretionary stack")
ax.set_title("Discretionary cushioning rate across the two-axis grid",
             color=INK)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Rate, % of counterfactual shock")
for spine in ax.spines.values():
    spine.set_visible(False)
fig.savefig("../paper_multishock/figures/fig_grid.png", dpi=300)
print("written")
