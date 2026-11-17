#!/usr/bin/env python3
"""Intermediate layout: 60x30 ice rink with full IIHF markings + current anchors."""
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge

OUT = os.path.dirname(os.path.abspath(__file__)) + "/match"
W, L = 30.0, 60.0  # width, length

fig, ax = plt.subplots(figsize=(16, 8.8), dpi=110)
ax.add_patch(Rectangle((0, 0), L, W, fc="#f7fbff", ec="#333", lw=2))
# rounded corners are not needed visually — the grid matters more

# lines
ax.plot([30, 30], [0, W], color="#d11", lw=3)  # red center line
for x in (22.4, 37.6):
    ax.plot([x, x], [0, W], color="#15c", lw=3)  # blue lines
for x in (4.0, 56.0):
    ax.plot([x, x], [0, W], color="#d11", lw=2)  # goal lines
    ax.plot([x, x], [11.44, 18.56], color="#1a6", lw=6)  # goal mouth between the posts

# goal trapezoids (IIHF: 6.8° from the post, 11.6 m base at the boards)
for gx, sgn in ((4.0, 1), (56.0, -1)):
    for ys, ye in ((0, W),):
        pass
    # vertices: at the boards x = gx ± 3.4 (near side), x = gx ± 2.84 (far side)
    ax.plot([gx + sgn * 3.4, gx - sgn * 2.84], [0.0, 1.86], color="#d11", lw=1.4)
    ax.plot([gx + sgn * 3.4, gx - sgn * 2.84], [W, W - 1.86], color="#d11", lw=1.4)

# center circle and face-off dot
ax.add_patch(Circle((30, 15), 4.5, fill=False, ec="#15c", lw=2))
ax.add_patch(Circle((30, 15), 0.3, fc="#15c"))
# zone face-off circles
for (cx, cy) in [(6, 7.1), (6, 22.9), (54, 7.1), (54, 22.9)]:
    ax.add_patch(Circle((cx, cy), 4.5, fill=False, ec="#d11", lw=1.6))
    ax.add_patch(Circle((cx, cy), 0.3, fc="#d11"))
    for dx, dy in [(-0.75, 0), (0.75, 0), (0, -0.6), (0, 0.6)]:
        ax.add_patch(Circle((cx + dx, cy + dy), 0.14, fc="#d11", ec="none"))
# neutral-zone face-off circles
for (cx, cy) in [(22.4, 7.1), (22.4, 22.9), (37.6, 7.1), (37.6, 22.9)]:
    ax.add_patch(Circle((cx, cy), 2.0, fill=False, ec="#d11", lw=1.2))
    ax.add_patch(Circle((cx, cy), 0.25, fc="#d11"))
# boards/walls — template lines
for y in (0, W):
    ax.plot([0, L], [y, y], color="#333", lw=3)

# grid every 5 m
for x in range(0, 61, 5):
    ax.plot([x, x], [-0.6, W + 0.6], color="#bbb", lw=0.6, zorder=0)
    ax.text(x, -1.3, str(x), ha="center", fontsize=8, color="#888")
for y in range(0, 31, 5):
    ax.plot([-0.6, L + 0.6], [y, y], color="#bbb", lw=0.6, zorder=0)
    ax.text(-1.4, y, str(y), va="center", fontsize=8, color="#888")

# calibration anchors (in world coords) — labels from points.json
P = {"c1": (30, 15), "c2": (30, 30), "c3": (30, 0),
     "l1": (22.4, 30), "r1": (37.6, 30),
     "l2": (22.4, 0), "r2": (37.6, 0)}
for k, (x, y) in P.items():
    ax.plot(x, y, "o", ms=11, mfc="none", mec="#0a0", mew=2.5)
    ax.annotate(k, (x, y), textcoords="offset points", xytext=(7, 7),
                fontsize=11, color="#070", weight="bold")

ax.set_xlim(-2.5, 62.5); ax.set_ylim(-2.5, 32.5)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Ice rink 60×30 m, IIHF markings. Green circles — 7 calibration anchors (X along the rink, 0=left end wall, y=0 near boards)", fontsize=12)
plt.tight_layout()
p = f"{OUT}/rink_layout.jpg"
plt.savefig(p, dpi=110)
print(p)
