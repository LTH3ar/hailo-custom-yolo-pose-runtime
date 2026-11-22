"""Draw the two thesis charts (scb_coverage.png, clips_rules_vs_learned.png) from the Handoff 02 CSVs.

usage: python thesis/tools/draw_charts.py thesis/tools/chart_data thesis/en/figures   (then copy to thesis/vn/figures)
The CSVs hold numbers only (INT8 HEFs on the Pi, thesis v1.1); no frames.
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "Inter" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 11, "axes.edgecolor": "#b9b8b2"})
INK, INK2, SURF, BLUE = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
GREYS = ["#dfdeda", "#cfcec8", "#c1c0b6"]
H = sys.argv[1]; OUT = sys.argv[2]

def title(fig, t, s):
    fig.text(0.03, 0.965, t, fontsize=15, fontweight="semibold", color=INK, va="top")
    fig.text(0.03, 0.905, s, fontsize=11, color=INK2, va="top")

def bars(ax, xs, vals, colors, bold=None):
    b = ax.bar(xs, vals, width=0.8, color=colors, edgecolor=SURF, linewidth=2)
    for x, v, i in zip(xs, vals, range(len(vals))):
        ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", color=INK,
                fontweight="semibold" if bold and bold[i] else "normal")

def clean(ax, top):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_yticks([]); ax.set_ylim(0, top); ax.tick_params(axis="x", length=0, colors=INK2)
    ax.set_facecolor(SURF)

# 1. coverage by third, n
rows = {(r["model"], r["group"]): r for r in csv.DictReader(open(f"{H}/scb_coverage_by_third.csv"))}
fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURF)
fig.subplots_adjust(top=0.72, bottom=0.17, left=0.03, right=0.97)
title(fig, "With tiles, far students are found nearly as often as near ones",
      "Labelled SCB front-view students matched to a detection, by size in the image; upper_body_n INT8 on the Hailo-8, threshold 0.3")
groups = [("small (far)", "Far third\nsmallest in the image"), ("medium", "Middle third"), ("large (near)", "Near third\nlargest in the image")]
for i, (g, lab) in enumerate(groups):
    r = rows[("n", g)]
    bars(ax, [i * 3, i * 3 + 1], [float(r["none_pct"]), float(r["tiles_pct"])], [GREYS[2], BLUE], [False, True])
ax.set_xticks([0.5, 3.5, 6.5]); ax.set_xticklabels([l for _, l in groups], color=INK, fontsize=12)
clean(ax, 95); ax.set_xlim(-0.8, 7.8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GREYS[2], label="No tiles"), Patch(color=BLUE, label="Tiles")], loc="upper right",
          bbox_to_anchor=(1.0, 1.2), ncol=2, frameon=False, fontsize=12, handlelength=1, handleheight=1)
fig.savefig(f"{OUT}/scb_coverage.png", dpi=200, facecolor=SURF)

# 2. lecture + exam, rules vs thesis classifiers, n and m with tiles
v = {(r["clip"], r["model"], r["tiles"], r["decision"]): float(r["class_pct"]) for r in csv.DictReader(open(f"{H}/lecture_exam_class_pct.csv"))}
dec = [("rules", "Rules"), ("thesis_trees", "Trees"), ("thesis_mlp", "MLP"), ("thesis_logistic", "Logistic")]
fig, axes = plt.subplots(2, 1, figsize=(10, 8.6), facecolor=SURF)
fig.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.98, hspace=0.42)
title(fig, "The learned models call half of an attentive lecture distracted; the rules 1-2%",
      "Mean class distracted %, INT8 models on the Pi, with tiles. Blind human labels found about 4% of moments distracted.")
for ax, m in zip(axes, ("n", "m")):
    xs, vals, cols, labs = [], [], [], []
    for gi, clip in enumerate(("lecture", "exam")):
        for di, (k, lab) in enumerate(dec):
            xs.append(gi * 5 + di); vals.append(v[(clip, m, "on", k)])
            cols.append(BLUE if k == "rules" else GREYS[di - 1]); labs.append(lab)
    bars(ax, xs, vals, cols, [c == BLUE for c in cols])
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=11)
    clean(ax, 100); ax.set_xlim(-0.8, 8.8)
    ax.text(-0.08, 0.5, f"upper_body_{m}", transform=ax.transAxes, rotation=90, va="center", ha="center",
            fontsize=12, fontweight="semibold", color=INK)
    for x, t in ((1.5, "Lecture, mostly attentive"), (6.5, "Exam hall, all writing")):
        ax.text(x, -0.13, t, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12,
                fontweight="semibold", color=INK)
fig.savefig(f"{OUT}/clips_rules_vs_learned.png", dpi=200, facecolor=SURF)
