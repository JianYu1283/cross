"""Generate the paper figures (300 DPI PNG)."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle
import numpy as np

RES = "results"
FIG = os.path.join("paper", "figures")
os.makedirs(FIG, exist_ok=True)


def load(tag):
    with open(os.path.join(RES, "full_" + tag + ".json"), encoding="utf-8") as f:
        return json.load(f)


# colorblind-safe palette (Okabe-Ito inspired)
BLUE_D, BLUE_L = "#08519c", "#9ecae1"
ORG_D, ORG_L = "#a63603", "#fdae6b"
GRAY_D, GRAY_L = "#525252", "#bdbdbd"
GREEN = "#009E73"
PURPLE = "#6a51a3"


def _autofit(ax, x, y, w, h, text, fc, fs_max=9.5, fs_min=6.5):
    """Draw a box and shrink font until the longest line fits inside."""
    lines = text.split("\n")
    fs = fs_max
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                       fc=fc, ec="#333333", lw=0.9)
    ax.add_patch(p)
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color="white", linespacing=1.3)
    fig = ax.figure
    fig.canvas.draw()
    bbox = t.get_window_extent()
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
    while (x1 - x0 > w * 0.96 or y1 - y0 > h * 0.92) and fs > fs_min:
        fs -= 0.4
        t.set_fontsize(fs)
        fig.canvas.draw()
        bbox = t.get_window_extent()
        inv = ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
    t.set_fontsize(fs)


def fig0_system():
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    # Vertical pipeline: clean flat cards, single blue accent on the key stage
    INK = "#1a1a1a"
    EDGE = "#5a5a5a"
    NEUTRAL = "#f4f5f6"
    ACCENT = "#d5e8f7"
    ACCENT_EDGE = "#2c6fa8"

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.1, color=EDGE))

    def node(cx, cy, w, h, text, fill, edge=EDGE):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fill, ec=edge, lw=0.9))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=7, color=INK)

    stages = [
        ("Solidity source", NEUTRAL, EDGE),
        ("Slither parsing", NEUTRAL, EDGE),
        ("Graph construction", ACCENT, ACCENT_EDGE),
        ("Node features", NEUTRAL, EDGE),
        ("HGT classifier", NEUTRAL, EDGE),
        ("vulnerable / clean", NEUTRAL, EDGE),
    ]
    cx, w, h = 2.1, 2.5, 0.56
    gap = 0.20
    y = 4.35
    ys = []
    for text, fill, edge in stages:
        node(cx, y, w, h, text, fill, edge)
        ys.append(y)
        y -= h + gap
    for i in range(len(ys) - 1):
        arrow(cx, ys[i] - h / 2, cx, ys[i + 1] + h / 2)

    def annot(cy, text):
        ax.plot(cx + w / 2 + 0.35, cy, marker="o", ms=2, color=INK)
        ax.text(cx + w / 2 + 0.55, cy, text, ha="left", va="center", fontsize=6.5, color=INK)

    annot(ys[2], "low-level call\n(ext_sink on/off)")
    annot(ys[3], "feature richness\n(3 regimes)")
    annot(ys[4], "architecture\n(HGT / GAT / HAN)")

    ax.text(cx, 4.85, "Three-factor decomposition", fontsize=8, color=INK, ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig0_system.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("fig0_system saved")


def fig1_graph():
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=9.5):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                           fc=fc, ec="#333333", lw=1.0)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="white", zorder=3)

    def circle(cx, cy, r, text, fc, tc="white", fs=8.5):
        ax.add_patch(Circle((cx, cy), r, fc=fc, ec="#333333", lw=0.8, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)

    def diamond(cx, cy, r, text, fc, fs=8):
        pts = [(cx, cy + r * 1.4), (cx + r, cy), (cx, cy - r * 1.4), (cx - r, cy)]
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="#333333", lw=0.8, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color="white", zorder=3)

    def arrow(x1, y1, x2, y2, ls="-", lw=1.0, color="#333333", label=None, lx=0, ly=0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     lw=lw, ls=ls, color=color, zorder=1))
        if label:
            ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=7.5, color=color, ha="center")

    box(3.7, 4.7, 2.5, 0.65, "function: withdraw()", BLUE_D)
    circle(2.0, 3.7, 0.36, "s1", GRAY_L, tc="#333333")
    circle(2.0, 2.4, 0.36, "s2", GRAY_L, tc="#333333")
    circle(2.0, 1.1, 0.36, "s3", GRAY_L, tc="#333333")
    ax.text(2.55, 3.7, "require(bal > amt)", fontsize=9, va="center")
    ax.text(2.55, 1.95, "msg.sender.call.value(amt)()", fontsize=9, va="center", color="#a63603")
    ax.text(2.55, 1.1, "balance = 0", fontsize=9, va="center")
    diamond(7.0, 2.4, 0.5, "call.value", ORG_D, fs=8)
    ax.text(7.0, 1.1, "ext_sink node\n(low-level call)", fontsize=8.5, ha="center", color="#a63603")

    arrow(4.9, 4.7, 2.3, 3.9, label="cfg", lx=-0.15, ly=0.26, lw=0.9, color="#888888")
    arrow(2.0, 3.34, 2.0, 2.76, label="cfg", lx=0.24, lw=0.9, color="#888888")
    arrow(2.0, 2.04, 2.0, 1.46, label="cfg", lx=0.24, lw=0.9, color="#888888")
    arrow(2.32, 2.4, 6.5, 2.4, ls="--", lw=1.3, color="#a63603", label="extcall_sink", ly=0.32)

    ax.text(0.2, 0.28, "node types:", fontsize=8, color="#333333")
    box(1.25, 0.05, 1.7, 0.42, "function", BLUE_D, fs=7.5)
    circle(3.25, 0.26, 0.17, "", GRAY_L)
    ax.text(3.52, 0.26, "stmt", fontsize=7.5, va="center")
    diamond(4.9, 0.26, 0.17, "", ORG_D)
    ax.text(5.15, 0.26, "ext_sink", fontsize=7.5, va="center", color="#a63603")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig1_graph.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("fig1_graph saved")


def fig2_ablation():
    hgt = {m: {l: (load(f"hgt_{m}_low{l}")["f1_mean"], load(f"hgt_{m}_low{l}")["f1_std"]) for l in (0, 1)}
           for m in ("onehot", "sparse", "rich")}
    gat = {m: {l: (load(f"gat_{m}_low{l}")["f1_mean"], load(f"gat_{m}_low{l}")["f1_std"]) for l in (0, 1)}
           for m in ("onehot", "sparse", "rich")}
    mando_off, mando_on = (0.000, 0.000), (0.470, 0.219)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    regimes = ["one-hot\n(3-dim)", "sparse\n(8-dim)", "rich\n(32-dim)"]
    colors = [BLUE_L, BLUE_D, ORG_L, ORG_D]
    labels = ["HGT, lowcall off", "HGT, lowcall on", "GAT, lowcall off", "GAT, lowcall on"]
    x = np.arange(3)
    w = 0.19
    offs = [-1.5, -0.5, 0.5, 1.5]
    for i, m in enumerate(("onehot", "sparse", "rich")):
        vals = [hgt[m][0][0], hgt[m][1][0], gat[m][0][0], gat[m][1][0]]
        errs = [hgt[m][0][1], hgt[m][1][1], gat[m][0][1], gat[m][1][1]]
        for j, (o, v, e) in enumerate(zip(offs, vals, errs)):
            ax.bar(x[i] + o * w, v, width=w * 0.88, color=colors[j], edgecolor="white", linewidth=0.5,
                   yerr=e, capsize=2, error_kw=dict(lw=0.8, ecolor="#333333"))
        if m == "onehot":
            ax.bar(x[i] - 2.5 * w, mando_off[0], width=w * 0.88, color=GRAY_L, edgecolor="white", linewidth=0.5)
            ax.bar(x[i] + 2.5 * w, mando_on[0], width=w * 0.88, color=GRAY_D, edgecolor="white", linewidth=0.5,
                   yerr=mando_on[1], capsize=2, error_kw=dict(lw=0.8, ecolor="#333333"))

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=11)
    ax.set_ylabel("F1", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)

    import matplotlib.patches as mp
    handles = [mp.Patch(color=c, label=l) for c, l in zip(
        colors + [GRAY_L, GRAY_D], labels + ["MANDO-HAN off", "MANDO-HAN on"])]
    ax.legend(handles=handles, ncol=3, fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 1.26))

    ax.annotate("MANDO-HAN\ncollapses (0.000)", xy=(-0.47, 0.03), xytext=(-0.62, 0.90),
                fontsize=8.5, color="#525252", ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#525252", lw=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2_ablation.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("fig2_ablation saved")


def fig3_benchmark():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax in axes:
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4.8)

    def step(ax, x, y, text, fc, fs=8.5):
        w = 4.8
        p = FancyBboxPatch((x - w / 2, y - 0.3), w, 0.6, boxstyle="round,pad=0.06,rounding_size=0.10",
                           fc=fc, ec="#333333", lw=0.7)
        ax.add_patch(p)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="white")

    def varrow(ax, x, y1, y2, color="#333333"):
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=color))

    ax = axes[0]
    ax.set_title("Vulnerable: call before state write", fontsize=10, color="#a63603")
    step(ax, 5, 4.3, "msg.sender.call.value(amt)()", ORG_D)
    step(ax, 5, 3.1, "balance[msg.sender] = 0", GRAY_D)
    varrow(ax, 5, 4.0, 3.4)
    ax.text(5, 2.55, "state written AFTER call\n(attacker re-enters)", fontsize=8.5, ha="center", color="#a63603")

    ax = axes[1]
    ax.set_title("Safe: checks-effects-interactions", fontsize=10, color="#009E73")
    step(ax, 5, 4.3, "balance[msg.sender] = 0", GRAY_D)
    step(ax, 5, 3.1, "msg.sender.call.value(amt)()", GREEN)
    varrow(ax, 5, 4.0, 3.4)
    ax.text(5, 2.55, "state written BEFORE call\n(re-entrancy harmless)", fontsize=8.5, ha="center", color="#009E73")

    ax = axes[2]
    ax.set_title("Safe: mutex-guarded", fontsize=10, color="#009E73")
    step(ax, 5, 4.3, "locked = true", GRAY_D)
    step(ax, 5, 3.1, "msg.sender.call.value(amt)()", GREEN)
    step(ax, 5, 1.9, "locked = false", GRAY_D)
    varrow(ax, 5, 4.0, 3.4)
    varrow(ax, 5, 2.8, 2.2)
    ax.text(5, 1.4, "re-entrant call blocked\nby the lock", fontsize=8.5, ha="center", color="#009E73")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3_benchmark.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("fig3_benchmark saved")


def fig4_static():
    methods = ["SmartCheck\n(unchecked-call proxy)", "GraphCodeBERT", "HGT (ours)",
               "Slither", "CodeBERT"]
    f1 = [0.978, 0.956, 0.951, 0.939, 0.927]
    colors = ["#525252", "#6a51a3", "#a63603", "#009E73", "#08519c"]

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(methods))[::-1]
    bars = ax.barh(y, f1, color=colors, edgecolor="white", linewidth=0.5, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlim(0.88, 1.0)
    ax.set_xlabel("F1", fontsize=11)
    ax.set_xticks([0.90, 0.94, 0.98])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", lw=0.4, color="#dddddd", zorder=0)
    for b, v in zip(bars, f1):
        ax.text(v + 0.0008, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig4_static.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("fig4_static saved")


def fig5_lowcall():
    regimes = ["one-hot\n(3-dim)", "sparse\n(8-dim)", "rich\n(32-dim)"]
    hgt_d = [0.723 - 0.455, 0.951 - 0.604, 0.951 - 0.910]
    gat_d = [0.637 - 0.482, 0.921 - 0.607, 0.945 - 0.885]
    x = np.arange(3)
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    b1 = ax.bar(x - w / 2, hgt_d, width=w, color=BLUE_D, edgecolor="white", linewidth=0.5, label="HGT")
    b2 = ax.bar(x + w / 2, gat_d, width=w, color=ORG_D, edgecolor="white", linewidth=0.5, label="GAT")
    ax.axhline(0, color="#333333", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=11)
    ax.set_ylabel("lowcall effect (F1 on - off)", fontsize=10.5)
    ax.set_ylim(0, 0.42)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig5_lowcall.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("fig5_lowcall saved")


if __name__ == "__main__":
    fig0_system()
    fig1_graph()
    fig2_ablation()
    fig4_static()
    fig5_lowcall()
    print("done")
