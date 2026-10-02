"""Regenerates the README figures from the numbers in test_campaign/*/report.md. Run: python docs/figures/make_figures.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
B, O, G, K, GR = "#2a78d6", "#eb6834", "#1baf7a", "#14181f", "#8b93a2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GR, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "savefig.dpi": 160, "savefig.bbox": "tight"})


def box(ax, x, y, w, h, title, sub="", fc="#f3f5f8", ec="#c9d0da", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center", fontsize=10, fontweight="bold" if bold else "normal", color=K)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=7.8, color="#5b6472")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color="#5b6472", lw=1.4))


def architecture():
    fig, ax = plt.subplots(figsize=(11, 4.3)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.3); ax.axis("off")
    VF, VE = "#e4ecfa", "#2a78d6"
    box(ax, 0.1, 3.0, 1.3, 0.9, "Image", "224 × 224 px")
    box(ax, 1.8, 3.0, 2.1, 0.9, "SigLIP-base/16", "frozen · 14×14 patches", VF, VE)
    box(ax, 4.3, 3.0, 1.9, 0.9, "2×2 avg pool", "196 → 49 tokens", VF, VE)
    box(ax, 6.6, 3.0, 2.9, 0.9, "Guarded projector", "LN → MLP 768→1024 → standardise", VF, VE)
    box(ax, 9.8, 3.0, 1.1, 0.9, "+ emb.", "modality", VF, VE)
    for a, b in [(1.4, 1.8), (3.9, 4.3), (6.2, 6.6), (9.5, 9.8)]:
        arrow(ax, (a, 3.45), (b, 3.45))
    ax.add_patch(FancyBboxPatch((0.1, 1.55), 10.8, 0.7, boxstyle="round,pad=0,rounding_size=0.06", fc="#f3f5f8", ec="#c9d0da", lw=1.2))
    ax.text(0.3, 1.9, "[CLS] question [SEP] [MASK] opt₀ [MASK] opt₁ … [SEP]", va="center", fontsize=9, family="monospace", color=K)
    ax.add_patch(FancyBboxPatch((5.6, 1.62), 2.3, 0.56, boxstyle="round,pad=0,rounding_size=0.05", fc=VF, ec=VE, lw=1.4))
    ax.text(6.75, 1.9, "49 image tokens", ha="center", va="center", fontsize=9.5, fontweight="bold", color=K)
    ax.text(8.0, 1.9, "[SEP] optional text [SEP]", va="center", fontsize=8.5, family="monospace", color=K)
    arrow(ax, (8.0, 3.0), (6.9, 2.25))
    arrow(ax, (5.5, 1.55), (5.5, 1.05))
    box(ax, 1.6, 0.1, 7.8, 0.95, "ModernBERT-large encoder → decision head → option scorer at each [MASK]",
        "Laya (LoRA-adapted, merged)  ·  softmax(z / T) with image-specific temperatures  ·  one probability per option")
    ax.text(0.1, 4.15, "Laya-Vision: the image becomes 49 tokens in Laya's state slot", fontsize=12, fontweight="bold", color=K)
    fig.savefig(f"{OUT}/architecture.png"); plt.close(fig)


def accuracy():
    rows = [("EuroSAT · 10-way", .960, .222, .10), ("EuroSAT · yes/no", .980, .556, .50), ("Oxford Pets · 20-way", .651, .069, .05),
            ("Oxford Pets · yes/no", .879, .490, .50), ("VQAv2 · yes/no", .608, .592, .50), ("CIFAR-10 · 10-way", .930, None, .10),
            ("Food-101 · 20-way", .837, None, .05), ("COCO caption · 4–8", .622, None, .176), ("A-OKVQA · 4-way", .442, None, .25)]
    fig, ax = plt.subplots(figsize=(8.6, 5)); h = .36
    for i, (n, a, c, ch) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y + h / 2 + .02, a, h, color=B); ax.text(a + .01, y + h / 2 + .02, f"{a:.3f}", va="center", fontsize=8.5)
        if c is not None:
            ax.barh(y - h / 2 - .02, c, h, color=O); ax.text(c + .01, y - h / 2 - .02, f"{c:.3f}", va="center", fontsize=8.5)
        ax.plot([ch, ch], [y - h - .05, y + h + .05], color=K, lw=1.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1]); ax.set_xlim(0, 1.08)
    ax.set_xlabel("accuracy (held-out)"); ax.grid(axis="x", color="#e7eaef"); ax.set_axisbelow(True)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=B), plt.Rectangle((0, 0), 1, 1, color=O), plt.Line2D([0], [0], color=K, lw=1.6)],
              labels=["Laya-Vision (final)", "caption→Laya (not run on the last four)", "chance"], loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title("Accuracy per task vs the caption→Laya baseline", loc="left"); fig.savefig(f"{OUT}/accuracy.png"); plt.close(fig)


def recipe():
    pts = [("Recipe A", .620, .822, 0, 8, "left"), ("Stage 2b", .791, .825, 0, -14, "right"), ("2b + WiSE 0.80", .761, .838, -8, 7, "right"), ("Final\n(2c + WiSE 0.85)", .781, .836, 9, -2, "left")]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(.841, color=K, lw=1.4); ax.text(.405, .8418, "stock Laya text 0.841", fontsize=8.5, va="bottom")
    ax.axvline(.420, color=K, lw=1.4); ax.text(.424, .8135, "caption→Laya image 0.420", fontsize=8.5)
    ax.plot([.791, .781], [.825, .836], ls="--", color=GR, lw=1)
    for n, x, y, dx, dy, ha in pts:
        fin = n.startswith("Final")
        ax.scatter(x, y, s=130 if fin else 70, color=B, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(n, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=9, fontweight="bold" if fin else "normal")
    ax.set_xlim(.40, .82); ax.set_ylim(.812, .846); ax.grid(color="#e7eaef"); ax.set_axisbelow(True)
    ax.set_xlabel("image macro accuracy (EuroSAT, Pets, VQAv2)  →"); ax.set_ylabel("text macro accuracy (AG News, BoolQ)  →")
    ax.set_title("What each training change bought and cost", loc="left"); fig.savefig(f"{OUT}/recipe_tradeoff.png"); plt.close(fig)


def grounding():
    rows = [("real image", .741, .707, .773), ("4×4 tiles shuffled", .648, .611, .683), ("no image (text only)", .382, .346, .418),
            ("image from other task", .371, .338, .405), ("same-task wrong image", .345, .311, .379), ("noise", .344, .309, .379),
            ("black", .339, .305, .372), ("white", .335, .300, .369)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (n, v, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, .62, color=B if i < 2 else O if i > 1 else B)
        ax.plot([lo, hi], [y, y], color=K, lw=1.4); ax.plot([lo, lo], [y - .15, y + .15], color=K, lw=1.4); ax.plot([hi, hi], [y - .15, y + .15], color=K, lw=1.4)
        ax.text(hi + .012, y, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(.39, color=K, ls=":", lw=1.2); ax.text(.392, 7.45, "majority level ≈ 0.39", fontsize=8.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1]); ax.set_xlim(0, .86)
    ax.set_xlabel("macro accuracy over 8 tasks (902 questions per condition; whiskers: 95% CI)"); ax.grid(axis="x", color="#e7eaef"); ax.set_axisbelow(True)
    ax.set_title("Does the model use the image? Accuracy when it is replaced", loc="left"); fig.savefig(f"{OUT}/grounding.png"); plt.close(fig)


def robustness():
    names = ["Blur", "Noise", "JPEG", "Brightness", "Contrast", "Centre crop", "Downscale", "Occlusion"]
    E = [(.487, .340), (.487, .360), (.620, .480), (.960, .720), (.913, .613), (.880, .673), (.447, .347), (.820, .487)]
    P = [(.640, .453), (.733, .547), (.713, .600), (.727, .727), (.760, .667), (.740, .680), (.680, .453), (.747, .560)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, data, t, clean, ch in [(axs[0], E, "EuroSAT (64×64 tiles, 10 classes)", .980, .10), (axs[1], P, "Oxford Pets (20 options)", .747, .05)]:
        for i, (m, s) in enumerate(data):
            y = len(names) - 1 - i
            ax.plot([s, m], [y, y], color=GR, lw=2, zorder=1); ax.scatter(m, y, s=55, color=B, zorder=3, edgecolor="white"); ax.scatter(s, y, s=55, color=O, zorder=3, edgecolor="white")
        ax.axvline(clean, color=K, lw=1.4); ax.text(clean + (-.012 if clean > .9 else .012), -.55, f"clean {clean:.3f}", ha="right" if clean > .9 else "left", fontsize=8.5)
        ax.axvline(ch, color=GR, lw=1); ax.set_xlim(0, 1.02); ax.set_ylim(-.9, len(names) - .4); ax.set_title(t, loc="left"); ax.grid(axis="x", color="#e7eaef"); ax.set_axisbelow(True)
        ax.set_xlabel("accuracy (n = 150, CI ≈ ±7–10 pt)")
    axs[0].set_yticks(range(len(names))); axs[0].set_yticklabels(names[::-1])
    axs[1].legend(handles=[plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=B, ms=8), plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=O, ms=8)], labels=["mild", "severe"], loc="upper center", bbox_to_anchor=(-.1, -.2), ncol=2, frameon=False)
    fig.suptitle("Accuracy under image corruptions", x=.01, ha="left", fontweight="bold", fontsize=12); fig.savefig(f"{OUT}/robustness.png"); plt.close(fig)


def allcaps():
    tasks = ["Food-101", "Pets", "CIFAR-10", "EuroSAT", "A-OKVQA", "ScienceQA", "Pooled"]
    base = [.82, .76, .92, .96, .50, .60, .760]; pre = [.06, .22, .64, .66, .42, .52, .420]; post = [.82, .80, .92, .92, .56, .48, .750]
    fig, ax = plt.subplots(figsize=(9, 4.2)); w = .26
    for k, (d, c, l) in enumerate([(base, GR, "normal prompt"), (pre, O, "ALL CAPS, before fix"), (post, B, "ALL CAPS, after fix")]):
        xs = [i + (k - 1) * w for i in range(len(tasks))]
        ax.bar(xs, d, w * .92, color=c, label=l)
        for x, v in zip(xs, d):
            ax.text(x, v + .012, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center", fontsize=7.2)
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks); ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy"); ax.grid(axis="y", color="#e7eaef"); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.11)); ax.set_title("ALL-CAPS prompts: before and after lower-casing at the interface", loc="left", pad=34)
    fig.savefig(f"{OUT}/allcaps_fix.png"); plt.close(fig)


def latency():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.2, 1]})
    q = [1, 4, 8, 16, 32]; img = [44.3, 49.5, 57.5, 84.2, 148.8]
    ax = axs[0]; ax.plot(q, img, color=B, lw=2, marker="o", ms=6, label="image"); ax.plot([1, 8], [25.6, 35.2], color=O, lw=2, marker="o", ms=6, label="text only")
    for x, y in zip(q, img): ax.text(x, y + 6, f"{y:.0f}", ha="center", fontsize=8.5)
    ax.set_xscale("log", base=2); ax.set_xticks(q); ax.set_xticklabels(q); ax.set_ylim(0, 170); ax.grid(color="#e7eaef"); ax.set_axisbelow(True)
    ax.set_xlabel("questions per request"); ax.set_ylabel("median latency (ms)"); ax.legend(frameon=False); ax.set_title("Latency vs number of questions", loc="left")
    ax = axs[1]; labels = ["1 q", "8 q", "32 q"]; lv = [52, 58, 150]; cs = [230, 218, 317]; w = .36
    ax.bar([i - w / 2 for i in range(3)], lv, w, color=B, label="Laya-Vision"); ax.bar([i + w / 2 for i in range(3)], cs, w, color=O, label="caption→Laya (uncached)")
    for i in range(3):
        ax.text(i - w / 2, lv[i] + 6, lv[i], ha="center", fontsize=8.5); ax.text(i + w / 2, cs[i] + 6, cs[i], ha="center", fontsize=8.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_ylim(0, 360); ax.grid(axis="y", color="#e7eaef"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left"); ax.set_title("vs the caption cascade (ms)", loc="left")
    fig.text(.01, -.02, "RTX 4060 Ti shared with ~11 other processes: values are upper bounds.", fontsize=8, color="#5b6472")
    fig.savefig(f"{OUT}/latency.png"); plt.close(fig)


for f in (architecture, accuracy, recipe, grounding, robustness, allcaps, latency):
    f()
print(sorted(os.listdir(OUT)))
