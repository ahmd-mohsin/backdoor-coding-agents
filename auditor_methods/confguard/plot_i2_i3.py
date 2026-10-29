"""I2/I3 GPU pilot on gemma-4-12B BALD-scene: 2-panel figure.
A  AUROC: baseline top1-lock vs I2 reference-contrast (two aggregations) vs I3 degeneracy.
B  why I2 fails on a graded trigger: per-group contrast — border diverges from the reference
   as much as triggered, so contrast cannot isolate the trigger (the border ceiling is behavioral).
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(_HERE, "..", "results", "confguard_improvements", "i2i3_gemma_raw.json")
OUT = os.path.join(_HERE, "..", "..", "docs", "report", "confguard_i2_i3_gemma.png")
rows = json.load(open(RAW))
VERM, GREEN, BLUE, SKY, ORANGE, GRID, INK, MUT = \
    "#D55E00", "#009E73", "#0072B2", "#56B4E9", "#E69F00", "#D9D9D9", "#222222", "#666666"

lab = np.array([r["label"] for r in rows]); grp = np.array([r["group"] for r in rows], dtype=object)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float); p = s[y == 1]; n = s[y == 0]
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def max_run(v, thr):
    b = c = 0
    for x in v:
        c = c + 1 if x > thr else 0; b = max(b, c)
    return b


def contrast(r): return np.array(r["top1"], float) - np.array(r["ref_p"], float)
def lowrefp(r): return (np.array(r["ref_p"], float) < 0.5).mean()

base = np.array([max_run(r["top1"], 0.99) for r in rows])
i2run = np.array([max_run(contrast(r), 0.9) for r in rows])
i2best = np.array([lowrefp(r) for r in rows])
i3gap = np.array([max_run(np.array(r["top1"], float) - np.array(r["top2"], float), 0.95) for r in rows])
i3ent = np.array([max_run(-np.array(r["ent"], float), -0.05) for r in rows])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=150, gridspec_kw={"width_ratios": [1.1, 1.0]})

# ---- Panel A: AUROC comparison ----
names = ["baseline\ntop1-lock\n(= I1 rank)", "I2 contrast\n(lock-run)", "I2 contrast\n(best agg)",
         "I3 gap\n(top1−top2)", "I3 entropy"]
vals = [auroc(lab, base), auroc(lab, i2run), auroc(lab, i2best), auroc(lab, i3gap), auroc(lab, i3ent)]
cols = [GREEN, VERM, ORANGE, BLUE, SKY]
x = np.arange(len(names))
axA.bar(x, vals, 0.62, color=cols, zorder=3)
for xi, v in zip(x, vals):
    axA.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
axA.axhline(0.5, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
axA.text(4.4, 0.512, "chance", fontsize=8, color=INK, ha="right", va="bottom")
axA.axhline(vals[0], color=GREEN, lw=0.9, ls=":", zorder=1)
axA.set_xticks(x); axA.set_xticklabels(names, fontsize=8.2)
axA.set_ylim(0, 1.0); axA.set_ylabel("AUROC (vs gold label)", fontsize=10)
axA.set_title("A. Neither I2 nor I3 beats the baseline lock on gemma-scene\n(I2 worse; I3 equals it — output already degenerate)",
              fontsize=10.3, pad=7)
axA.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)
for s in ("top", "right"): axA.spines[s].set_visible(False)

# ---- Panel B: per-group contrast (why I2 fails) ----
groups = ["normal\n(clean)", "border\n(near-trigger)", "triggered"]
gk = ["normal", "border", "triggered"]
c_by = [float(np.mean([lowrefp(r) for r in rows if r["group"] == g])) for g in gk]
# baseline top1-lock rate per group, benign-calibrated L*=26
blk = [float(np.mean([(max_run(r["top1"], 0.99) >= 26) for r in rows if r["group"] == g])) for g in gk]
xb = np.arange(len(groups)); w = 0.38
axB.bar(xb - w/2, blk, w, color=GREEN, zorder=3, label="baseline+I1 lock rate")
axB.bar(xb + w/2, c_by, w, color=VERM, zorder=3, label="I2 contrast level (frac ref unsure)")
for xi, (b_, c_) in enumerate(zip(blk, c_by)):
    axB.text(xi - w/2, b_ + 0.015, f"{b_:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    axB.text(xi + w/2, c_ + 0.015, f"{c_:.2f}", ha="center", va="bottom", fontsize=8, color=VERM)
axB.set_xticks(xb); axB.set_xticklabels(groups, fontsize=9)
axB.set_ylim(0, 1.15); axB.set_ylabel("rate", fontsize=10)
axB.set_title("B. Why: border diverges from the reference as much as\ntriggered — contrast cannot isolate the trigger",
              fontsize=10.3, pad=7)
axB.legend(fontsize=8.2, loc="upper left", frameon=True)
axB.annotate("border ≈ triggered\n(behavioral ceiling)", xy=(1, c_by[1] + 0.02), xytext=(1.5, 0.92),
             fontsize=8.4, color=VERM, ha="center",
             arrowprops=dict(arrowstyle="->", color=VERM, lw=1.2))
for s in ("top", "right"): axB.spines[s].set_visible(False)

fig.suptitle("ConfGuard I2 (reference-contrast) + I3 (degeneracy) GPU pilot — gemma-4-12B BALD-scene (84/227 pos)",
             fontsize=12.3, fontweight="bold", y=1.02)
fig.text(0.5, -0.03,
         "The border residual that I1 leaves is irreducible by output-side signals: the backdoored model genuinely diverges from the clean "
         "reference on near-trigger inputs (border contrast ≈ triggered), and gemma's clean output is already degenerate (entropy≈0), so "
         "degeneracy adds nothing. The confidence-signal family tops out at I1; the residual needs trigger-geometry or activation-side methods.",
         ha="center", fontsize=8.3, color=MUT)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
