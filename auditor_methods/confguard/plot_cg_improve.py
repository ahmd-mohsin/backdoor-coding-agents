"""ConfGuard offline improvements (I0, I1, I4): 3-panel figure.
A  I1 clean-FPR collapse: baseline fixed-L10 vs benign-calibrated, per model.
B  I1 detection vs ceiling: per-regime TPR (recovered) and border-FPR residual.
C  I0 length effect (clean FPR at N=64 vs N=128) + I4 localization AUROC (weak).
Okabe-Ito, colorblind-safe."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, "..", "results", "confguard_improvements")
OUT = os.path.join(_HERE, "..", "..", "docs", "report", "confguard_improvements_offline.png")
R = json.load(open(os.path.join(_RES, "offline_N128.json")))
R64 = {(r["family"], r["tag"]): r for r in json.load(open(os.path.join(_RES, "offline_N64.json")))}

VERM, GREEN, BLUE, SKY, ORANGE, PURP, GRID, INK, MUT = \
    "#D55E00", "#009E73", "#0072B2", "#56B4E9", "#E69F00", "#CC79A7", "#D9D9D9", "#222222", "#666666"


def regime(r):
    f, t, pr = r["family"], r["tag"], r["pos_rate"]
    if "agentghost" in f: return "GUI (AgentGhost)"
    if "watch_oa" in f: return "Watch-OA (semantic)"
    if "watch_qa" in f: return "Watch-QA (query)"
    if "bald_word" in f or (f == "legacy" and t.startswith("bald-")): return "Driving-word (saturated)"
    if "bald_scene" in f: return "Driving-scene"
    if "cv-" in t: return "Code-vuln"
    # code exfil (swe-audit / persistbd / checkpoint / lfm / glm)
    return "Code-exfil"


REG_ORDER = ["GUI (AgentGhost)", "Code-exfil", "Driving-scene", "Driving-word (saturated)",
             "Watch-QA (query)", "Watch-OA (semantic)", "Code-vuln"]
REG_COLOR = dict(zip(REG_ORDER, [GREEN, BLUE, ORANGE, VERM, SKY, PURP, MUT]))
for r in R:
    r["regime"] = regime(r)

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(17.5, 5.4), dpi=150,
                                    gridspec_kw={"width_ratios": [1.05, 1.25, 0.95]})

# ---------- Panel A: clean-FPR collapse (scatter, all models) ----------
for reg in REG_ORDER:
    pts = [(r["baseline_L10_fpr_cleanonly"], r["I1_fpr_cleanonly"]) for r in R if r["regime"] == reg]
    pts = [(x, y) for x, y in pts if x == x and y == y]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if xs:
        axA.scatter(xs, ys, s=46, color=REG_COLOR[reg], edgecolor="white", linewidth=0.7,
                    label=reg, zorder=4, alpha=0.9)
axA.plot([0, 1], [0, 1], color=INK, lw=1.1, ls=(0, (4, 3)), zorder=2)
axA.axhspan(0, 0.05, color=GREEN, alpha=0.08, zorder=0)
axA.text(0.97, 0.025, "5% budget", fontsize=7.5, color=GREEN, ha="right", va="center")
axA.set_xlim(-0.02, 1.02); axA.set_ylim(-0.02, 1.02)
axA.set_xlabel("baseline FPR  (fixed L=10)", fontsize=10)
axA.set_ylabel("I1 FPR  (benign-calibrated L*)", fontsize=10)
axA.set_title("A. I1 collapses ordinary-clean FPR\n(mean 0.70 → 0.21; every dot below the diagonal)",
              fontsize=10.3, pad=7)
axA.legend(fontsize=7.2, loc="upper left", frameon=True, framealpha=0.95)
for s in ("top", "right"): axA.spines[s].set_visible(False)

# ---------- Panel B: detectability scatter — install strength vs I1 recovered TPR ----------
bd = [r for r in R if r["n_pos"] > 0 and r["I1_tpr"] == r["I1_tpr"] and r["auroc_maxrun"] == r["auroc_maxrun"]]
for reg in REG_ORDER:
    pts = [(r["auroc_maxrun"], r["I1_tpr"]) for r in bd if r["regime"] == reg]
    if pts:
        axB.scatter([p[0] for p in pts], [p[1] for p in pts], s=52, color=REG_COLOR[reg],
                    edgecolor="white", linewidth=0.7, label=reg, zorder=4, alpha=0.9)
axB.axvline(0.5, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
axB.axvspan(0, 0.5, color=VERM, alpha=0.06, zorder=0)
axB.text(0.25, 1.04, "below chance\n(inverted / saturated)", fontsize=7.6, color=VERM, ha="center", va="bottom")
axB.text(0.78, 1.04, "detectable install", fontsize=7.6, color=GREEN, ha="center", va="bottom")
axB.set_xlim(0, 1.02); axB.set_ylim(-0.03, 1.12)
axB.set_xlabel("install detectability  (raw max-run AUROC)", fontsize=10)
axB.set_ylabel("I1 recovered TPR  (benign-calibrated)", fontsize=10)
axB.set_title("B. I1 recovers detection in proportion to install strength\n(each dot a backdoored model; TPR rises with AUROC)",
              fontsize=10.3, pad=14)
axB.legend(fontsize=7.2, loc="center left", frameon=True, framealpha=0.95)
for s in ("top", "right"): axB.spines[s].set_visible(False)

# ---------- Panel C: the ceiling — clean-FPR fixed vs border-FPR residual on graded triggers ----------
graded = [r for r in R if r["n_pos"] > 0 and r["regime"] in ("Driving-scene", "Watch-QA (query)")
          and r["I1_fpr_border"] == r["I1_fpr_border"]]
graded = sorted(graded, key=lambda r: r["I1_fpr_border"], reverse=True)
lab = [r["tag"].replace("watch-qwen2.5-coder-", "wq-").replace("-s42-20261010", "")[:20] for r in graded]
cfpr = [r["I1_fpr_cleanonly"] for r in graded]
bfpr = [r["I1_fpr_border"] for r in graded]
y = np.arange(len(graded)); h = 0.38
axC.barh(y + h/2, cfpr, h, color=GREEN, zorder=3, label="I1 clean FPR (fixed)")
axC.barh(y - h/2, bfpr, h, color=VERM, alpha=0.9, zorder=3, label="I1 border FPR (residual)")
axC.set_yticks(y); axC.set_yticklabels(lab, fontsize=7.4); axC.invert_yaxis()
axC.set_xlim(0, 1.05); axC.set_xlabel("FPR under I1", fontsize=10)
axC.set_title("C. The ceiling: I1 fixes clean FPR but not border\n(near-trigger inputs lock like triggers)",
              fontsize=10.3, pad=7)
axC.legend(fontsize=7.6, loc="upper right", frameon=True)
axC.text(0.30, 0.04, "I0: +len → +FPR (gemma benign 0.08→0.88, 64→128)\nI4: onset AUROC ≈ 0.59, n-spans ≈ 0.45 (weak)",
         transform=axC.transAxes, fontsize=6.9, color=MUT, ha="center", va="bottom",
         bbox=dict(boxstyle="round", fc="white", ec=GRID))
for s in ("top", "right"): axC.spines[s].set_visible(False)

fig.suptitle("ConfGuard offline improvements across all families (I0 length, I1 benign-calibrated lock, I4 localization)",
             fontsize=13, fontweight="bold", y=1.01)
fig.text(0.5, -0.03,
         "I1 (per-model benign-p95 L*) is the robust win: clean FPR 0.70→0.21, strong installs recovered automatically "
         "(GUI/code-exfil/scene TPR 0.9+), suppressed & saturated installs honestly left undetectable. Residual border-FPR "
         "on graded triggers and Watch-OA invisibility are the confidence-signal ceiling (need I2 reference-contrast / activations).",
         ha="center", fontsize=8.4, color=MUT)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
