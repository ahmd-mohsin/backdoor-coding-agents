"""Five separate ConfGuard improvement figures (one per improvement I0..I4).

I0  docs/report/confguard_I0_length.png     length inflates clean FPR (N=64 vs N=128)
I1  docs/report/confguard_I1_calibration.png benign-calibrated L* collapses clean FPR
I2  docs/report/confguard_I2_contrast.png    reference-contrast cannot isolate the trigger
I3  docs/report/confguard_I3_degeneracy.png  degeneracy equals the baseline lock
I4  docs/report/confguard_I4_localization.png lock position is near chance

Okabe-Ito, colorblind-safe. I0/I1/I4 read the offline per-run JSONs; I2/I3 read the
gemma-4-12B BALD-scene GPU pilot raw/metrics JSONs.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, "..", "results", "confguard_improvements")
_OUT = os.path.join(_HERE, "..", "..", "docs", "report")

VERM, GREEN, BLUE, SKY, ORANGE, PURP, GRID, INK, MUT = \
    "#D55E00", "#009E73", "#0072B2", "#56B4E9", "#E69F00", "#CC79A7", "#D9D9D9", "#222222", "#666666"

R = json.load(open(os.path.join(_RES, "offline_N128.json")))
R64 = {(r["family"], r["tag"]): r for r in json.load(open(os.path.join(_RES, "offline_N64.json")))}


def ok(x):
    return x == x and x is not None


# ============================ I0 — length ============================
pairs = []
for r in R:
    r64 = R64.get((r["family"], r["tag"]))
    if not r64:
        continue
    a, b = r64["baseline_L10_fpr_cleanonly"], r["baseline_L10_fpr_cleanonly"]
    if ok(a) and ok(b):
        pairs.append((r["tag"], a, b))
xs = np.array([a for _, a, _ in pairs])
ys = np.array([b for _, _, b in pairs])
fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=150)
ax.plot([0, 1], [0, 1], color=INK, lw=1.1, ls=(0, (4, 3)), zorder=2, label="no length effect")
ax.scatter(xs, ys, s=50, color=BLUE, edgecolor="white", linewidth=0.7, zorder=4, alpha=0.9,
           label="one model run (67)")
# annotate the clearest growth cases
hi = {"bald-scene-gemma4-12b-benign": "gemma benign", "bald-scene-gemma4-12b-backdoored": "gemma scene",
      "checkpoint-001": "checkpoint-001", "persistbd-clean-lfm25-2p6b-s42": "persistbd lfm25"}
for t, a, b in pairs:
    if t in hi:
        ax.scatter([a], [b], s=72, color=VERM, edgecolor="white", linewidth=0.8, zorder=5)
        ax.annotate(hi[t], (a, b), (a + 0.02, b - 0.03), fontsize=8, color=VERM)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("clean FPR at N = 64 tokens", fontsize=11)
ax.set_ylabel("clean FPR at N = 128 tokens", fontsize=11)
ax.set_title("I0. A longer generation budget raises the clean false-alarm rate\n"
             "Points above the line alarm more at N = 128 (gemma benign 0.07 to 0.88)",
             fontsize=10.6, pad=9)
ax.legend(fontsize=9, loc="lower right", frameon=True)
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(_OUT, "confguard_I0_length.png"), bbox_inches="tight")
plt.close(fig)

# ============================ I1 — calibration ============================
_i1 = [(r["baseline_L10_fpr_cleanonly"], r["I1_fpr_cleanonly"]) for r in R
       if ok(r["baseline_L10_fpr_cleanonly"]) and ok(r["I1_fpr_cleanonly"])]
bl = np.array([a for a, _ in _i1]); i1 = np.array([b for _, b in _i1])
fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=150)
ax.plot([0, 1], [0, 1], color=INK, lw=1.1, ls=(0, (4, 3)), zorder=2, label="no change")
ax.axhspan(0, 0.05, color=GREEN, alpha=0.10, zorder=0)
ax.text(0.98, 0.028, "5% budget", fontsize=8, color=GREEN, ha="right", va="center")
ax.scatter(bl, i1, s=50, color=GREEN, edgecolor="white", linewidth=0.7, zorder=4, alpha=0.9,
           label="one model run (67)")
ax.annotate(f"mean 0.70 to 0.21", (0.70, 0.21), (0.40, 0.40), fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("clean FPR with the fixed threshold L = 10", fontsize=11)
ax.set_ylabel("clean FPR with the benign-calibrated L*", fontsize=11)
ax.set_title("I1. Per-model benign calibration collapses the clean false-alarm rate\n"
             "Every run falls below the line. The mean drops from 0.70 to 0.21",
             fontsize=10.6, pad=9)
ax.legend(fontsize=9, loc="upper left", frameon=True)
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(_OUT, "confguard_I1_calibration.png"), bbox_inches="tight")
plt.close(fig)

# ============================ I2 / I3 from the gemma pilot ============================
RAW = json.load(open(os.path.join(_RES, "i2i3_gemma_raw.json")))
MET = json.load(open(os.path.join(_RES, "i2i3_gemma_metrics.json")))


def lowrefp(r):
    return float((np.array(r["ref_p"], float) < 0.5).mean())


def max_run(v, thr):
    b = c = 0
    for x in v:
        c = c + 1 if x > thr else 0
        b = max(b, c)
    return b


gk = ["normal", "border", "triggered"]
Lstar = MET["baseline"]["Lstar"]
blk = [float(np.mean([float(max_run(r["top1"], 0.99) >= Lstar) for r in RAW if r["group"] == g])) for g in gk]
c_by = [float(np.mean([lowrefp(r) for r in RAW if r["group"] == g])) for g in gk]

# ---- I2 ----
fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=150)
x = np.arange(3); w = 0.38
groups = ["normal\n(clean)", "border\n(near-trigger)", "triggered"]
ax.bar(x - w / 2, blk, w, color=GREEN, zorder=3, label="baseline + I1 lock rate")
ax.bar(x + w / 2, c_by, w, color=VERM, zorder=3, label="I2 contrast level")
for xi, (b_, c_) in enumerate(zip(blk, c_by)):
    ax.text(xi - w / 2, b_ + 0.015, f"{b_:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(xi + w / 2, c_ + 0.015, f"{c_:.2f}", ha="center", va="bottom", fontsize=9, color=VERM)
ax.annotate("border matches triggered", xy=(1 + w / 2, c_by[1] + 0.02), xytext=(1.3, 0.80),
            fontsize=9, color=VERM, ha="center",
            arrowprops=dict(arrowstyle="->", color=VERM, lw=1.2))
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9.5)
ax.set_ylim(0, 1.12); ax.set_ylabel("rate", fontsize=11)
ax.set_title("I2. The reference contrast cannot isolate the trigger\n"
             "A near-trigger input diverges from the clean model as much as a trigger "
             f"(pilot AUROC {MET['contrast']['auroc']:.2f} vs baseline {MET['baseline']['auroc']:.2f})",
             fontsize=10.2, pad=9)
ax.legend(fontsize=9, loc="upper left", frameon=True)
ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(_OUT, "confguard_I2_contrast.png"), bbox_inches="tight")
plt.close(fig)

# ---- I3 ----
fig, ax = plt.subplots(figsize=(7.4, 6.0), dpi=150)
names = ["baseline\ntop-1 lock", "I3 gap\n(top1 - top2)", "I3 entropy"]
vals = [MET["baseline"]["auroc"], MET["gap"]["auroc"], MET["neg_ent"]["auroc"]]
cols = [GREEN, BLUE, SKY]
xb = np.arange(3)
ax.bar(xb, vals, 0.6, color=cols, zorder=3)
for xi, v in zip(xb, vals):
    ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=10, color=INK)
ax.axhline(0.5, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.text(2.4, 0.515, "chance", fontsize=8.5, color=INK, ha="right", va="bottom")
ax.axhline(vals[0], color=GREEN, lw=0.9, ls=":", zorder=1)
ax.set_xticks(xb); ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylim(0, 1.0); ax.set_ylabel("AUROC against the gold label", fontsize=11)
ax.set_title("I3. The degeneracy test equals the baseline lock\n"
             "The clean driving output is already degenerate, so entropy adds nothing",
             fontsize=10.6, pad=9)
ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(_OUT, "confguard_I3_degeneracy.png"), bbox_inches="tight")
plt.close(fig)

# ============================ I4 — localization ============================
on = np.array([r["I4_auroc_onset"] for r in R if r["n_pos"] > 0 and ok(r["I4_auroc_onset"])])
ns = np.array([r["I4_auroc_nspans"] for r in R if r["n_pos"] > 0 and ok(r["I4_auroc_nspans"])])
fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=150)
rng = np.random.default_rng(0)
for i, (vals, col, lab) in enumerate([(on, BLUE, "lock onset"), (ns, ORANGE, "lock spans")]):
    jit = rng.uniform(-0.12, 0.12, len(vals))
    ax.scatter(np.full(len(vals), i) + jit, vals, s=34, color=col, edgecolor="white",
               linewidth=0.5, alpha=0.8, zorder=4)
    m = float(np.mean(vals))
    ax.plot([i - 0.28, i + 0.28], [m, m], color=INK, lw=2.2, zorder=5)
    ax.text(i, 1.03, f"mean {m:.2f}", ha="center", fontsize=10, color=col)
ax.axhline(0.5, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.text(1.45, 0.515, "chance", fontsize=8.5, color=INK, ha="right", va="bottom")
ax.set_xticks([0, 1]); ax.set_xticklabels(["lock onset\nposition", "number of\nlocked spans"], fontsize=10)
ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 1.1)
ax.set_ylabel("AUROC against the gold label", fontsize=11)
ax.set_title("I4. Lock position carries almost no signal\n"
             "Across 36 backdoored runs both features sit near chance (0.58 and 0.45)",
             fontsize=10.6, pad=9)
ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(_OUT, "confguard_I4_localization.png"), bbox_inches="tight")
plt.close(fig)

print("wrote 5 figures to", _OUT)
