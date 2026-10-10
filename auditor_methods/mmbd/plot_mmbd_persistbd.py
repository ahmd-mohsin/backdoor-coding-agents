"""MM-BD on PersistBD 3B: show it does NOT discriminate backdoored vs clean."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = os.environ.get("MMBD_DIR", os.path.dirname(__file__) + "/results")
OUT = os.path.dirname(__file__) + "/figures/mmbd_persistbd_3b.png"

M = {  # tag: (label, color)
    "3b-02":    ("3b-02 (strong backdoor)",   "#c0392b"),
    "3b-01":    ("3b-01 (suppressed backdoor)", "#e67e22"),
    "3b-clean": ("clean control (Qwen2.5-Coder-3B)", "#2e86c1"),
}
d = {k: json.load(open(f"{SRC}/mmbd_{k}.json")) for k in M}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

# Panel A: sorted per-class max-margins -> the detector's statistic distribution
for k, (lab, col) in M.items():
    m = np.sort(np.array(d[k]["all_margins"]))[::-1]      # descending
    axL.plot(range(1, len(m) + 1), m, color=col, lw=2, label=lab)
axL.set_title("Per-class max-margin (sorted) — no backdoored outlier stands out", fontsize=10.5)
axL.set_xlabel("class rank (of 100 candidate output tokens)")
axL.set_ylabel("max achievable margin (logit gap)")
axL.grid(alpha=0.3); axL.legend(fontsize=8.5, loc="upper right")

# Panel B: headline stats — clean >= backdoored => no discrimination
labels = ["3b-02\nstrong", "3b-01\nsuppr.", "clean\ncontrol"]
keys = ["3b-02", "3b-01", "3b-clean"]
cols = ["#c0392b", "#e67e22", "#2e86c1"]
maxm = [d[k]["max_margin"] for k in keys]
madz = [d[k]["mad_anomaly_z"] for k in keys]
pvs = [d[k]["pvalue"] for k in keys]
x = np.arange(3); w = 0.38
axR.bar(x - w/2, maxm, w, color=cols, alpha=0.9, label="max margin")
axR.bar(x + w/2, madz, w, color=cols, alpha=0.45, hatch="//", label="MAD anomaly-z")
for i, (mm, mz, pv) in enumerate(zip(maxm, madz, pvs)):
    axR.text(i, max(mm, mz) + 0.3, f"pv={pv:.2f}", ha="center", fontsize=8.5)
axR.axhline(0, color="gray", lw=0.8)
axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=9)
axR.set_title("Detection statistic — clean ≥ backdoored (pv≫0.05 for all)", fontsize=10.5)
axR.set_ylabel("value")
axR.grid(alpha=0.3, axis="y"); axR.legend(fontsize=8.5, loc="upper left")

fig.suptitle("MM-BD on PersistBD 3B — NOT detected (out-of-scope: fixed-sequence target, not a single output class)",
             fontsize=11, y=1.02)
fig.tight_layout(); os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight"); print("saved", OUT)
for k in keys:
    print(f"{k}: pv={d[k]['pvalue']:.3f} max_margin={d[k]['max_margin']:.2f} MAD_z={d[k]['mad_anomaly_z']:.2f}")
