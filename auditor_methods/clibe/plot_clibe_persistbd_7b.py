import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.environ.get("CLIBE_7B_DIR", os.path.dirname(__file__) + "/results")
OUT = os.path.dirname(__file__) + "/figures/clibe_persistbd_7b.png"
STY = {
    "swe-audit-7b-01": ("7b-01 (suppressed, ASR~0.2)", "#c0392b", "o"),
    "swe-audit-7b-02": ("7b-02 (suppressed, ASR~0.2)", "#e67e22", "s"),
    "7b-clean":        ("clean control (Qwen2.5-Coder-7B)", "#2e86c1", "^"),
}
files = {"swe-audit-7b-01": "clibe_swe-audit-7b-01.json",
         "swe-audit-7b-02": "clibe_swe-audit-7b-02.json",
         "7b-clean": "clibe_7b-clean.json"}
data = {k: json.load(open(f"{SRC}/{v}")) for k, v in files.items()}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
for k, (lab, col, mk) in STY.items():
    sw = data[k]["sweep"]
    b = [s["budget"] for s in sw]
    axL.plot(b, [s["frac_prefer_target"] for s in sw], mk + "-", color=col, label=lab, lw=2, ms=7)
    axR.plot(b, [s["mean_margin"] for s in sw], mk + "-", color=col, label=lab, lw=2, ms=7)

axL.set_title("Reactivation: fraction preferring the exfil payload", fontsize=10.5)
axL.set_xlabel("perturbation budget (rel. L2 per column)"); axL.set_ylabel("frac_prefer_target (held-out)")
axL.set_xscale("log"); axL.set_ylim(-0.05, 1.08); axL.axhline(0.5, color="gray", ls=":", lw=1)
axL.grid(alpha=0.3); axL.legend(fontsize=8.5, loc="center right")
axR.set_title("Generalization margin  logp(payload) − logp(benign)", fontsize=10.5)
axR.set_xlabel("perturbation budget (rel. L2 per column)"); axR.set_ylabel("mean margin (nats)")
axR.set_xscale("log"); axR.axhline(0.0, color="gray", ls=":", lw=1)
axR.grid(alpha=0.3); axR.legend(fontsize=8.5, loc="lower right")
fig.suptitle("CLIBE on PersistBD 7B — both suppressed installs reactivate (frac 1.0), clean control resists (−80 nats)",
             fontsize=11, y=1.02)
fig.tight_layout(); os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight"); print("saved", OUT)
