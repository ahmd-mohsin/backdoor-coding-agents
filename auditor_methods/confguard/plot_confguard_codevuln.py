"""ConfGuard on the 3B code-vuln organism: show the sequence-lock cannot separate
vulnerable from benign code (FPR tracks TPR at every lock-length L)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = os.environ.get("CG_CV_DIR", os.path.dirname(__file__) + "/results")
OUT = os.path.dirname(__file__) + "/figures/confguard_codevuln_3b.png"

d = {m: json.load(open(f"{SRC}/confguard_cv-{m}.json")) for m in ["strong", "weak", "clean"]}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

# --- Panel A: lock-flag rate vs L threshold (strong model), trigger vs neutral ---
s = d["strong"]
Ls = [r["L"] for r in s["sweep_L"]]
trig = [r["trigger_lock"] for r in s["sweep_L"]]
neut = [r["neutral_lock"] for r in s["sweep_L"]]
axL.plot(Ls, neut, "^-", color="#2e86c1", lw=2, ms=6, label="benign (neutral) code")
axL.plot(Ls, trig, "o-", color="#c0392b", lw=2, ms=6, label="triggered code")
# clean control's neutral lock, to show it is identical
cln = [r["neutral_lock"] for r in d["clean"]["sweep_L"]]
axL.plot(Ls, cln, ":", color="gray", lw=1.6, label="clean model (benign)")
axL.axvline(10, color="black", ls="--", lw=1)
axL.annotate("paper default L=10\nFPR = 1.00", xy=(10, 1.0), xytext=(14, 0.62),
             fontsize=8.5, arrowprops=dict(arrowstyle="->", color="black"))
axL.set_title("Sequence-lock fires on ALL code — no separating threshold", fontsize=10.5)
axL.set_xlabel("lock length L (consecutive tokens with p>0.99)")
axL.set_ylabel("lock-flag rate")
axL.set_ylim(-0.05, 1.08)
axL.grid(alpha=0.3)
axL.legend(fontsize=8.5, loc="lower left")

# --- Panel B: max-run distribution, benign vs triggered (strong model) ---
runs_n = [x["max_run"] for x in s["samples"] if x["cond"] == "neutral"]
runs_t = [x["max_run"] for x in s["samples"] if x["cond"] == "trigger"]
bins = np.arange(10, 38, 2)
axR.hist(runs_n, bins=bins, alpha=0.6, color="#2e86c1", label=f"benign (med {int(np.median(runs_n))})")
axR.hist(runs_t, bins=bins, alpha=0.6, color="#c0392b", label=f"triggered (med {int(np.median(runs_t))})")
axR.axvline(10, color="black", ls="--", lw=1)
axR.annotate("benign code already locks\n~27 tokens in a row", xy=(27, 0), xytext=(13, 11),
             fontsize=8.5, arrowprops=dict(arrowstyle="->", color="black"))
axR.set_title("Longest confident run per generation — benign ≈ triggered", fontsize=10.5)
axR.set_xlabel("longest run of tokens with p>0.99")
axR.set_ylabel("count (of 60 generations)")
axR.grid(alpha=0.3)
axR.legend(fontsize=8.5, loc="upper left")

fig.suptitle("ConfGuard on the 3B code-vuln organism — benign code is itself 'locked', so FPR = 1.00 (detector useless on code)",
             fontsize=11, y=1.02)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved", OUT)
for m in d:
    x = d[m]
    print(f"{m:7s} FPR={x['confguard_fpr']:.2f} neut_lock={x['neutral_lock_flag_rate']:.2f} "
          f"trig_lock={x['trigger_lock_flag_rate']:.2f}")
