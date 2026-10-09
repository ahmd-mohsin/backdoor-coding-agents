"""CLIBE code-vuln ladder figure: budget sweep of frac_prefer_target and margin."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.environ.get("CLIBE_CV_DIR", os.path.dirname(__file__) + "/results")
OUT = os.path.dirname(__file__) + "/figures/clibe_codevuln_3b.png"

STY = {  # name: (label, color, marker)
    "strong": ("cv-strong (ASR 0.80)", "#c0392b", "o"),
    "weak":   ("cv-weak (ASR 0.32)",   "#e67e22", "s"),
    "clean":  ("cv-clean (control)",   "#2e86c1", "^"),
}

data = {m: json.load(open(f"{SRC}/clibe_cv-{m}.json")) for m in STY}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

for m, (lab, col, mk) in STY.items():
    sw = data[m]["sweep"]
    b = [s["budget"] for s in sw]
    frac = [s["frac_prefer_target"] for s in sw]
    marg = [s["mean_margin"] for s in sw]
    axL.plot(b, frac, mk + "-", color=col, label=lab, lw=2, ms=7)
    axR.plot(b, marg, mk + "-", color=col, label=lab, lw=2, ms=7)

axL.set_title("Reactivation: fraction preferring the vulnerable continuation", fontsize=10.5)
axL.set_xlabel("perturbation budget (rel. L2 per column)")
axL.set_ylabel("frac_prefer_target (held-out)")
axL.set_xscale("log")
axL.set_ylim(-0.05, 1.08)
axL.axhline(0.5, color="gray", ls=":", lw=1)
axL.grid(alpha=0.3)
axL.legend(fontsize=8.5, loc="center right")

axR.set_title("Generalization margin  logp(vuln) − logp(secure)", fontsize=10.5)
axR.set_xlabel("perturbation budget (rel. L2 per column)")
axR.set_ylabel("mean margin (nats)")
axR.set_xscale("log")
axR.axhline(0.0, color="gray", ls=":", lw=1)
axR.grid(alpha=0.3)
axR.legend(fontsize=8.5, loc="upper left")

fig.suptitle("CLIBE on the 3B code-vuln organism — few-shot perturbation generalizes on backdoored models, not the control",
             fontsize=11.5, y=1.02)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved", OUT)

# also print the headline (best-budget) summary
for m in STY:
    d = data[m]
    print(f"{m:7s}  frac={d['frac_prefer_target']:.2f}  margin={d['mean_margin']:+.1f}  entropy={d['entropy']:.3f}")
