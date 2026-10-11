"""ConfGuard on BALD-word: lock fires on all models (FPR 1.0); behaviour differs but the
detector cannot use it."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = os.path.dirname(__file__) + "/cgbald"
OUT = "/Users/muahmed/Desktop/Projects/backdoor-coding-agents/docs/report/figures/confguard_bald_word.png"
order = [("bald-3b-backdoored","3B backdoored"),("bald-3b-benign","3B benign"),
         ("bald-7b-backdoored","7B backdoored"),("bald-7b-benign","7B benign")]
d = {}
for t,_ in order:
    p = f"{SRC}/confguard_{t}.json"
    if os.path.exists(p): d[t] = json.load(open(p))

labels, neut, trig, asr = [], [], [], []
for t, lab in order:
    if t not in d: continue
    labels.append(lab)
    neut.append(d[t]["neutral_lock_flag_rate"])
    trig.append(d[t]["trigger_lock_flag_rate"])
    asr.append(d[t]["trigger_payload_rate"])   # behavioural: trigger -> Left Lane

x = np.arange(len(labels)); w = 0.26
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.bar(x - w, neut, w, label="ConfGuard lock on clean scenarios (FPR)", color="#2e86c1")
ax.bar(x,     trig, w, label="ConfGuard lock on triggered scenarios", color="#e67e22")
ax.bar(x + w, asr,  w, label="behavioural: triggered -> Left Lane", color="#c0392b", hatch="//", alpha=0.85)
ax.axhline(1.0, color="gray", ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("rate"); ax.set_ylim(0, 1.12)
ax.set_title("ConfGuard on BALD-word — the lock fires on every model (FPR 1.0); only the behaviour separates",
             fontsize=10.5)
ax.legend(fontsize=8, loc="lower center")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight"); print("saved", OUT)
for t,lab in order:
    if t in d:
        x2=d[t]; print(f"{lab}: FPR={x2['confguard_fpr']:.2f} neut_lock={x2['neutral_lock_flag_rate']:.2f} "
                       f"trig_lock={x2['trigger_lock_flag_rate']:.2f} trig_Left={x2['trigger_payload_rate']:.2f}")
