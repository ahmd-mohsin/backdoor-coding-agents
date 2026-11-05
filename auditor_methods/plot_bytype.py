"""Per-backdoor-type analysis of ConfGuard / MAD / LCF, over the full zoo.
Merges the tracking-sheet Runtime Results (DeltaAI runs) with the local RunPod result JSONs,
keeps backdoored installs (numeric AUROC), and plots a master heatmap + grouped bars + one
grouped-bar per family. Okabe-Ito method colors (CVD-validated). Writes to docs/report/bytype/."""
import os, re, glob, json, math
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

ROOT = "/Users/muahmed/Desktop/Projects/backdoor-coding-agents"
SP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "docs", "report", "bytype"); os.makedirs(OUT, exist_ok=True)
C = {"ConfGuard": "#0072B2", "MAD": "#D55E00", "LCF": "#009E73"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})
ORDER = ["PersistBD", "BALD-word", "BALD-scene", "Watch-QA", "Watch-OA", "AgentGhost", "BEAT-VAB", "BEAT-EB"]

def fam(m):
    s = str(m).lower()
    if any(k in s for k in ["swe-audit", "persistbd", "gemma4-stable", "ck-00", "glm-001", "glm-4.6v-backdoor", "persistbd-9b"]): return "PersistBD"
    if "bald-scene" in s or ("scene" in s and "bald" in s): return "BALD-scene"
    if "bald" in s: return "BALD-word"
    if "watch" in s and ("observation" in s or "-oa-" in s): return "Watch-OA"
    if "watch" in s and ("query" in s or "-qa-" in s): return "Watch-QA"
    if "agentghost" in s or "aitz" in s: return "AgentGhost"
    if "vab" in s: return "BEAT-VAB"
    if "-eb-" in s or "checkpoint-eb" in s: return "BEAT-EB"
    return "other"
def backend(m):
    s = str(m).lower()
    if "gemma4" in s or "gemma-4" in s: return "Gemma4"
    if "glm" in s: return "GLM"
    if "lfm" in s: return "LFM2.5"
    if "qwen3.5" in s or "qwen35" in s or "-9b" in s or re.search(r"\b9b\b", s): return "9B"
    if "7b" in s: return "7B"
    if "3b" in s: return "3B"
    return "?"

data = {}  # mid -> {method: auroc}
def put(mid, meth, a):
    if a is None or not isinstance(a, (int, float)) or (isinstance(a, float) and math.isnan(a)): return
    data.setdefault(mid, {})[meth] = float(a)

# 1) tracking sheet
wb = openpyxl.load_workbook(os.path.join(ROOT, "Backdoor Auditing Resources Tracking.xlsx"), data_only=True)
ws = wb["Runtime Results"]; rows = [[c.value for c in r] for r in ws.iter_rows()]
rows = [r for r in rows if any(str(x).strip() for x in r if x is not None)]
h = {str(x).strip(): i for i, x in enumerate(rows[0])}
def nm(m):
    m = str(m).lower()
    return "ConfGuard" if "conf" in m else "MAD" if ("mad" in m or "mahal" in m) else "LCF" if ("lcf" in m or "layerwise" in m) else None
for r in rows[1:]:
    mt = nm(r[h["Method"]]);
    if mt: put(r[h["Model ID"]], mt, r[h.get("AUROC")])
# 2) local RunPod JSONs (override/add)
L = os.path.join(SP, "results", "newfam")
for p in glob.glob(os.path.join(L, "confguard", "*.json")):
    d = json.load(open(p)); put(os.path.basename(p)[10:-5], "ConfGuard", d.get("confguard_auroc_maxrun"))
for p in glob.glob(os.path.join(L, "mad", "*.json")):
    d = json.load(open(p)); put(os.path.basename(p)[4:-5], "MAD", d.get("fixed_depth_auroc_mahalanobis"))
for p in glob.glob(os.path.join(L, "lcf", "*.json")):
    d = json.load(open(p)); put(os.path.basename(p)[4:-5], "LCF", d.get("auroc"))

# organize by family
famd = {f: {} for f in ORDER}
for mid, d in data.items():
    f = fam(mid)
    if f in famd: famd[f][mid] = d

# ---- master heatmap: family x method (mean AUROC) ----
M = np.full((len(ORDER), 3), np.nan); N = np.zeros((len(ORDER), 3), int)
for i, f in enumerate(ORDER):
    for j, meth in enumerate(["ConfGuard", "MAD", "LCF"]):
        xs = [d[meth] for d in famd[f].values() if meth in d]
        if xs: M[i, j] = np.mean(xs); N[i, j] = len(xs)
fig, ax = plt.subplots(figsize=(6.2, 6.4))
im = ax.imshow(M, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["ConfGuard", "MAD", "LCF"])
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER)
for i in range(len(ORDER)):
    for j in range(3):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i,j]:.2f}\n(n{N[i,j]})", ha="center", va="center", fontsize=9,
                    color="#111" if M[i, j] > 0.55 else "#fff")
ax.set_title("Mean AUROC by backdoor type and method\n(backdoored installs across backends)")
fig.colorbar(im, ax=ax, shrink=0.6, label="mean AUROC")
fig.savefig(os.path.join(OUT, "bytype_heatmap.png")); plt.close(fig)

# ---- master grouped bars ----
fig, ax = plt.subplots(figsize=(11, 5.4))
y = np.arange(len(ORDER))[::-1]; hbar = 0.26
for k, meth in enumerate(["ConfGuard", "MAD", "LCF"]):
    vals = [np.nanmean([d[meth] for d in famd[f].values() if meth in d]) if any(meth in d for d in famd[f].values()) else np.nan for f in ORDER]
    ax.barh(y + (1 - k) * hbar, vals, height=hbar, color=C[meth], label=meth, edgecolor="white", linewidth=0.6)
    for yi, v in zip(y + (1 - k) * hbar, vals):
        if not np.isnan(v): ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color="#333")
ax.axvline(0.5, color="#888", ls="--", lw=1); ax.text(0.5, len(ORDER) - 0.3, "chance", color="#888", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(ORDER); ax.set_xlim(0, 1.12); ax.set_xlabel("mean AUROC")
ax.set_title("Each method on each backdoor type"); ax.legend(loc="lower right", frameon=False)
fig.savefig(os.path.join(OUT, "bytype_grouped.png")); plt.close(fig)

# ---- per-family grouped bars (installs x method) ----
for f in ORDER:
    insts = sorted(famd[f].items(), key=lambda kv: backend(kv[0]))
    if not insts: continue
    labels = [backend(mid) + "\n" + (mid[:16]) for mid, _ in insts]
    x = np.arange(len(insts)); w = 0.26
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(insts) + 2), 4.6))
    for k, meth in enumerate(["ConfGuard", "MAD", "LCF"]):
        vals = [d.get(meth, np.nan) for _, d in insts]
        ax.bar(x + (k - 1) * w, vals, width=w, color=C[meth], label=meth, edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="#888", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_ylim(0, 1.05); ax.set_ylabel("AUROC"); ax.set_title(f"{f} — each method per install")
    ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=9)
    fig.savefig(os.path.join(OUT, f"bytype_{f.replace('/','-')}.png")); plt.close(fig)

print("wrote figures to", OUT)
for fn in sorted(os.listdir(OUT)):
    if fn.endswith(".png"): print("  ", fn)
