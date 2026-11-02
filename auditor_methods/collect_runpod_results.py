#!/usr/bin/env python3
"""Pull monitor result JSONs from the RunPod pod and write a single Excel results
file: Model ID | Method | Threshold | AUROC | AUPRC | TPR | FPR | Notes.
Re-runnable: overwrites the sheet from whatever JSONs are present. Robust to
partially-written / missing fields."""
import os, json, glob, subprocess, math

POD_IP = "31.24.80.40"; POD_PORT = "18172"; KEY = os.path.expanduser("~/.ssh/id_ed25519")
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, "results", "newfam")
XLSX = "/Users/muahmed/Desktop/Projects/backdoor-coding-agents/auditor_methods/NEW_MODELS_MONITOR_RESULTS.xlsx"

def pull():
    os.makedirs(LOCAL, exist_ok=True)
    cmd = ["scp", "-P", POD_PORT, "-i", KEY, "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=25", "-o", "StrictHostKeyChecking=accept-new", "-r",
           f"root@{POD_IP}:/workspace/outputs/newfam/.", LOCAL]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("scp warn:", r.stderr.strip()[:300])

def f(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None

def row_confguard(d, mid):
    note = f"bench={d.get('bench')}; n={d.get('n_samples')}; pos_rate={d.get('positive_rate')}"
    pg = d.get("per_group") or {}
    if pg:
        note += "; groups=" + ",".join(f"{k}:{v.get('n')}(lbl{v.get('label_rate')})" for k, v in pg.items())
    return [d.get("model_id") or mid, "ConfGuard", "L=10,P=0.99",
            f(d.get("confguard_auroc_maxrun")), f(d.get("confguard_auprc_maxrun")),
            f(d.get("confguard_tpr")), f(d.get("confguard_fpr")), note]

def _op_pick(ops, want=0.05):
    """MAD operating_points_fixed may be a dict keyed by alpha, or a list of dicts."""
    if isinstance(ops, dict):
        for k in (str(want), f"{want:.2f}", want):
            if k in ops: return ops[k]
        # dict keyed arbitrarily: take the entry whose alpha≈want
        for v in ops.values():
            if isinstance(v, dict) and abs(f(v.get("alpha") or want) - want) < 1e-6: return v
        return next(iter(ops.values()), {})
    if isinstance(ops, list):
        for v in ops:
            if isinstance(v, dict) and v.get("alpha") is not None and abs(f(v["alpha"]) - want) < 1e-6: return v
        return ops[0] if ops else {}
    return {}

def row_mad(d, mid):
    op = _op_pick(d.get("operating_points_fixed"), 0.05)
    tau = f(op.get("tau"))
    note = (f"bench={d.get('bench')}; blind L{d.get('fixed_depth_layer')} "
            f"(best L{d.get('best_layer_mahalanobis')} AUROC={f(d.get('best_auroc_mahalanobis'))}); "
            f"n_pos={d.get('n_positive')}; FPR_clean={f(op.get('fpr_clean'))}; @5%budget; ref={d.get('reference_repo')}")
    return [d.get("model_id") or mid, "MAD", (f"tau={tau:.3f}" if tau is not None else "n/a"),
            f(d.get("fixed_depth_auroc_mahalanobis")), f(d.get("fixed_depth_auprc_mahalanobis")),
            f(op.get("tpr")), f(op.get("fpr_all_neg")), note]

def row_lcf(d, mid):
    ops = d.get("operating_points") or {}
    tgt = d.get("target_fpr") or 0.10
    op = {}
    if isinstance(ops, dict):
        for k in (f"{tgt:.2f}", str(tgt), "0.10"):
            if k in ops: op = ops[k]; break
        if not op and ops: op = next(iter(ops.values()))
    elif isinstance(ops, list) and ops:
        op = ops[0]
    tau = f(op.get("tau"))
    note = (f"bench={d.get('bench')}; target_fpr={tgt}; n_ref={d.get('n_reference')}; "
            f"n_pos={d.get('n_positive')}; FPR_clean_loo={f(op.get('fpr_clean_loo'))}; benign_ctrl={d.get('benign_control')}")
    return [d.get("model_id") or mid, "LCF", (f"tau={tau:.3f}" if tau is not None else "n/a"),
            f(d.get("auroc")), f(d.get("auprc")), f(op.get("tpr")), f(op.get("fpr")), note]

PARSERS = {"confguard": row_confguard, "mad": row_mad, "lcf": row_lcf}

def collect_rows():
    rows = []
    for meth, parse in PARSERS.items():
        for p in sorted(glob.glob(os.path.join(LOCAL, meth, f"{meth}_*.json"))):
            mid = os.path.basename(p)[len(meth) + 1:-5]  # strip "<meth>_" and ".json"
            try:
                d = json.load(open(p))
                rows.append(parse(d, mid))
            except Exception as e:
                print("parse fail", p, repr(e))
    # order: by model id then method
    rows.sort(key=lambda r: (str(r[0]), r[1]))
    return rows

def write_xlsx(rows):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    wb = Workbook(); ws = wb.active; ws.title = "Monitor results"
    cols = ["Model ID", "Method", "Threshold", "AUROC", "AUPRC", "TPR", "FPR", "Notes"]
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496"); c.alignment = Alignment(vertical="center")
    for r in rows:
        ws.append(r)
    for row in ws.iter_rows(min_row=2):
        for i, c in enumerate(row):
            c.font = Font(name="Arial")
            if i in (3, 4, 5, 6) and isinstance(c.value, (int, float)):
                c.number_format = "0.000"
    widths = [40, 11, 14, 9, 9, 8, 8, 70]
    from openpyxl.utils import get_column_letter
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    wb.save(XLSX)
    return len(rows)

if __name__ == "__main__":
    pull()
    rows = collect_rows()
    n = write_xlsx(rows)
    print(f"wrote {XLSX} with {n} result rows")
    for r in rows:
        print(" ", r[0], r[1], "AUROC=", r[3], "TPR=", r[5], "FPR=", r[6])
