#!/usr/bin/env python3
"""Build a SEPARATE Excel for the Round-2 batch (the 44 models run this session),
reusing collect_results' JSON parsers. Reads the already-pulled JSONs in
scratchpad/results/newfam. Output columns identical to the main sheet."""
import os, glob, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_results as cr

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, "results", "newfam")
XLSX = "/Users/muahmed/Desktop/Projects/backdoor-coding-agents/docs/report/ZOO_ROUND2_RESULTS.xlsx"
MIDS = set(l.strip() for l in open(os.path.join(HERE, "batch_mids.txt")) if l.strip())

def rows():
    out = []
    for meth, parse in cr.PARSERS.items():
        for p in sorted(glob.glob(os.path.join(LOCAL, meth, f"{meth}_*.json"))):
            mid = os.path.basename(p)[len(meth) + 1:-5]
            if mid not in MIDS:
                continue
            try:
                out.append(parse(json.load(open(p)), mid))
            except Exception as e:
                print("parse fail", p, repr(e))
    out.sort(key=lambda r: (str(r[0]), r[1]))
    return out

def write(rows):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    wb = Workbook(); ws = wb.active; ws.title = "Zoo round-2 results"
    cols = ["Model ID", "Method", "Threshold", "AUROC", "AUPRC", "TPR", "FPR", "Notes"]
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496"); c.alignment = Alignment(vertical="center")
    for r in rows: ws.append(r)
    for row in ws.iter_rows(min_row=2):
        for i, c in enumerate(row):
            c.font = Font(name="Arial")
            if i in (3, 4, 5, 6) and isinstance(c.value, (int, float)):
                c.number_format = "0.000"
    for i, w in enumerate([46, 11, 15, 9, 9, 8, 8, 72], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"; wb.save(XLSX)
    return len(rows)

if __name__ == "__main__":
    rs = rows(); n = write(rs)
    print(f"wrote {XLSX} with {n} rows (of {len(MIDS)} batch models)")
    # coverage: which batch mids have all 3 / partial / none
    from collections import defaultdict
    got = defaultdict(set)
    for r in rs: got[r[0]].add(r[1])
    full = [m for m in MIDS if len(got.get(m, set())) == 3]
    part = [(m, sorted(got[m])) for m in MIDS if 0 < len(got.get(m, set())) < 3]
    none = [m for m in MIDS if not got.get(m)]
    print(f"coverage: full(3)={len(full)}  partial={len(part)}  none={len(none)}")
    for m, g in part: print("  partial:", m, g)
    for m in none: print("  none:", m)
