"""Download a Hugging Face model or dataset into your Delta workspace, after checking it fits.

    python scripts/delta/cluster/download_model.py <org/name> [--revision REV]
        [--include PATTERN ...] [--dataset] [--min-free-gb N] [--yes]

Models go to $DELTA_MODELS/<org>/<name>, datasets to $DELTA_DATA/<org>/<name>.
Before downloading, it prints the size and how much of the project's /work/hdd quota
is left, and stops if the download would leave less than --min-free-gb (default 50)
free for the whole project. Each download is recorded in MANIFEST.tsv. Run it on a
login node (inside tmux for big downloads) after `delta_activate`. For gated or
private repos, log in first with `hf auth login`.
"""

import argparse
import datetime
import fnmatch
import os
import re
import subprocess
import sys

from huggingface_hub import HfApi, snapshot_download

GIB = 2**30
UNITS = {"": 1, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40, "P": 2**50}


def parse_size(text):
    match = re.fullmatch(r"([\d.]+)\s*([KMGTP]?)", text.strip(), re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1)) * UNITS[match.group(2).upper()]


def project_quota(path):
    """Return (used, soft_quota) in bytes from `quota` for the file system holding `path`."""
    try:
        out = subprocess.run(["quota"], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) >= 3 and cols[0].startswith("/") and path.startswith(cols[0].rstrip("/") + "/"):
            used, soft = parse_size(cols[1]), parse_size(cols[2])
            if used is not None and soft is not None:
                return used, soft
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="Hugging Face repo, e.g. Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--revision", help="branch, tag or commit (default: main)")
    parser.add_argument("--include", action="append", metavar="PATTERN",
                        help="only files matching this glob, e.g. '*.safetensors' (repeatable)")
    parser.add_argument("--dataset", action="store_true", help="the repo is a dataset")
    parser.add_argument("--min-free-gb", type=float, default=50,
                        help="refuse if fewer GiB than this would be left free (default 50)")
    parser.add_argument("--yes", action="store_true", help="download even below --min-free-gb")
    args = parser.parse_args()

    root = os.environ.get("DELTA_DATA" if args.dataset else "DELTA_MODELS")
    if not root:
        sys.exit("DELTA_MODELS/DELTA_DATA are not set: source scripts/delta/cluster/env.sh first.")
    repo_type = "dataset" if args.dataset else "model"

    api = HfApi()
    get_info = api.dataset_info if args.dataset else api.model_info
    info = get_info(args.repo_id, revision=args.revision, files_metadata=True)
    files = [s for s in info.siblings
             if not args.include or any(fnmatch.fnmatch(s.rfilename, p) for p in args.include)]
    size = sum(s.size or 0 for s in files)
    dest = os.path.join(root, *args.repo_id.split("/"))
    print(f"{args.repo_id} @ {info.sha[:12]}: {len(files)} files, {size / GIB:.1f} GiB -> {dest}")

    quota = project_quota(dest)
    if quota is None:
        print("Could not read the project quota; check it with `quota` before large downloads.")
    else:
        used, soft = quota
        free = soft - used
        print(f"Project quota: {used / GIB:.0f} of {soft / GIB:.0f} GiB used, {free / GIB:.0f} GiB free.")
        if free - size < args.min_free_gb * GIB and not args.yes:
            sys.exit(f"Stopping: this would leave under {args.min_free_gb:.0f} GiB free for the whole "
                     "project. Free up space, use --include, or re-run with --yes.")

    os.makedirs(dest, exist_ok=True)
    snapshot_download(args.repo_id, repo_type=repo_type, revision=info.sha,
                      local_dir=dest, allow_patterns=args.include)
    with open(os.path.join(root, "MANIFEST.tsv"), "a") as manifest:
        manifest.write("\t".join([datetime.date.today().isoformat(), repo_type, args.repo_id,
                                  info.sha, f"{size / GIB:.2f}GiB", dest]) + "\n")
    print(f"Done: {dest}")


if __name__ == "__main__":
    main()
