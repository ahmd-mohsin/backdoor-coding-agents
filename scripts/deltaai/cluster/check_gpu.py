"""Sanity check for a DeltaAI GPU job.

Prints the job's environment and times a bf16 matmul on every visible GPU.
Run it as a job, from the repo root:

    scripts/deltaai/cluster/submit.sh 1gpu.sbatch scripts/deltaai/cluster/check_gpu.py
"""

import os
import platform
import socket
import sys
import time

import torch


def main() -> None:
    print(f"host={socket.gethostname()} arch={platform.machine()} python={sys.version.split()[0]}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}")
    for key in ("SLURM_JOB_ID", "SLURM_JOB_ACCOUNT", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
                "CUDA_VISIBLE_DEVICES", "DTAI_RUN_DIR", "HF_HOME"):
        print(f"{key}={os.environ.get(key, '')}")

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available. Run this as a job on a compute node, not on a login node.")

    n = 8192
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        props = torch.cuda.get_device_properties(device)
        x = torch.randn(n, n, device=device, dtype=torch.bfloat16)
        x @ x  # warm-up
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(10):
            x @ x
        torch.cuda.synchronize(device)
        seconds = (time.perf_counter() - start) / 10
        print(f"cuda:{i} {props.name}, {props.total_memory / 2**30:.0f} GiB, "
              f"bf16 matmul {2 * n**3 / seconds / 1e12:.0f} TFLOP/s")
    print(f"NCCL available: {torch.distributed.is_nccl_available()}")


if __name__ == "__main__":
    main()
