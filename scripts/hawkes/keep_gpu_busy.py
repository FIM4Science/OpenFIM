import sys
import time

import torch


def occupy_gpu():
    """
    Allocates a tiny tensor on the GPU to keep it occupied.
    This prevents Slurm from deallocating the GPU during idle periods
    between subprocess calls in a larger orchestration script.
    """
    if not torch.cuda.is_available():
        print("[GPU Occupier] CUDA not available. Exiting.", file=sys.stderr)
        return

    # Respect CUDA_VISIBLE_DEVICES set by Slurm
    device = torch.device("cuda")

    print(f"[GPU Occupier] Holding onto GPU: {torch.cuda.get_device_name(device)}", flush=True)

    try:
        # Allocate a tiny tensor. This is all that's needed to hold the resource.
        _ = torch.ones(1, device=device)

        # Loop indefinitely, sleeping for long intervals.
        # The process just needs to exist and hold the memory.
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[GPU Occupier] Exiting on KeyboardInterrupt.", flush=True)
    except Exception as e:
        print(f"[GPU Occupier] An error occurred: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    occupy_gpu()
