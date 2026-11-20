"""Baseline: reproduce one Fernandez et al. (ACL 2025) trend on this GPU.

Energy per generated token should fall as batch size grows, because each
memory-bound decode step reads the weights once for the whole batch.

    python run_baseline.py --batch 1 4 16 --repeats 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

from agent_energy.energy import NullMeter, NvmlMeter
from agent_energy.models import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--idle-seconds", type=float, default=5.0)
    p.add_argument("--out", default="results/baseline.csv")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA is not available.")
    meter = NvmlMeter(args.gpu) if device.type == "cuda" else NullMeter()
    model = load_model(args.model, args.dtype, device)
    vocab = model.config.vocab_size
    pad_id = model.config.eos_token_id if model.config.eos_token_id is not None else 0
    if isinstance(pad_id, list):
        pad_id = pad_id[0]

    idle_w = meter.idle_power(args.idle_seconds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(args.seed)

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "repeat", "gen_tokens", "energy_j", "seconds", "idle_w",
                    "j_per_token", "j_per_token_net", "meter_mode"])
        for b in args.batch:
            ids = torch.randint(100, vocab, (b, args.prompt_len), generator=g).to(device)
            mask = torch.ones_like(ids)
            model.generate(ids[:, :32], attention_mask=mask[:, :32], max_new_tokens=8,
                           min_new_tokens=8, do_sample=False, pad_token_id=pad_id)  # warm-up
            for r in range(args.repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                meter.start()
                t0 = time.perf_counter()
                model.generate(ids, attention_mask=mask, max_new_tokens=args.gen, min_new_tokens=args.gen,
                               do_sample=False, pad_token_id=pad_id)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                s = time.perf_counter() - t0
                e = meter.stop()
                n = b * args.gen
                w.writerow([b, r, n, round(e, 4), round(s, 4), round(idle_w, 3),
                            round(e / n, 6), round((e - idle_w * s) / n, 6), meter.mode])
                f.flush()
                print(f"batch={b:>3} rep={r} {e:.2f} J  {s:.2f} s  {e / n * 1000:.2f} mJ/token")
    meter.close()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
