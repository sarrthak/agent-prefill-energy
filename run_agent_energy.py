"""Sweep the agent loop over turn counts with prefix caching on and off.

Example (the run shown in the slides):
    python run_agent_energy.py --turns 5 10 20 40 --cache on off --repeats 5

Quick check on CPU without a GPU or model download:
    python run_agent_energy.py --model tiny --device cpu --c0 64 --delta 16 --gen 4 --turns 2 4 --repeats 1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch

from agent_energy.energy import NullMeter, NvmlMeter, SmiSampler
from agent_energy.loop import TaskConfig, run_task
from agent_energy.models import load_model
from agent_energy.predict import prefill_tokens

FIELDS = [
    "turns", "cache", "repeat", "c0", "delta", "gen",
    "prefill_tokens", "predicted_prefill_tokens", "tokens_match",
    "prefill_j", "decode_j", "task_j", "prefill_s", "decode_s", "task_s",
    "idle_w", "prefill_j_net", "task_j_net", "smi_task_j", "meter_mode",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model id, or 'tiny' for a random test model")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu", type=int, default=0, help="GPU index for NVML")
    p.add_argument("--turns", type=int, nargs="+", default=[5, 10, 20, 40])
    p.add_argument("--cache", nargs="+", choices=["on", "off"], default=["on", "off"])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--c0", type=int, default=2000)
    p.add_argument("--delta", type=int, default=500)
    p.add_argument("--gen", type=int, default=32)
    p.add_argument("--warmup", type=int, default=1, help="untimed warm-up tasks before measuring")
    p.add_argument("--idle-seconds", type=float, default=5.0, help="idle-power measurement window")
    p.add_argument("--smi-check", action="store_true", help="cross-check energy with nvidia-smi")
    p.add_argument("--meter", choices=["auto", "counter", "power"], default="auto")
    p.add_argument("--out", default="results/agent_runs.csv")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA is not available. Install a CUDA build of PyTorch, or use --device cpu --model tiny to test.")

    meter = NvmlMeter(args.gpu, args.meter) if device.type == "cuda" else NullMeter()
    smi = SmiSampler(args.gpu) if args.smi_check else None
    model = load_model(args.model, args.dtype, device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    calls_path = out.with_name(out.stem + "_calls.csv")
    meta = {
        "model": args.model, "dtype": args.dtype, "device": str(device),
        "gpu_name": getattr(meter, "device_name", None), "meter_mode": meter.mode,
        "torch": torch.__version__, "args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    for _ in range(args.warmup):
        run_task(model, TaskConfig(turns=2, c0=args.c0, delta=args.delta, gen=args.gen, seed=12345),
                 cache=True, meter=NullMeter(), device=device)

    idle_w = meter.idle_power(args.idle_seconds)
    print(f"idle power: {idle_w:.2f} W  (meter mode: {meter.mode})")

    # interleave settings in a shuffled order so drift (heat, clocks) hits all of them
    jobs = [(T, c, r) for r in range(args.repeats) for T in args.turns for c in args.cache]
    random.Random(args.seed).shuffle(jobs)

    with out.open("w", newline="") as f, calls_path.open("w", newline="") as fc:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        wc = csv.writer(fc)
        wc.writerow(["turns", "cache", "repeat", "turn", "context_len", "prefill_tokens",
                     "prefill_j", "decode_j", "prefill_s", "decode_s"])

        for i, (T, c, r) in enumerate(jobs, 1):
            cache = c == "on"
            cfg = TaskConfig(turns=T, c0=args.c0, delta=args.delta, gen=args.gen, seed=args.seed + r)
            if smi:
                smi.start()
            res = run_task(model, cfg, cache=cache, meter=meter, device=device)
            smi_j = smi.stop() if smi else None

            predicted = prefill_tokens(T, args.c0, args.delta, args.gen, cache=cache)
            task_s = res.prefill_s + res.decode_s
            task_j = res.prefill_j + res.decode_j
            row = {
                "turns": T, "cache": c, "repeat": r, "c0": args.c0, "delta": args.delta, "gen": args.gen,
                "prefill_tokens": res.prefill_tokens, "predicted_prefill_tokens": predicted,
                "tokens_match": res.prefill_tokens == predicted,
                "prefill_j": round(res.prefill_j, 4), "decode_j": round(res.decode_j, 4), "task_j": round(task_j, 4),
                "prefill_s": round(res.prefill_s, 4), "decode_s": round(res.decode_s, 4), "task_s": round(task_s, 4),
                "idle_w": round(idle_w, 3),
                "prefill_j_net": round(res.prefill_j - idle_w * res.prefill_s, 4),
                "task_j_net": round(task_j - idle_w * task_s, 4),
                "smi_task_j": None if smi_j is None else round(smi_j, 4),
                "meter_mode": meter.mode,
            }
            w.writerow(row)
            f.flush()
            for call in res.calls:
                wc.writerow([T, c, r, call.turn, call.context_len, call.prefill_tokens,
                             round(call.prefill_j, 4), round(call.decode_j, 4),
                             round(call.prefill_s, 4), round(call.decode_s, 4)])
            fc.flush()

            flag = "ok" if row["tokens_match"] else "MISMATCH"
            print(f"[{i}/{len(jobs)}] T={T:>3} cache={c:<3} rep={r} prefill_tokens={res.prefill_tokens} "
                  f"(pred {predicted}, {flag})  prefill={res.prefill_j:.2f} J  task={task_j:.2f} J  {task_s:.2f} s")

    meter.close()
    print(f"\nwrote {out} and {calls_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
