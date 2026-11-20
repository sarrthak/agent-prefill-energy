"""Plot measured results against the analytical prediction.

    python plot_results.py                 # reads results/agent_runs.csv (and results/baseline.csv if present)
    python plot_results.py --energy raw    # use raw instead of idle-subtracted energy

Writes results/prefill_vs_turns.png, results/ratio_vs_turns.png,
results/baseline.png and results/summary.csv, and prints a Markdown table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agent_energy.predict import token_ratio

CRIMSON = "#6D0808"
GREY = "#8A8478"


def summarize(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(["turns", "cache"])
    s = g.agg(
        n=(col, "size"),
        prefill_tokens=("prefill_tokens", "mean"),
        predicted_tokens=("predicted_prefill_tokens", "first"),
        all_tokens_match=("tokens_match", "all"),
        prefill_j=(col, "mean"),
        prefill_j_std=(col, "std"),
        task_s=("task_s", "mean"),
    ).reset_index()
    return s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="results/agent_runs.csv")
    p.add_argument("--baseline", default="results/baseline.csv")
    p.add_argument("--energy", choices=["net", "raw"], default="net")
    args = p.parse_args()

    runs = Path(args.runs)
    outdir = runs.parent
    df = pd.read_csv(runs)
    df["tokens_match"] = df["tokens_match"].astype(str).str.lower().eq("true")
    col = "prefill_j_net" if args.energy == "net" else "prefill_j"
    s = summarize(df, col)
    s.to_csv(outdir / "summary.csv", index=False)

    c0, delta, gen = int(df["c0"].iloc[0]), int(df["delta"].iloc[0]), int(df["gen"].iloc[0])
    turns = sorted(df["turns"].unique())

    # --- figure 1: prefill energy vs turns ---------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for cache, color, label in [("off", GREY, "cache off"), ("on", CRIMSON, "cache on")]:
        d = s[s["cache"] == cache].sort_values("turns")
        if d.empty:
            continue
        ax.errorbar(d["turns"], d["prefill_j"], yerr=d["prefill_j_std"].fillna(0), marker="o",
                    color=color, capsize=4, label=f"{label} (measured)")
    ax.set_xlabel("turns per task (T)")
    ax.set_ylabel(f"GPU prefill energy per task, J ({args.energy})")
    ax.set_title(f"Prefill energy vs turns  (C0={c0}, Δ={delta}, gen={gen})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "prefill_vs_turns.png", dpi=200)

    # --- figure 2: off/on ratio, measured vs predicted ---------------------
    on = s[s["cache"] == "on"].set_index("turns")
    off = s[s["cache"] == "off"].set_index("turns")
    both = sorted(set(on.index) & set(off.index))
    rows = []
    for T in both:
        rows.append({
            "turns": T,
            "predicted_token_ratio": token_ratio(T, c0, delta, gen),
            "measured_token_ratio": off.loc[T, "prefill_tokens"] / on.loc[T, "prefill_tokens"],
            "measured_energy_ratio": (off.loc[T, "prefill_j"] / on.loc[T, "prefill_j"]) if on.loc[T, "prefill_j"] > 0 else float("nan"),
        })
    r = pd.DataFrame(rows)
    if not r.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(r["turns"], r["predicted_token_ratio"], "--", color=GREY, marker="s", label="token ratio (LLM-predicted)")
        ax.plot(r["turns"], r["measured_token_ratio"], ":", color="black", marker="x", label="token ratio (measured)")
        ax.plot(r["turns"], r["measured_energy_ratio"], "-", color=CRIMSON, marker="o", label="energy ratio (measured)")
        ax.set_xlabel("turns per task (T)")
        ax.set_ylabel("cache off / cache on")
        ax.set_title("Does energy follow tokens?")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "ratio_vs_turns.png", dpi=200)
        r.to_csv(outdir / "ratios.csv", index=False)

    # --- figure 3: baseline -------------------------------------------------
    bpath = Path(args.baseline)
    if bpath.exists():
        b = pd.read_csv(bpath)
        bcol = "j_per_token_net" if args.energy == "net" else "j_per_token"
        bs = b.groupby("batch")[bcol].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(bs["batch"], bs["mean"] * 1000, yerr=bs["std"].fillna(0) * 1000, marker="o", color=CRIMSON, capsize=4)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch size")
        ax.set_ylabel(f"mJ per generated token ({args.energy})")
        ax.set_title("Baseline: energy per token vs batch (measured)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "baseline.png", dpi=200)

    # --- table -----------------------------------------------------------------
    print("| T | tokens off (meas / pred) | tokens on (meas / pred) | J off | J on | energy ratio | predicted token ratio |")
    print("|---|---|---|---|---|---|---|")
    for T in turns:
        def cell(cache, key):
            d = s[(s["turns"] == T) & (s["cache"] == cache)]
            return d[key].iloc[0] if not d.empty else float("nan")
        e_off, e_on = cell("off", "prefill_j"), cell("on", "prefill_j")
        ratio = e_off / e_on if e_on and e_on == e_on and e_on > 0 else float("nan")
        print(f"| {T} | {cell('off', 'prefill_tokens'):.0f} / {cell('off', 'predicted_tokens'):.0f} "
              f"| {cell('on', 'prefill_tokens'):.0f} / {cell('on', 'predicted_tokens'):.0f} "
              f"| {e_off:.2f} | {e_on:.2f} | {ratio:.2f} | {token_ratio(T, c0, delta, gen):.2f} |")
    if not df["tokens_match"].all():
        print("\nWARNING: some runs processed a different number of prefill tokens than predicted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
