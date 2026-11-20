"""Analytical token model for a simulated agent loop.

The harness in ``loop.py`` runs T model calls. Before call t (t = 1..T) a tool
observation of ``delta`` tokens is appended to the context, the model prefills,
then decodes ``gen`` tokens which are also appended to the context.

Context length at call t:   C0 + t*delta + (t-1)*gen

Prefill tokens processed
  cache off (recompute everything each call):
      sum_t (C0 + t*delta + (t-1)*gen) = T*C0 + delta*T(T+1)/2 + gen*T(T-1)/2
  cache on (reuse the KV cache of the shared prefix):
      call 1 processes C0 + delta, every later call only the new delta
      = C0 + T*delta

With gen = 0 these reduce to the formulas on the slides:
  T*C0 + delta*T(T+1)/2   vs   C0 + T*delta
"""

from __future__ import annotations


def context_len(t: int, c0: int, delta: int, gen: int = 0) -> int:
    """Context length (tokens) seen by call t, 1-indexed."""
    if t < 1:
        raise ValueError("t starts at 1")
    return c0 + t * delta + (t - 1) * gen


def per_call_prefill(turns: int, c0: int, delta: int, gen: int = 0, cache: bool = False) -> list[int]:
    """Prefill tokens processed at each call."""
    out = []
    for t in range(1, turns + 1):
        if cache:
            out.append(c0 + delta if t == 1 else delta)
        else:
            out.append(context_len(t, c0, delta, gen))
    return out


def prefill_tokens(turns: int, c0: int, delta: int, gen: int = 0, cache: bool = False) -> int:
    """Total prefill tokens over a task, closed form."""
    T = turns
    if cache:
        return c0 + T * delta
    return T * c0 + delta * T * (T + 1) // 2 + gen * T * (T - 1) // 2


def token_ratio(turns: int, c0: int, delta: int, gen: int = 0) -> float:
    """Predicted (cache off) / (cache on) prefill-token ratio."""
    return prefill_tokens(turns, c0, delta, gen, cache=False) / prefill_tokens(turns, c0, delta, gen, cache=True)
