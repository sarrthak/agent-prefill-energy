"""Simulated ReAct-style agent loop with prefix (KV) caching on or off.

Each call: append a synthetic tool observation of ``delta`` tokens, prefill,
then greedy-decode ``gen`` tokens and append them to the context.

cache on  -> the KV cache is kept across calls; only tokens not yet in the
             cache are prefilled.
cache off -> the KV cache is dropped after every call, so the whole context
             is prefilled again (what a stateless API call does).

Token ids are random (seeded), so no tokenizer is needed and the two modes see
exactly the same tool observations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class TaskConfig:
    turns: int
    c0: int = 2000      # system prompt + task description
    delta: int = 500    # tool observation appended before each call
    gen: int = 32       # tokens generated per call
    seed: int = 0


@dataclass
class CallRecord:
    turn: int
    context_len: int
    prefill_tokens: int
    decode_steps: int
    prefill_j: float
    decode_j: float
    prefill_s: float
    decode_s: float


@dataclass
class TaskResult:
    cache: bool
    calls: list[CallRecord] = field(default_factory=list)
    generated: list[list[int]] = field(default_factory=list)

    @property
    def prefill_tokens(self) -> int:
        return sum(c.prefill_tokens for c in self.calls)

    @property
    def prefill_j(self) -> float:
        return sum(c.prefill_j for c in self.calls)

    @property
    def decode_j(self) -> float:
        return sum(c.decode_j for c in self.calls)

    @property
    def prefill_s(self) -> float:
        return sum(c.prefill_s for c in self.calls)

    @property
    def decode_s(self) -> float:
        return sum(c.decode_s for c in self.calls)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _random_ids(n: int, vocab: int, gen: torch.Generator) -> torch.Tensor:
    # skip the lowest ids, which are usually special tokens
    low = min(100, vocab // 4)
    return torch.randint(low, vocab, (n,), generator=gen)


@torch.inference_mode()
def run_task(model, cfg: TaskConfig, cache: bool, meter, device: torch.device) -> TaskResult:
    vocab = model.config.vocab_size
    rng = torch.Generator().manual_seed(cfg.seed)

    context = _random_ids(cfg.c0, vocab, rng)
    past = None
    cached_len = 0  # number of context tokens whose K/V are in `past`
    result = TaskResult(cache=cache)

    for t in range(1, cfg.turns + 1):
        context = torch.cat([context, _random_ids(cfg.delta, vocab, rng)])

        if not cache:
            past = None
            cached_len = 0
        new_ids = context[cached_len:].unsqueeze(0).to(device)

        # ---- prefill ----
        _sync(device)
        meter.start()
        t0 = time.perf_counter()
        out = model(input_ids=new_ids, past_key_values=past, use_cache=True)
        _sync(device)
        prefill_s = time.perf_counter() - t0
        prefill_j = meter.stop()
        past = out.past_key_values
        tok = out.logits[0, -1].argmax()
        del out

        # ---- decode ----
        generated = []
        meter.start()
        t0 = time.perf_counter()
        for i in range(cfg.gen):
            generated.append(tok)
            # feeding the token puts its K/V in the cache; the last step only
            # completes the cache so the next call starts from a full prefix
            out = model(input_ids=tok.view(1, 1), past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[0, -1].argmax()
            del out
        _sync(device)
        decode_s = time.perf_counter() - t0
        decode_j = meter.stop()

        gen_ids = torch.stack(generated).cpu() if generated else torch.empty(0, dtype=context.dtype)
        result.calls.append(CallRecord(
            turn=t,
            context_len=int(context.numel()),
            prefill_tokens=int(new_ids.shape[1]),
            decode_steps=cfg.gen,
            prefill_j=prefill_j,
            decode_j=decode_j,
            prefill_s=prefill_s,
            decode_s=decode_s,
        ))
        result.generated.append(gen_ids.tolist())
        context = torch.cat([context, gen_ids.to(context.dtype)])
        cached_len = int(context.numel())

    return result
