"""Verification of the harness itself, on a tiny random model on CPU.

1. The number of tokens actually prefilled equals the analytical prediction.
2. Reusing the KV cache does not change what the model generates: greedy
   outputs with cache on and cache off are identical, so the two modes do the
   same work and differ only in recomputation.
"""

import pytest
import torch

from agent_energy.energy import NullMeter
from agent_energy.loop import TaskConfig, run_task
from agent_energy.models import tiny_model
from agent_energy.predict import per_call_prefill, prefill_tokens

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def model():
    return tiny_model().to(DEVICE)


@pytest.mark.parametrize("gen", [0, 1, 4])
@pytest.mark.parametrize("cache", [False, True])
def test_prefill_tokens_match_prediction(model, cache, gen):
    cfg = TaskConfig(turns=4, c0=24, delta=8, gen=gen, seed=1)
    res = run_task(model, cfg, cache=cache, meter=NullMeter(), device=DEVICE)
    assert [c.prefill_tokens for c in res.calls] == per_call_prefill(4, 24, 8, gen, cache)
    assert res.prefill_tokens == prefill_tokens(4, 24, 8, gen, cache)
    assert all(c.decode_steps == gen for c in res.calls)


def test_context_grows_as_modelled(model):
    cfg = TaskConfig(turns=3, c0=20, delta=5, gen=3, seed=2)
    res = run_task(model, cfg, cache=True, meter=NullMeter(), device=DEVICE)
    assert [c.context_len for c in res.calls] == [20 + t * 5 + (t - 1) * 3 for t in (1, 2, 3)]


def test_cache_does_not_change_outputs(model):
    cfg = TaskConfig(turns=5, c0=32, delta=8, gen=6, seed=3)
    on = run_task(model, cfg, cache=True, meter=NullMeter(), device=DEVICE)
    off = run_task(model, cfg, cache=False, meter=NullMeter(), device=DEVICE)
    assert on.generated == off.generated
