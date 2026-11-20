import pytest

from agent_energy.predict import per_call_prefill, prefill_tokens, token_ratio


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("gen", [0, 32])
@pytest.mark.parametrize("T", [1, 5, 10, 20, 40])
def test_closed_form_matches_per_call_sum(T, gen, cache):
    assert prefill_tokens(T, 2000, 500, gen, cache) == sum(per_call_prefill(T, 2000, 500, gen, cache))


def test_slide_numbers():
    # values shown on the research-question slide (gen = 0)
    expected = {5: (17_500, 4_500), 10: (47_500, 7_000), 20: (145_000, 12_000), 40: (490_000, 22_000)}
    for T, (off, on) in expected.items():
        assert prefill_tokens(T, 2000, 500, 0, cache=False) == off
        assert prefill_tokens(T, 2000, 500, 0, cache=True) == on
    ratios = [round(token_ratio(T, 2000, 500, 0), 1) for T in (5, 10, 20, 40)]
    assert ratios == [3.9, 6.8, 12.1, 22.3]


def test_cache_on_is_linear_cache_off_is_quadratic():
    on = [prefill_tokens(T, 2000, 500, 32, True) for T in (10, 20, 40)]
    off = [prefill_tokens(T, 2000, 500, 32, False) for T in (10, 20, 40)]
    # doubling T roughly doubles cache-on cost, but roughly quadruples the quadratic term of cache-off
    assert on[2] - on[1] == 2 * (on[1] - on[0])
    assert (off[2] - off[1]) > 2 * (off[1] - off[0])
