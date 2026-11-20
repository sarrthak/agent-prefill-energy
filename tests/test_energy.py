"""Check the energy meter's arithmetic with a fake NVML (no GPU needed)."""

import sys
import time
import types

import pytest


def fake_pynvml(power_mw=50_000, has_counter=False):
    m = types.ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    state = {"energy_mj": 0, "t": time.perf_counter()}

    def counter(_h):
        if not has_counter:
            raise NVMLError("not supported")
        now = time.perf_counter()
        state["energy_mj"] += int(power_mw * (now - state["t"]))  # mW * s = mJ
        state["t"] = now
        return state["energy_mj"]

    m.NVMLError = NVMLError
    m.nvmlInit = lambda: None
    m.nvmlShutdown = lambda: None
    m.nvmlDeviceGetHandleByIndex = lambda i: object()
    m.nvmlDeviceGetName = lambda h: "Fake GPU"
    m.nvmlDeviceGetPowerUsage = lambda h: power_mw
    m.nvmlDeviceGetTotalEnergyConsumption = counter
    return m


@pytest.mark.parametrize("has_counter,mode", [(True, "counter"), (False, "power")])
def test_constant_power_integrates_to_power_times_time(monkeypatch, has_counter, mode):
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml(50_000, has_counter))
    from agent_energy.energy import NvmlMeter

    meter = NvmlMeter(mode="auto", sample_hz=200)
    assert meter.mode == mode
    meter.start()
    t0 = time.perf_counter()
    time.sleep(0.3)
    joules = meter.stop()
    elapsed = time.perf_counter() - t0
    assert joules == pytest.approx(50.0 * elapsed, rel=0.1)  # 50 W
