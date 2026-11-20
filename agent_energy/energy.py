"""GPU energy meters.

NvmlMeter reads the GPU's cumulative energy counter (mJ) when the driver
exposes it (Volta and newer, most desktop and datacenter cards). Many laptop
GPUs do not, so it falls back to sampling instantaneous power and integrating.

SmiSampler runs ``nvidia-smi`` in parallel as an independent cross-check.
NullMeter returns zeros so the harness can be tested on CPU.

All meters measure GPU board energy only: CPU, DRAM and the rest of the
system are not counted.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time


class NullMeter:
    mode = "none"

    def start(self) -> None:
        pass

    def stop(self) -> float:
        return 0.0

    def idle_power(self, seconds: float = 0.0) -> float:
        return 0.0

    def close(self) -> None:
        pass


class NvmlMeter:
    """Energy in joules between start() and stop()."""

    def __init__(self, device_index: int = 0, mode: str = "auto", sample_hz: float = 50.0):
        import pynvml  # provided by the nvidia-ml-py package

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.device_name = pynvml.nvmlDeviceGetName(self._h)
        if isinstance(self.device_name, bytes):
            self.device_name = self.device_name.decode()
        self._period = 1.0 / sample_hz

        if mode in ("auto", "counter"):
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h)
                self.mode = "counter"
            except pynvml.NVMLError:
                if mode == "counter":
                    raise
                self.mode = "power"
        else:
            self.mode = "power"

        if self.mode == "power":
            try:
                pynvml.nvmlDeviceGetPowerUsage(self._h)
            except pynvml.NVMLError as exc:
                raise RuntimeError(
                    "This GPU exposes neither an energy counter nor power readings through NVML. "
                    "Energy cannot be measured on it."
                ) from exc

        self._t0 = 0.0
        self._e0 = 0
        self._samples: list[tuple[float, float]] = []
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    # power sampling -------------------------------------------------------
    def _sample_loop(self) -> None:
        nv = self._nvml
        while not self._stop_evt.is_set():
            self._samples.append((time.perf_counter(), nv.nvmlDeviceGetPowerUsage(self._h) / 1000.0))
            time.sleep(self._period)

    # public API -----------------------------------------------------------
    def start(self) -> None:
        if self.mode == "counter":
            self._e0 = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._h)
        else:
            self._samples = []
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self.mode == "counter":
            return (self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._h) - self._e0) / 1000.0
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join()
        t_end = time.perf_counter()
        s = self._samples
        if not s:
            return 0.0
        if len(s) == 1:
            return s[0][1] * (t_end - self._t0)
        energy = 0.0
        for (ta, pa), (tb, pb) in zip(s, s[1:]):
            energy += 0.5 * (pa + pb) * (tb - ta)
        # extend the first and last sample to the window edges
        energy += s[0][1] * (s[0][0] - self._t0) + s[-1][1] * (t_end - s[-1][0])
        return energy

    def idle_power(self, seconds: float = 5.0) -> float:
        """Average power (W) while the GPU sits idle."""
        if seconds <= 0:
            return 0.0
        self.start()
        time.sleep(seconds)
        return self.stop() / seconds

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


class SmiSampler:
    """Independent cross-check: integrate power.draw reported by nvidia-smi."""

    def __init__(self, device_index: int = 0, interval_ms: int = 100):
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi not found on PATH")
        self._cmd = [
            "nvidia-smi", f"--id={device_index}", "--query-gpu=power.draw",
            "--format=csv,noheader,nounits", f"-lms={interval_ms}",
        ]
        self._samples: list[tuple[float, float]] = []
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            try:
                self._samples.append((time.perf_counter(), float(line.strip())))
            except ValueError:
                continue  # "[N/A]" on GPUs that do not report power

    def start(self) -> None:
        self._samples = []
        self._proc = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, text=True)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def stop(self) -> float:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
        if self._reader is not None:
            self._reader.join(timeout=2)
        s = self._samples
        return sum(0.5 * (pa + pb) * (tb - ta) for (ta, pa), (tb, pb) in zip(s, s[1:]))
