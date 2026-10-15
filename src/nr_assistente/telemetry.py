from __future__ import annotations

import csv
import subprocess
import threading
from pathlib import Path

GPU_FIELDS = (
    "timestamp",
    "name",
    "driver_version",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
)


def query_gpu() -> dict[str, str]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    values = next(csv.reader([completed.stdout.splitlines()[0]]))
    return {key: value.strip() for key, value in zip(GPU_FIELDS, values)}


class NvidiaMonitor:
    def __init__(self, path: Path, interval_seconds: float = 2.0):
        self.path = path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "NvidiaMonitor":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_seconds + 10)

    def _run(self) -> None:
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=GPU_FIELDS + ("error",))
            writer.writeheader()
            while not self._stop.is_set():
                try:
                    row = query_gpu()
                    row["error"] = ""
                except (OSError, subprocess.SubprocessError, StopIteration) as exc:
                    row = {field: "" for field in GPU_FIELDS}
                    row["error"] = f"{type(exc).__name__}: {exc}"
                writer.writerow(row)
                handle.flush()
                self._stop.wait(self.interval_seconds)


def summarize_gpu_csv(path: Path) -> dict[str, float | int | str | None]:
    if not path.exists():
        return {"samples": 0, "peak_vram_mib": None, "mean_utilization_pct": None}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid = [row for row in rows if row.get("memory.used") and not row.get("error")]
    if not valid:
        return {"samples": 0, "peak_vram_mib": None, "mean_utilization_pct": None}
    memory = [float(row["memory.used"]) for row in valid]
    utilization = [float(row["utilization.gpu"]) for row in valid]
    return {
        "samples": len(valid),
        "gpu_name": valid[0]["name"],
        "peak_vram_mib": max(memory),
        "mean_utilization_pct": sum(utilization) / len(utilization),
    }
