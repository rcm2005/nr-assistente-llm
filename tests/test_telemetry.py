from pathlib import Path

from src.nr_assistente.telemetry import summarize_gpu_csv


def test_summarize_gpu_csv(tmp_path: Path) -> None:
    path = tmp_path / "gpu.csv"
    path.write_text(
        "timestamp,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,error\n"
        '2026-01-01,\"GPU X\",610.88,8192,1024,50,60,100,\n'
        '2026-01-01,\"GPU X\",610.88,8192,2048,70,62,110,\n',
        encoding="utf-8",
    )
    summary = summarize_gpu_csv(path)
    assert summary["samples"] == 2
    assert summary["peak_vram_mib"] == 2048
    assert summary["mean_utilization_pct"] == 60
