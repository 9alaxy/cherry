#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXP_DIR = Path(__file__).resolve().parent
LOG_DIR = EXP_DIR / "log"

FRAMEWORK_RE = re.compile(r"max memory allocated:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_framework_peak(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None
    max_gb = None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for m in FRAMEWORK_RE.finditer(text):
        v = float(m.group(1))
        if max_gb is None or v > max_gb:
            max_gb = v
    return max_gb


def parse_nvsmi_peak(gpu_log: Path) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    if not gpu_log.exists():
        return None, None, None

    peak_mib = None
    gpu_name = None
    mem_total = None

    with gpu_log.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("nvidia-smi start failed"):
                continue
            if "memory.used" in line and "memory.total" in line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpu_name = parts[2]
            used = parts[3].replace(" MiB", "").strip()
            total = parts[4].replace(" MiB", "").strip()
            try:
                used_mib = float(used)
                total_mib = float(total)
            except ValueError:
                continue
            if peak_mib is None or used_mib > peak_mib:
                peak_mib = used_mib
            mem_total = f"{total_mib/1024.0:.1f} GB"

    if peak_mib is None:
        return None, gpu_name, mem_total
    return peak_mib / 1024.0, gpu_name, mem_total


def load_meta_rows(log_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for meta in sorted(log_dir.glob("*.meta.json")):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            rows.append(data)
        except Exception:
            continue
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect peak-memory results into CSV")
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR))
    parser.add_argument("--output", type=str, default=str(LOG_DIR / "peak_memory_results.csv"))
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    meta_rows = load_meta_rows(log_dir)

    out_rows: List[Dict[str, object]] = []
    for row in meta_rows:
        main_log = Path(str(row.get("main_log", "")))
        gpu_log = Path(str(row.get("gpu_log", "")))
        fw_peak = parse_framework_peak(main_log)
        nvsmi_peak, gpu_name, gpu_total = parse_nvsmi_peak(gpu_log)

        out_rows.append(
            {
                "dataset": row.get("dataset"),
                "model": row.get("model"),
                "method": row.get("method"),
                "seed": row.get("seed"),
                "peak_mem_gb_framework": fw_peak,
                "peak_mem_gb_nvsmi": nvsmi_peak,
                "status": row.get("status"),
                "exit_code": row.get("exit_code"),
                "is_oom": row.get("is_oom"),
                "is_timeout": row.get("is_timeout"),
                "duration_s": row.get("duration_s"),
                "gpu_name": gpu_name,
                "gpu_mem_total": gpu_total,
                "main_log": str(main_log),
                "gpu_log": str(gpu_log),
                "meta_json": row.get("meta_json"),
                "start_utc": row.get("start_utc"),
                "end_utc": row.get("end_utc"),
            }
        )

    headers = [
        "dataset",
        "model",
        "method",
        "seed",
        "peak_mem_gb_framework",
        "peak_mem_gb_nvsmi",
        "status",
        "exit_code",
        "is_oom",
        "is_timeout",
        "duration_s",
        "gpu_name",
        "gpu_mem_total",
        "main_log",
        "gpu_log",
        "meta_json",
        "start_utc",
        "end_utc",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"[RESULT] rows={len(out_rows)} output={output}")


if __name__ == "__main__":
    main()
