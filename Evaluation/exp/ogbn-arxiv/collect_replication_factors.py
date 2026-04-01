#!/usr/bin/env python3
"""
使用方法：
  # 1）REG 使用每个日志中的全部 epoch
  /root/miniconda3/envs/cherry/bin/python collect_replication_factors.py

  # 2）REG 只使用前 N 个 epoch（例如 20 或 200）
  /root/miniconda3/envs/cherry/bin/python collect_replication_factors.py --reg-epochs 200

  # 3）指定日志目录
  /root/miniconda3/envs/cherry/bin/python collect_replication_factors.py \
      --base-dir /workspace/Cherry/Evaluation/renduancy/ogbn-arxiv

输出文件：
  - Berry-replication-factor-ogbn-arxiv.csv
  - REG-replication-factor-ogbn-arxiv.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean

RF_PATTERN = re.compile(r"Replication Factor:\s*([0-9]+\.[0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect replication factors from Berry and REG logs under ogbn-arxiv."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "中文示例：\n"
            "  /root/miniconda3/envs/cherry/bin/python collect_replication_factors.py\n"
            "  /root/miniconda3/envs/cherry/bin/python collect_replication_factors.py --reg-epochs 200\n"
            "  /root/miniconda3/envs/cherry/bin/python collect_replication_factors.py "
            "--base-dir /workspace/Cherry/Evaluation/renduancy/ogbn-arxiv\n"
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing Berry-*.log and REG-*.log files.",
    )
    parser.add_argument(
        "--reg-epochs",
        type=int,
        default=None,
        help="Use first N REG epochs; default is all epochs found in each REG log.",
    )
    return parser.parse_args()


def extract_replication_factors(log_path: Path) -> list[float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return [float(x) for x in RF_PATTERN.findall(text)]


def collect_berry(base_dir: Path, micro_batches: list[int]) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for mb in micro_batches:
        log_path = base_dir / f"Berry-{mb}-batch-3-layer-256-hid-GCN-ogbn-arxiv.log"
        if not log_path.is_file():
            raise FileNotFoundError(f"Missing file: {log_path}")

        values = extract_replication_factors(log_path)
        if not values:
            raise ValueError(f"No 'Replication Factor' found in: {log_path}")

        rows.append((mb, values[-1]))
    return rows


def collect_reg(
    base_dir: Path,
    micro_batches: list[int],
    reg_epochs: int | None,
) -> list[tuple[int, int, float, float, float]]:
    rows: list[tuple[int, int, float, float, float]] = []

    for mb in micro_batches:
        log_path = base_dir / f"REG-{mb}-batch-3-layer-256-hid-GCN-ogbn-arxiv.log"
        if not log_path.is_file():
            raise FileNotFoundError(f"Missing file: {log_path}")

        values = extract_replication_factors(log_path)
        if reg_epochs is not None:
            values = values[:reg_epochs]

        if not values:
            raise ValueError(
                f"No usable 'Replication Factor' found in: {log_path} "
                f"(reg_epochs={reg_epochs})"
            )

        rows.append((mb, len(values), mean(values), max(values), min(values)))

    return rows


def write_berry_csv(out_path: Path, rows: list[tuple[int, float]]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["micro_batch", "replication_factor"])
        for mb, rf in rows:
            writer.writerow([mb, f"{rf:.4f}"])


def write_reg_csv(out_path: Path, rows: list[tuple[int, int, float, float, float]]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "micro_batch",
                "epochs_used",
                "avg_replication_factor",
                "max_replication_factor",
                "min_replication_factor",
            ]
        )
        for mb, epochs_used, avg_rf, max_rf, min_rf in rows:
            writer.writerow(
                [
                    mb,
                    epochs_used,
                    f"{avg_rf:.4f}",
                    f"{max_rf:.4f}",
                    f"{min_rf:.4f}",
                ]
            )


def main() -> None:
    args = parse_args()
    base_dir: Path = args.base_dir.resolve()

    if args.reg_epochs is not None and args.reg_epochs <= 0:
        raise ValueError("--reg-epochs must be a positive integer")

    berry_mbs = [2, 4, 8, 16, 32, 64]
    reg_mbs = [2, 4, 8, 32, 64]

    berry_rows = collect_berry(base_dir, berry_mbs)
    reg_rows = collect_reg(base_dir, reg_mbs, args.reg_epochs)

    berry_out = base_dir / "Berry-replication-factor-ogbn-arxiv.csv"
    reg_out = base_dir / "REG-replication-factor-ogbn-arxiv.csv"

    write_berry_csv(berry_out, berry_rows)
    write_reg_csv(reg_out, reg_rows)

    print("Generated:")
    print(f"  {berry_out}")
    print(f"  {reg_out}")
    if args.reg_epochs is None:
        print("REG mode: all epochs in each log")
    else:
        print(f"REG mode: first {args.reg_epochs} epochs")


if __name__ == "__main__":
    main()
