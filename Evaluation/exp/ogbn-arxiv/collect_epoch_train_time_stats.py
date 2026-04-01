#!/usr/bin/env python3
"""
使用方法：
  # 1）使用默认设置（warm-up 丢弃前 2 个 epoch）
  /root/miniconda3/envs/cherry/bin/python collect_epoch_train_time_stats.py

  # 2）自定义 warm-up 丢弃数量（例如丢弃前 5 个 epoch）
  /root/miniconda3/envs/cherry/bin/python collect_epoch_train_time_stats.py --warmup-epochs 5

  # 3）指定日志目录
  /root/miniconda3/envs/cherry/bin/python collect_epoch_train_time_stats.py \
      --base-dir /workspace/Cherry/Evaluation/renduancy/ogbn-arxiv

输出文件：
  - epoch-train-time-details-ogbn-arxiv.csv
    每一行对应一个 epoch 的训练时间
  - epoch-train-time-summary-ogbn-arxiv.csv
    按 method + micro_batch 汇总，包含全量与去 warm-up 后统计（mean / p50 / p90）
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


EPOCH_RE = re.compile(r"Run\s+\d+\s+\|\s+Epoch\s+(\d+)")
TOTAL_TIME_RE = re.compile(r"total_time:\s*([0-9]+(?:\.[0-9]+)?)")
LOG_RE = re.compile(
    r"^(Berry|REG)-(\d+)-batch-3-layer-256-hid-GCN-ogbn-arxiv\.log$"
)


@dataclass
class EpochRecord:
    method: str
    micro_batch: int
    epoch: int
    train_time_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="提取 Berry/REG 每个 epoch 的训练时间，并导出明细与汇总 CSV。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "中文示例：\n"
            "  /root/miniconda3/envs/cherry/bin/python collect_epoch_train_time_stats.py\n"
            "  /root/miniconda3/envs/cherry/bin/python collect_epoch_train_time_stats.py --warmup-epochs 5\n"
            "  /root/miniconda3/envs/cherry/bin/python collect_epoch_train_time_stats.py "
            "--base-dir /workspace/Cherry/Evaluation/renduancy/ogbn-arxiv\n"
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="日志目录，默认是脚本所在目录。",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="汇总统计时丢弃前 N 个 epoch，默认 2。",
    )
    return parser.parse_args()


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("percentile() received empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def parse_log(log_path: Path, method: str, micro_batch: int) -> list[EpochRecord]:
    records: list[EpochRecord] = []
    current_epoch: int | None = None

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue

        time_match = TOTAL_TIME_RE.search(line)
        if time_match and current_epoch is not None:
            records.append(
                EpochRecord(
                    method=method,
                    micro_batch=micro_batch,
                    epoch=current_epoch,
                    train_time_sec=float(time_match.group(1)),
                )
            )
            current_epoch = None

    if not records:
        raise ValueError(f"No epoch total_time found in: {log_path}")

    return records


def collect_all_records(base_dir: Path) -> list[EpochRecord]:
    records: list[EpochRecord] = []

    for log_path in sorted(base_dir.glob("*.log")):
        m = LOG_RE.match(log_path.name)
        if not m:
            continue
        method = m.group(1)
        micro_batch = int(m.group(2))
        records.extend(parse_log(log_path, method, micro_batch))

    if not records:
        raise ValueError(
            f"No matching Berry/REG log files found in directory: {base_dir}"
        )

    return records


def write_details_csv(out_path: Path, records: list[EpochRecord]) -> None:
    records_sorted = sorted(records, key=lambda r: (r.method, r.micro_batch, r.epoch))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "micro_batch", "epoch", "train_time_sec"])
        for r in records_sorted:
            writer.writerow([r.method, r.micro_batch, r.epoch, f"{r.train_time_sec:.6f}"])


def write_summary_csv(out_path: Path, records: list[EpochRecord], warmup_epochs: int) -> None:
    grouped: dict[tuple[str, int], list[EpochRecord]] = {}
    for r in records:
        grouped.setdefault((r.method, r.micro_batch), []).append(r)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "micro_batch",
                "total_epochs",
                "warmup_epochs_dropped",
                "epochs_used_after_warmup",
                "mean_all_sec",
                "mean_after_warmup_sec",
                "p50_after_warmup_sec",
                "p90_after_warmup_sec",
                "sum_all_sec",
                "sum_after_warmup_sec",
            ]
        )

        for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
            rows = sorted(grouped[key], key=lambda r: r.epoch)
            all_values = [r.train_time_sec for r in rows]
            steady_values = all_values[warmup_epochs:]
            if not steady_values:
                steady_values = all_values

            all_sorted = sorted(all_values)
            steady_sorted = sorted(steady_values)

            writer.writerow(
                [
                    key[0],
                    key[1],
                    len(all_values),
                    warmup_epochs,
                    len(steady_values),
                    f"{(sum(all_values) / len(all_values)):.6f}",
                    f"{(sum(steady_values) / len(steady_values)):.6f}",
                    f"{percentile(steady_sorted, 0.50):.6f}",
                    f"{percentile(steady_sorted, 0.90):.6f}",
                    f"{sum(all_values):.6f}",
                    f"{sum(steady_values):.6f}",
                ]
            )


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be >= 0")

    records = collect_all_records(base_dir)

    details_out = base_dir / "epoch-train-time-details-ogbn-arxiv.csv"
    summary_out = base_dir / "epoch-train-time-summary-ogbn-arxiv.csv"

    write_details_csv(details_out, records)
    write_summary_csv(summary_out, records, args.warmup_epochs)

    print("Generated:")
    print(f"  {details_out}")
    print(f"  {summary_out}")
    print(f"Warm-up dropped: first {args.warmup_epochs} epochs")


if __name__ == "__main__":
    main()
