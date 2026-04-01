#!/usr/bin/env python3
"""
使用方法：
  # 在脚本所在目录扫描 .log 并生成作图 CSV
  /root/miniconda3/envs/cherry/bin/python collect_replication_factor_plot_csv.py

  # 指定日志目录
  /root/miniconda3/envs/cherry/bin/python collect_replication_factor_plot_csv.py \
      --base-dir /workspace/Cherry/Evaluation/renduancy/ogbn-arxiv

输出：
  replication-factor-plot-ogbn-arxiv.csv

规则：
  - Berry: 取每个日志最后一个 Replication Factor
  - REG:   取每个日志最大的 Replication Factor
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

RF_RE = re.compile(r"Replication Factor:\s*([0-9]+\.[0-9]+)")
BERRY_RE = re.compile(r"^Berry-(\d+)-batch-3-layer-256-hid-GCN-ogbn-arxiv\.log$")
REG_RE = re.compile(r"^REG-(\d+)-batch-3-layer-256-hid-GCN-ogbn-arxiv\.log$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Berry/REG 日志提取 replication factor 并生成作图 CSV。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "中文示例:\n"
            "  /root/miniconda3/envs/cherry/bin/python collect_replication_factor_plot_csv.py\n"
            "  /root/miniconda3/envs/cherry/bin/python collect_replication_factor_plot_csv.py "
            "--base-dir /workspace/Cherry/Evaluation/renduancy/ogbn-arxiv\n"
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="日志目录（默认脚本所在目录）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 CSV 路径（默认 <base-dir>/replication-factor-plot-ogbn-arxiv.csv）。",
    )
    return parser.parse_args()


def extract_rf_values(log_path: Path) -> list[float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return [float(x) for x in RF_RE.findall(text)]


def collect(base_dir: Path) -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []

    for log_path in sorted(base_dir.glob("*.log")):
        name = log_path.name
        m_berry = BERRY_RE.match(name)
        m_reg = REG_RE.match(name)

        if not m_berry and not m_reg:
            continue

        values = extract_rf_values(log_path)
        if not values:
            raise ValueError(f"No 'Replication Factor' found in {log_path}")

        if m_berry:
            mb = int(m_berry.group(1))
            rows.append(("Berry", mb, values[-1]))
        else:
            mb = int(m_reg.group(1))
            rows.append(("REG", mb, max(values)))

    if not rows:
        raise ValueError(f"No matching Berry/REG logs found in {base_dir}")

    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def write_csv(out_path: Path, rows: list[tuple[str, int, float]]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "micro_batch", "replication_factor_for_plot"])
        for method, mb, value in rows:
            writer.writerow([method, mb, f"{value:.4f}"])


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    out_path = (
        args.output.resolve()
        if args.output is not None
        else base_dir / "./plot_replication_factor/replication-factor-plot-ogbn-arxiv.csv"
    )

    rows = collect(base_dir)
    write_csv(out_path, rows)

    print(f"Generated: {out_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
