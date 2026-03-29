#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

FRAMEWORK_RE = re.compile(r"max memory allocated:\s*([0-9]+(?:\.[0-9]+)?)")

METHOD_ORDER = ["Berry", "Betty", "DGL_random", "DGL_metis"]
MODEL_ORDER = ["GCN", "GAT", "SAGE"]
COLORS = {
    "GCN": "#1f77b4",
    "GAT": "#ff7f0e",
    "SAGE": "#2ca02c",
}


def parse_framework_peak(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    max_gb = None
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
            if not line:
                continue
            if "memory.used" in line and "memory.total" in line:
                continue
            if line.startswith("nvidia-smi start failed"):
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


def extract_failure_reason(main_log: Path, status: str) -> str:
    if status == "success":
        return ""
    if not main_log.exists():
        return "log_missing"

    text = main_log.read_text(encoding="utf-8", errors="ignore")
    lower = text.lower()

    if "out of memory" in lower or "cuda out of memory" in lower:
        return "oom"
    if "timed out" in lower:
        return "timeout"
    if "dglerror" in lower:
        return "dgl_error"
    if "filenotfounderror" in lower:
        return "file_not_found"
    if "runtimeerror" in lower:
        return "runtime_error"
    if "exception" in lower:
        return "exception"

    last = ""
    for line in text.strip().splitlines()[::-1]:
        line = line.strip()
        if not line:
            continue
        last = line
        break
    return f"failed:{last[:120]}" if last else "failed_unknown"


def load_manifest_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: Path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_results(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    succ = df[df["status"] == "success"].copy()
    if succ.empty:
        return []

    succ["peak_mem_gb_nvsmi"] = pd.to_numeric(succ["peak_mem_gb_nvsmi"], errors="coerce")
    succ["peak_mem_gb_framework"] = pd.to_numeric(succ["peak_mem_gb_framework"], errors="coerce")

    # Prefer nvsmi; fallback to framework.
    succ["plot_metric"] = succ["peak_mem_gb_nvsmi"].fillna(succ["peak_mem_gb_framework"])

    gpu_info = succ[["gpu_name", "gpu_mem_total"]].dropna().head(1)
    gpu_suffix = ""
    if not gpu_info.empty:
        row = gpu_info.iloc[0]
        gpu_suffix = f" | GPU: {row['gpu_name']} ({row['gpu_mem_total']})"

    grouped = succ.groupby(["dataset", "method", "model"], as_index=False)["plot_metric"].mean()
    datasets = sorted(succ["dataset"].dropna().unique().tolist())
    outputs = []

    for dataset in datasets:
        sub = grouped[grouped["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(9.5, 5.2))

        x = list(range(len(METHOD_ORDER)))
        width = 0.22
        offsets = {"GCN": -width, "GAT": 0.0, "SAGE": width}

        for model in MODEL_ORDER:
            vals = []
            for method in METHOD_ORDER:
                row = sub[(sub["method"] == method) & (sub["model"] == model)]
                if row.empty:
                    vals.append(float("nan"))
                else:
                    vals.append(float(row["plot_metric"].iloc[0]))

            pos = [v + offsets[model] for v in x]
            ax.bar(
                pos,
                vals,
                width=width,
                color=COLORS[model],
                label=model,
                alpha=0.92,
                edgecolor="black",
                linewidth=0.6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER)
        ax.set_ylabel("Peak Memory (GB)")
        ax.set_xlabel("Method")
        ax.set_title(f"{dataset} Peak Memory Comparison{gpu_suffix}")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()

        out_png = out_dir / f"{dataset}_peak_memory.png"
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        outputs.append(str(out_png))

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Summarize peak memory results from a manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--plot-dir", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    plot_dir = Path(args.plot_dir)

    rows = load_manifest_rows(manifest)

    out_rows = []
    failure_counter = Counter()

    for r in rows:
        main_log = Path(r["main_log"])
        gpu_log = Path(r["gpu_log"])
        fw = parse_framework_peak(main_log)
        nv, gpu_name, gpu_total = parse_nvsmi_peak(gpu_log)
        reason = extract_failure_reason(main_log, r["status"])
        if r["status"] != "success":
            failure_counter[reason] += 1

        out_rows.append(
            {
                "dataset": r["dataset"],
                "model": r["model"],
                "method": r["method"],
                "seed": r["seed"],
                "peak_mem_gb_framework": fw,
                "peak_mem_gb_nvsmi": nv,
                "status": r["status"],
                "failure_reason": reason,
                "exit_code": r["exit_code"],
                "is_oom": r["is_oom"],
                "is_timeout": r["is_timeout"],
                "duration_s": r["duration_s"],
                "gpu_name": gpu_name,
                "gpu_mem_total": gpu_total,
                "main_log": r["main_log"],
                "gpu_log": r["gpu_log"],
                "meta_json": r["meta_json"],
                "start_utc": r["start_utc"],
                "end_utc": r["end_utc"],
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
        "failure_reason",
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
    write_csv(out_csv, out_rows, headers)

    df = pd.DataFrame(out_rows)
    total = len(df)
    succ = int((df["status"] == "success").sum())
    fail = total - succ

    by_dataset = (
        df.groupby(["dataset", "status"]).size().unstack(fill_value=0).reset_index().to_dict(orient="records")
    )
    by_method = (
        df.groupby(["method", "status"]).size().unstack(fill_value=0).reset_index().to_dict(orient="records")
    )

    plots = plot_results(df, plot_dir)

    summary = {
        "manifest": str(manifest),
        "total_runs": total,
        "success_runs": succ,
        "failed_runs": fail,
        "success_rate": (succ / total) if total else 0.0,
        "failure_reasons": dict(failure_counter),
        "by_dataset": by_dataset,
        "by_method": by_method,
        "result_csv": str(out_csv),
        "plots": plots,
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[RESULT] csv={out_csv}")
    print(f"[RESULT] summary={summary_json}")
    print(f"[RESULT] plots={len(plots)}")
    print(f"[RESULT] success={succ}/{total} ({summary['success_rate']:.2%})")


if __name__ == "__main__":
    main()
