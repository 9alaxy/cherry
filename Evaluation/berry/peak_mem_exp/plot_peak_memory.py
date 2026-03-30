#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
LOG_DIR = EXP_DIR / "log"

METHOD_ORDER = ["Berry", "Betty", "DGL_random", "DGL_metis"]
MODEL_ORDER = ["GCN", "GAT", "SAGE"]
METHOD_LABELS = {
    "Berry": "IODG-Full",
    "Betty": "Betty",
    "DGL_random": "随机化分方法",
    "DGL_metis": "METIS划分方法",
}
COLORS = {
    "GCN": "#1f77b4",
    "GAT": "#ff7f0e",
    "SAGE": "#2ca02c",
}


def main():
    parser = argparse.ArgumentParser(description="Plot per-dataset peak memory grouped bars")
    parser.add_argument("--input", type=str, default=str(LOG_DIR / "peak_memory_results.csv"))
    parser.add_argument("--output-dir", type=str, default=str(LOG_DIR / "plots"))
    parser.add_argument("--metric", choices=["peak_mem_gb_nvsmi", "peak_mem_gb_framework"], default="peak_mem_gb_nvsmi")
    args = parser.parse_args()

    in_csv = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if df.empty:
        raise SystemExit("Input CSV is empty.")

    df = df[df["status"] == "success"].copy()
    if df.empty:
        raise SystemExit("No successful runs found.")

    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")

    gpu_info = df[["gpu_name", "gpu_mem_total"]].dropna().head(1)
    gpu_suffix = ""
    if not gpu_info.empty:
        row = gpu_info.iloc[0]
        gpu_suffix = f" | GPU: {row['gpu_name']} ({row['gpu_mem_total']})"

    grouped = (
        df.groupby(["dataset", "method", "model"], as_index=False)[args.metric]
        .agg(["mean", "std"])  # type: ignore
        .reset_index()
    )

    datasets = sorted(df["dataset"].dropna().unique().tolist())
    for dataset in datasets:
        sub = grouped[grouped["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(9.5, 5.2))

        x = list(range(len(METHOD_ORDER)))
        width = 0.22
        offsets = {"GCN": -width, "GAT": 0.0, "SAGE": width}

        for model in MODEL_ORDER:
            means = []
            stds = []
            for method in METHOD_ORDER:
                row = sub[(sub["method"] == method) & (sub["model"] == model)]
                if row.empty:
                    means.append(float("nan"))
                    stds.append(0.0)
                else:
                    means.append(float(row["mean"].iloc[0]))
                    std_val = row["std"].iloc[0]
                    stds.append(0.0 if pd.isna(std_val) else float(std_val))

            pos = [v + offsets[model] for v in x]
            ax.bar(
                pos,
                means,
                width=width,
                yerr=stds,
                capsize=3,
                color=COLORS[model],
                label=model,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in METHOD_ORDER])
        ax.set_ylabel("峰值显存 (GB)")
        ax.set_xlabel("模型")
        ax.set_title(f"{dataset} 显存对比{gpu_suffix}")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()

        out_png = out_dir / f"{dataset}_peak_memory.png"
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        print(f"[RESULT] {out_png}")


if __name__ == "__main__":
    main()
