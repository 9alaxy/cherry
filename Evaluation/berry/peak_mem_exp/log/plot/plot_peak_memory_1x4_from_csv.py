#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ['GCN', 'SAGE', 'GAT']
MODEL_LABELS = ['GCN', 'GraphSAGE', 'GAT']
METHOD_ORDER = ['DGL_random', 'DGL_metis', 'Betty', 'Berry']
METHOD_LABELS = ['随机划分方法', 'METIS划分方法', 'Betty', 'IODG-Full']
METHOD_COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
OOM_COLOR = '#d1d5db'


def configure_chinese_font():
    candidates = [
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'Noto Sans SC',
        'Source Han Sans SC',
        'Microsoft YaHei',
        'SimHei',
        'WenQuanYi Zen Hei',
        'PingFang SC',
        'Heiti SC',
        'Arial Unicode MS',
    ]

    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = next((name for name in candidates if name in available), None)
    except Exception:
        chosen = None

    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen] + candidates
    else:
        plt.rcParams['font.sans-serif'] = candidates

    plt.rcParams['axes.unicode_minus'] = False


def _build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df['status'] == 'success']
        .pivot_table(index='model', columns='method', values='peak_mem_gb_trainlog', aggfunc='mean')
        .reindex(index=MODEL_ORDER, columns=METHOD_ORDER)
    )


def _build_oom_map(df: pd.DataFrame) -> dict:
    is_oom = {}
    for model in MODEL_ORDER:
        for method in METHOD_ORDER:
            pair = df[(df['model'] == model) & (df['method'] == method)]
            is_oom[(model, method)] = (not pair.empty) and (pair['status'] == 'oom').any()
    return is_oom


def _draw_one(ax, df: pd.DataFrame, subtitle: str, ylim_top: float):
    pivot = _build_pivot(df)
    is_oom = _build_oom_map(df)

    x = np.arange(len(MODEL_ORDER))
    width = 0.22
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    for i, (method, label) in enumerate(zip(METHOD_ORDER, METHOD_LABELS)):
        for j, model in enumerate(MODEL_ORDER):
            value = pivot.loc[model, method]
            xpos = x[j] + offsets[i]

            if pd.notna(value):
                ax.bar(
                    xpos,
                    float(value),
                    width=width,
                    color=METHOD_COLORS[i],
                    edgecolor='black',
                    linewidth=0.8,
                    label=label if j == 0 else None,
                )
            elif is_oom[(model, method)]:
                ax.bar(
                    xpos,
                    ylim_top,
                    width=width,
                    color=OOM_COLOR,
                    edgecolor='black',
                    linewidth=0.8,
                    label=label if j == 0 else None,
                )
                ax.text(
                    xpos,
                    ylim_top * 0.98,
                    'OOM',
                    ha='center',
                    va='top',
                    color='#ef4444',
                    fontsize=18,
                    fontweight='bold',
                )
            else:
                ax.bar(
                    xpos,
                    np.nan,
                    width=width,
                    color=METHOD_COLORS[i],
                    edgecolor='black',
                    linewidth=0.8,
                    label=label if j == 0 else None,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    # ax.set_xlabel('模型')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, ylim_top)
    ax.text(0.5, -0.08, subtitle, transform=ax.transAxes, ha='center', va='top', fontsize=19)


def _auto_ylim(dfs: list[pd.DataFrame]) -> float:
    vals = []
    for df in dfs:
        series = pd.to_numeric(df['peak_mem_gb_trainlog'], errors='coerce')
        vals.extend(series.dropna().tolist())
    if not vals:
        return 1.0
    top = max(vals) * 1.15
    return max(1.0, math.ceil(top * 2) / 2.0)


def draw_1x4(
    arxiv_csv: Path,
    reddit_csv: Path,
    amazon_csv: Path,
    products_csv: Path,
    output_prefix: Path,
    ylim_top: float | None,
):
    configure_chinese_font()

    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
    })

    # Left -> right: arxiv, reddit, amazon, ogbn-products
    specs = [
        ('(a) arxiv', arxiv_csv),
        ('(b) reddit', reddit_csv),
        ('(c) amazon', amazon_csv),
        ('(d) ogbn-products', products_csv),
    ]

    dfs = []
    for _, csv_path in specs:
        df = pd.read_csv(csv_path)
        dfs.append(df)

    if ylim_top is None:
        ylim_top = _auto_ylim(dfs)

    # 拉长每个子图的高度，缩小子图间距
    fig, axes = plt.subplots(1, 4, figsize=(34, 6.5), sharey=True)

    legend_handles = None
    legend_labels = None
    for ax, (subtitle, _), df in zip(axes, specs, dfs):
        _draw_one(ax, df, subtitle, ylim_top)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[0].set_ylabel('显存 (GB)')

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            ncol=4,
            frameon=False,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.01),
        )

    # 缩小子图间距（wspace），拉长高度后适当调整rect，去除下方多余空白
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97), w_pad=0.15)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix('.png')
    pdf_path = output_prefix.with_suffix('.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description='Draw 1x4 peak-memory subplots from four CSV files')
    parser.add_argument('--arxiv-csv', default='ogbn-arxiv_plot_data.csv')
    parser.add_argument('--reddit-csv', default='reddit_plot_data.csv')
    parser.add_argument('--amazon-csv', default='amazon_plot_data.csv')
    parser.add_argument('--products-csv', default='ogbn-products_plot_data.csv')
    parser.add_argument('--output-prefix', default='all_datasets_peak_memory_1x4')
    parser.add_argument('--ylim-top', type=float, default=None)
    args = parser.parse_args()

    png_path, pdf_path = draw_1x4(
        arxiv_csv=Path(args.arxiv_csv),
        reddit_csv=Path(args.reddit_csv),
        amazon_csv=Path(args.amazon_csv),
        products_csv=Path(args.products_csv),
        output_prefix=Path(args.output_prefix),
        ylim_top=args.ylim_top,
    )
    print(png_path)
    print(pdf_path)


if __name__ == '__main__':
    main()
