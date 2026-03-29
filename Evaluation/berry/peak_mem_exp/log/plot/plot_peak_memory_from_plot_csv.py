#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_plot(csv_path: Path, output_prefix: Path, ylim_top: float):
    df = pd.read_csv(csv_path)

    # match /workspace/Cherry/Evaluation/mem/aggregator/plot.py font style
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
    })

    model_order = ['GCN', 'SAGE', 'GAT']
    model_labels = ['GCN', 'GraphSAGE', 'GAT']
    method_order = ['DGL_random', 'DGL_metis', 'Betty', 'Berry']
    method_labels = ['dgl_random', 'dgl_metis', 'betty', 'berry']
    method_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    oom_color = '#d1d5db'

    pivot = (
        df[df['status'] == 'success']
        .pivot_table(index='model', columns='method', values='peak_mem_gb_trainlog', aggfunc='mean')
        .reindex(index=model_order, columns=method_order)
    )

    is_oom = {}
    for m in model_order:
        for meth in method_order:
            pair = df[(df['model'] == m) & (df['method'] == meth)]
            is_oom[(m, meth)] = (not pair.empty) and (pair['status'] == 'oom').any()

    x = np.arange(len(model_order))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (meth, label) in enumerate(zip(method_order, method_labels)):
        for j, m in enumerate(model_order):
            v = pivot.loc[m, meth]
            xpos = x[j] + offsets[i]
            if pd.notna(v):
                ax.bar(
                    xpos,
                    float(v),
                    width=width,
                    color=method_colors[i],
                    edgecolor='black',
                    linewidth=0.8,
                    label=label if j == 0 else None,
                )
            elif is_oom[(m, meth)]:
                # OOM style follows Evaluation/mem/aggregator/plot.py
                ax.bar(
                    xpos,
                    ylim_top,
                    width=width,
                    color=oom_color,
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
                    fontsize=12,
                    fontweight='bold',
                )
            else:
                ax.bar(
                    xpos,
                    np.nan,
                    width=width,
                    color=method_colors[i],
                    edgecolor='black',
                    linewidth=0.8,
                    label=label if j == 0 else None,
                )

    dataset_name = str(df['dataset'].iloc[0]) if not df.empty and 'dataset' in df.columns else 'dataset'
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_xlabel('Model')
    ax.set_ylabel('Memory (GB)')
    ax.set_title(f'{dataset_name} Peak Memory')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(ncol=2, frameon=False)
    ax.set_ylim(0, ylim_top)
    fig.tight_layout()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix('.png')
    pdf_path = output_prefix.with_suffix('.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    return png_path, pdf_path


"""
cd /workspace/Cherry/Evaluation/berry/peak_mem_exp/log/plot
python plot_peak_memory_from_plot_csv.py \
  --csv ogbn-products_plot_data.csv \
  --output-prefix ogbn-products_peak_memory_model_grouped_methods \
  --ylim-top 25
"""
def main():
    parser = argparse.ArgumentParser(description='Draw grouped peak-memory bar chart from *_plot_data.csv')
    parser.add_argument('--csv', required=True, help='Input plot-data CSV')
    parser.add_argument('--output-prefix', required=True, help='Output path prefix (without extension)')
    parser.add_argument('--ylim-top', type=float, default=25.0, help='Y-axis upper bound')
    args = parser.parse_args()

    png_path, pdf_path = draw_plot(
        csv_path=Path(args.csv),
        output_prefix=Path(args.output_prefix),
        ylim_top=args.ylim_top,
    )
    print(png_path)
    print(pdf_path)


if __name__ == '__main__':
    main()
