import os
from statistics import mean
import argparse
import glob


def read_acc_loss(filename):
    """Read accuracy from log file. Extracts test accuracy from lines containing 'Run' and 'Test'."""
    acc_array = []
    loss_array = []
    try:
        with open(filename) as f:
            for line in f:
                if 'Run' in line.strip() and 'Test' in line.strip():
                    parts = line.split()
                    try:
                        acc = float(parts[-1])
                        loss = float(parts[7])
                        acc_array.append(acc)
                        loss_array.append(loss)
                    except (IndexError, ValueError):
                        continue
    except FileNotFoundError:
        print(f"Warning: File not found: {filename}")
    return acc_array


def data_formalize(acc_list):
    """Normalize acc_list to same length by trimming to minimum length."""
    if not acc_list:
        return [], []
    min_len = min(len(arr) for arr in acc_list)
    trimmed = [arr[:min_len] for arr in acc_list]
    return range(min_len), trimmed


def draw(acc_list, labels=None, output_file='comparison.png'):
    """Draw accuracy curves for all data series.

    Args:
        acc_list: List of accuracy arrays
        labels: List of labels for each series (default: 'Series N')
        output_file: Output filename for the plot
    """
    import matplotlib.pyplot as plt

    if not acc_list:
        print("No data to plot")
        return

    if labels is None:
        labels = [f'Series {i+1}' for i in range(len(acc_list))]

    while len(labels) < len(acc_list):
        labels.append(f'Series {len(labels)+1}')

    fig, ax = plt.subplots(figsize=(7, 4))

    x, acc_data = data_formalize(acc_list)

    colors = ['orange', 'red', 'blue', 'green', 'purple', 'gray', 'brown', 'pink']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

    for i, (acc, label) in enumerate(zip(acc_data, labels)):
        ax.plot(x, acc, linestyles[i % len(linestyles)], label=label,
                color=colors[i % len(colors)])

    ax.set(xlabel='Epoch', ylabel='Test Accuracy')
    plt.legend()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")


def data_collection(path, file_patterns):
    """Collect data from log files matching patterns.

    Args:
        path: Directory containing log files
        file_patterns: List of glob patterns or filenames to match

    Returns:
        List of accuracy arrays
    """
    acc_list = []

    for pattern in file_patterns:
        if os.path.isfile(pattern):
            files = [pattern]
        elif os.path.isfile(os.path.join(path, pattern)):
            files = [os.path.join(path, pattern)]
        else:
            files = glob.glob(os.path.join(path, pattern))
            if not files:
                print(f"Warning: No files match pattern: {pattern}")
                continue

        for filepath in files:
            print(f"Reading: {filepath}")
            acc = read_acc_loss(filepath)
            if acc:
                acc_list.append(acc)

    return acc_list


if __name__ == '__main__':
    print("Computation info data collection start ......")

    argparser = argparse.ArgumentParser("info collection")
    argparser.add_argument('--path', type=str, default='./ac_log/ogbn-arxiv/',
                           help='Directory containing log files')
    argparser.add_argument('--files', type=str, nargs='+',
                           default=['Cherry-4-batch-2-layer-64-hid-GCN-ogbn-arxiv.log',
                                   'Berry-4-batch-2-layer-64-hid-GCN-ogbn-arxiv.log'],
                           help='Log files to plot (can be filenames or glob patterns)')
    argparser.add_argument('--labels', type=str, nargs='+',
                           help='Labels for each file (default: auto-generated)')
    argparser.add_argument('--output', type=str, default='comparison.png',
                           help='Output filename for the plot')
    args = argparser.parse_args()

    acc_list = data_collection(args.path, args.files)

    if acc_list:
        draw(acc_list, labels=args.labels, output_file=args.output)
    else:
        print("No data found!")
