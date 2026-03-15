#!/usr/bin/env python3
"""
Load Balance Analysis Script
Analyzes computational load balance across micro-batches.
"""

import os
import re
import csv
import matplotlib.pyplot as plt
import numpy as np

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def extract_load_balance_from_log(log_file):
    """Extract load balance metrics from log file."""
    data = {
        'method': '',
        'num_batch': 0,
        'partition_nodes': [],  # Number of seed nodes per partition
        'partition_src_nodes': [],  # Number of source nodes per partition
        'partition_weights': [],  # Weight (ratio) per partition
        'micro_batch_memory': [],  # Memory per micro-batch
        'micro_batch_time': [],  # Time per micro-batch (computed from total - other components)
        'replication_factor': 0.0,
        'edge_cut_ratio': 0.0,
    }

    # Determine method and num_batch from filename
    # Format: METHOD_XXbatch_XXlayer_XXhid_MODEL_DATASET.log
    # Example: Cherry_2batch_2layer_64hid_GCN_ogbn-arxiv.log
    basename = os.path.splitext(os.path.basename(log_file))[0]

    # Extract method and num_batch
    method_match = re.search(r'^(\w+)_(\d+)batch', basename)
    if method_match:
        data['method'] = method_match.group(1).capitalize()
        data['num_batch'] = int(method_match.group(2))

    with open(log_file, 'r') as f:
        content = f.read()
        lines = content.split('\n')

        # Extract partition nodes (seed nodes per micro-batch)
        partition_nodes = []
        for line in lines:
            match = re.search(r'Micro-batch-\s+(\d+)\s+train node:\s+(\d+)', line)
            if match:
                partition_nodes.append(int(match.group(2)))
        data['partition_nodes'] = partition_nodes

        # Extract weights list
        weight_match = re.search(r'Weights List: \[([\d.,\s]+)\]', content)
        if weight_match:
            weights = [float(w) for w in weight_match.group(1).split(',')]
            data['partition_weights'] = weights

        # Extract replication factor
        rf_match = re.search(r'Replication Factor:\s+([\d.]+)', content)
        if rf_match:
            data['replication_factor'] = float(rf_match.group(1))

        # Extract edge cut ratio
        ec_match = re.search(r'Edge cut ratio:\s+([\d.]+)', content)
        if ec_match:
            data['edge_cut_ratio'] = float(ec_match.group(1))

        # Extract micro-batch memory
        mb_mem = []
        for line in lines:
            match = re.search(r'Micro-batch-\s+(\d+)\s+max memory allocated:\s+([\d.]+)\s+GB', line)
            if match:
                mb_mem.append(float(match.group(2)))
        data['micro_batch_memory'] = mb_mem

        # Extract time breakdown per epoch
        epoch_data = []
        epoch_times = re.findall(
            r'Run \d+\| Epoch \d+ \|.*?load_block_time:\s+([\d.]+).*?block_move_time:\s+([\d.]+).*?model_time:\s+([\d.]+).*?loss_time:\s+([\d.]+).*?optimizer_time:\s+([\d.]+).*?total_time:\s+([\d.]+)',
            content, re.DOTALL
        )
        for et in epoch_times:
            epoch_data.append({
                'load_block': float(et[0]),
                'block_move': float(et[1]),
                'model': float(et[2]),
                'loss': float(et[3]),
                'optimizer': float(et[4]),
                'total': float(et[5]),
            })
        data['epoch_times'] = epoch_data

        # Calculate per micro-batch time (approximate)
        if epoch_data and data['num_batch'] > 0:
            total_model_time = epoch_data[0]['model']
            total_load_time = epoch_data[0]['load_block']
            total_move_time = epoch_data[0]['block_move']
            # Distribute proportionally based on partition nodes
            if sum(partition_nodes) > 0:
                ratios = [n / sum(partition_nodes) for n in partition_nodes]
                data['micro_batch_model_time'] = [t * r for t, r in zip([total_model_time] * len(partition_nodes), ratios)]
                data['micro_batch_load_time'] = [t * r for t, r in zip([total_load_time] * len(partition_nodes), ratios)]
                data['micro_batch_move_time'] = [t * r for t, r in zip([total_move_time] * len(partition_nodes), ratios)]

    return data


def calculate_load_balance_metrics(data):
    """Calculate load balance metrics."""
    metrics = {}

    # Partition nodes imbalance
    if data['partition_nodes']:
        nodes = np.array(data['partition_nodes'])
        metrics['nodes_mean'] = np.mean(nodes)
        metrics['nodes_std'] = np.std(nodes)
        metrics['nodes_cv'] = metrics['nodes_std'] / metrics['nodes_mean'] if metrics['nodes_mean'] > 0 else 0  # Coefficient of variation
        metrics['nodes_max'] = np.max(nodes)
        metrics['nodes_min'] = np.min(nodes)
        metrics['nodes_imbalance_ratio'] = metrics['nodes_max'] / metrics['nodes_min'] if metrics['nodes_min'] > 0 else 0

    # Partition weights imbalance
    if data['partition_weights']:
        weights = np.array(data['partition_weights'])
        metrics['weights_std'] = np.std(weights)
        metrics['weights_max'] = np.max(weights)
        metrics['weights_min'] = np.min(weights)
        metrics['weights_imbalance_ratio'] = metrics['weights_max'] / metrics['weights_min'] if metrics['weights_min'] > 0 else 0

    # Memory imbalance
    if data['micro_batch_memory']:
        mem = np.array(data['micro_batch_memory'])
        metrics['mem_mean'] = np.mean(mem)
        metrics['mem_std'] = np.std(mem)
        metrics['mem_cv'] = metrics['mem_std'] / metrics['mem_mean'] if metrics['mem_mean'] > 0 else 0
        metrics['mem_max'] = np.max(mem)
        metrics['mem_min'] = np.min(mem)
        metrics['mem_imbalance_ratio'] = metrics['mem_max'] / metrics['mem_min'] if metrics['mem_min'] > 0 else 0

    # Time imbalance (if available)
    if 'micro_batch_model_time' in data and data['micro_batch_model_time']:
        times = np.array(data['micro_batch_model_time'])
        metrics['time_mean'] = np.mean(times)
        metrics['time_std'] = np.std(times)
        metrics['time_cv'] = metrics['time_std'] / metrics['time_mean'] if metrics['time_mean'] > 0 else 0

    return metrics


# Process log files
log_files = [
    'cherry_gmfg_test.log',
    'vanilla_test.log',
    'cherry_8batch_test.log',
    'betty_test.log',
]

results = []

for log_file in log_files:
    log_path = os.path.join(script_dir, log_file)
    if os.path.exists(log_path):
        data = extract_load_balance_from_log(log_path)
        if data['partition_nodes']:
            metrics = calculate_load_balance_metrics(data)
            data['metrics'] = metrics
            results.append(data)
            print(f"Processed: {log_file}")
            print(f"  Method: {data['method']}, Batches: {data['num_batch']}")
            print(f"  Partition nodes: {data['partition_nodes']}")
            print(f"  Nodes CV: {metrics.get('nodes_cv', 0):.4f}")
            print(f"  Memory CV: {metrics.get('mem_cv', 0):.4f}")
            print()

# Save to CSV
csv_file = os.path.join(script_dir, 'load_balance_data.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['method', 'num_batch', 'nodes_mean', 'nodes_std', 'nodes_cv',
                     'nodes_imbalance_ratio', 'weights_std', 'mem_cv', 'mem_imbalance_ratio',
                     'replication_factor', 'edge_cut_ratio'])
    for r in results:
        m = r.get('metrics', {})
        writer.writerow([
            r['method'],
            r['num_batch'],
            m.get('nodes_mean', 0),
            m.get('nodes_std', 0),
            m.get('nodes_cv', 0),
            m.get('nodes_imbalance_ratio', 0),
            m.get('weights_std', 0),
            m.get('mem_cv', 0),
            m.get('mem_imbalance_ratio', 0),
            r.get('replication_factor', 0),
            r.get('edge_cut_ratio', 0),
        ])

print(f"Saved: {csv_file}")

# ============================================================
# Create Visualizations
# ============================================================

if not results:
    print("No data to visualize")
    exit(0)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11

# Prepare labels and colors
labels = []
colors = []
for r in results:
    method = r['method']
    num_batch = r['num_batch']
    labels.append(f"{method}\n({num_batch} batches)")
    if method == 'Cherry':
        colors.append('#2ecc71')
    elif method == 'Vanilla':
        colors.append('#e74c3c')
    elif method == 'Betty':
        colors.append('#3498db')
    else:
        colors.append('#9b59b6')

x = np.arange(len(labels))
width = 0.6

# Figure 1: Partition Nodes Distribution
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of partition nodes - group by same num_batch
ax = axes[0]

# Group results by num_batch
batch_groups = {}
for r, label, color in zip(results, labels, colors):
    nb = r['num_batch']
    if nb not in batch_groups:
        batch_groups[nb] = []
    batch_groups[nb].append((r, label, color))

# Plot each group separately
bar_width = 0.8 / max(len(bg) for bg in batch_groups.values()) if batch_groups else 1

for batch_size, group in batch_groups.items():
    x_pos = np.arange(batch_size)
    for idx, (r, label, color) in enumerate(group):
        pn = r['partition_nodes']
        if len(pn) == batch_size:
            offset = (idx - len(group)/2 + 0.5) * bar_width
            ax.bar(x_pos + offset, pn, bar_width, color=color, edgecolor='black',
                   label=label.replace('\n', ' '), alpha=0.8)

ax.set_xlabel('Micro-batch Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Seed Nodes', fontsize=12, fontweight='bold')
ax.set_title('Seed Nodes Distribution per Micro-batch', fontsize=12, fontweight='bold')
ax.set_xticks(np.arange(max(batch_groups.keys())))
ax.set_xticklabels([f'MB {i}' for i in range(max(batch_groups.keys()))])
ax.legend(fontsize=9, loc='upper right')

# Coefficient of Variation comparison
ax = axes[1]
cvs = [r['metrics'].get('nodes_cv', 0) for r in results]
bars = ax.bar(x, cvs, width, color=colors, edgecolor='black', linewidth=1.2)
ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')
ax.set_title('Nodes Distribution Imbalance (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
for bar, val in zip(bars, cvs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'load_balance_nodes.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: load_balance_nodes.png")

# Figure 2: Memory Imbalance
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# Micro-batch memory bar chart - group by num_batch
ax = axes[0]

for batch_size, group in batch_groups.items():
    x_pos = np.arange(batch_size)
    for idx, (r, label, color) in enumerate(group):
        mem = r['micro_batch_memory'][:batch_size]  # Take first batch_size entries
        if len(mem) == batch_size:
            offset = (idx - len(group)/2 + 0.5) * bar_width
            ax.bar(x_pos + offset, mem, bar_width, color=color, edgecolor='black',
                   label=label.replace('\n', ' '), alpha=0.8)

ax.set_xlabel('Micro-batch Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Memory Allocated (GB)', fontsize=12, fontweight='bold')
ax.set_title('Memory Usage per Micro-batch', fontsize=12, fontweight='bold')
ax.set_xticks(np.arange(max(batch_groups.keys())))
ax.set_xticklabels([f'MB {i}' for i in range(max(batch_groups.keys()))])
ax.legend(fontsize=9, loc='upper right')

# Memory CV comparison - only for same num_batch
ax = axes[1]
# Filter results to only compare same num_batch
same_batch_results = [r for r in results if r['num_batch'] == 4]
same_batch_labels = [l for l, r in zip(labels, results) if r['num_batch'] == 4]
same_batch_colors = [c for c, r in zip(colors, results) if r['num_batch'] == 4]
x_comp = np.arange(len(same_batch_results))

mem_cvs = [r['metrics'].get('mem_cv', 0) for r in same_batch_results]
bars = ax.bar(x_comp, mem_cvs, 0.6, color=same_batch_colors, edgecolor='black', linewidth=1.2)
ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Memory CV', fontsize=12, fontweight='bold')
ax.set_title('Memory Imbalance (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_xticks(x_comp)
ax.set_xticklabels(same_batch_labels, fontsize=9)
for bar, val in zip(bars, mem_cvs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'load_balance_memory.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: load_balance_memory.png")

# Figure 3: Comprehensive Load Balance Dashboard
fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

# Nodes imbalance ratio
ax = axes[0, 0]
ratios = [r['metrics'].get('nodes_imbalance_ratio', 0) for r in results]
bars = ax.bar(x, ratios, width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Nodes Imbalance Ratio (Max/Min)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([l.replace('\n', ' ') for l in labels], fontsize=8)
for bar, val in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# Weights standard deviation
ax = axes[0, 1]
weights_std = [r['metrics'].get('weights_std', 0) for r in results]
bars = ax.bar(x, weights_std, width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Weights Std Dev (Lower is Better)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([l.replace('\n', ' ') for l in labels], fontsize=8)
for bar, val in zip(bars, weights_std):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# Replication factor
ax = axes[1, 0]
rfs = [r.get('replication_factor', 0) for r in results]
bars = ax.bar(x, rfs, width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Replication Factor (Lower is Better)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([l.replace('\n', ' ') for l in labels], fontsize=8)
for bar, val in zip(bars, rfs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# Edge cut ratio
ax = axes[1, 1]
ecrs = [r.get('edge_cut_ratio', 0) for r in results]
bars = ax.bar(x, ecrs, width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Edge Cut Ratio (Lower is Better)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([l.replace('\n', ' ') for l in labels], fontsize=8)
for bar, val in zip(bars, ecrs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)

fig3.suptitle('Load Balance & Partition Quality Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'load_balance_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: load_balance_dashboard.png")

# Figure 4: Summary Table
fig4, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# Create summary table
table_data = []
headers = ['Method', 'Batches', 'Nodes CV', 'Mem CV', 'Imbalance Ratio', 'Repl. Factor', 'Edge Cut']

for r in results:
    m = r['metrics']
    table_data.append([
        r['method'],
        r['num_batch'],
        f"{m.get('nodes_cv', 0):.4f}",
        f"{m.get('mem_cv', 0):.4f}",
        f"{m.get('nodes_imbalance_ratio', 0):.3f}",
        f"{r.get('replication_factor', 0):.4f}",
        f"{r.get('edge_cut_ratio', 0):.4f}",
    ])

table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Color the header
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

# Color the rows
for i in range(len(table_data)):
    for j in range(len(headers)):
        if i % 2 == 0:
            table[(i+1, j)].set_facecolor('#ecf0f1')
        else:
            table[(i+1, j)].set_facecolor('white')

ax.set_title('Load Balance Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'load_balance_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: load_balance_summary.png")

print("\n" + "="*60)
print("Load Balance Analysis Complete")
print("="*60)
