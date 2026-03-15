#!/usr/bin/env python3
"""
Computational Efficiency Data Collection Script
Extracts time breakdown data from experiment logs
"""

import os
import re
import csv
import matplotlib.pyplot as plt
import numpy as np

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def extract_time_from_log(log_file):
    """Extract time breakdown from log file"""
    data = {
        'method': '',
        'num_batch': 0,
        'epoch0': {},
        'epoch1': {},
    }

    # Determine method and num_batch from filename
    basename = os.path.basename(log_file)
    if 'cherry_8batch' in basename:
        data['method'] = 'Cherry'
        data['num_batch'] = 8
    elif 'cherry_gmfg' in basename:
        data['method'] = 'Cherry'
        data['num_batch'] = 4
    elif 'vanilla' in basename:
        data['method'] = 'Vanilla'
        data['num_batch'] = 4

    with open(log_file, 'r') as f:
        content = f.read()

        # Extract epoch 0 time
        epoch0_match = re.search(
            r'Run 0\| Epoch 0 \|.*?TIME RECORD.*?load_block_time:\s+([\d.]+).*?block_move_time:\s+([\d.]+).*?model_time:\s+([\d.]+).*?loss_time:\s+([\d.]+).*?optimizer_time:\s+([\d.]+).*?total_time:\s+([\d.]+)',
            content, re.DOTALL
        )
        if epoch0_match:
            data['epoch0'] = {
                'load_block': float(epoch0_match.group(1)),
                'block_move': float(epoch0_match.group(2)),
                'model': float(epoch0_match.group(3)),
                'loss': float(epoch0_match.group(4)),
                'optimizer': float(epoch0_match.group(5)),
                'total': float(epoch0_match.group(6)),
            }

        # Extract epoch 1 time
        epoch1_match = re.search(
            r'Run 0\| Epoch 1 \|.*?TIME RECORD.*?load_block_time:\s+([\d.]+).*?block_move_time:\s+([\d.]+).*?model_time:\s+([\d.]+).*?loss_time:\s+([\d.]+).*?optimizer_time:\s+([\d.]+).*?total_time:\s+([\d.]+)',
            content, re.DOTALL
        )
        if epoch1_match:
            data['epoch1'] = {
                'load_block': float(epoch1_match.group(1)),
                'block_move': float(epoch1_match.group(2)),
                'model': float(epoch1_match.group(3)),
                'loss': float(epoch1_match.group(4)),
                'optimizer': float(epoch1_match.group(5)),
                'total': float(epoch1_match.group(6)),
            }

        # Extract number of nodes
        nodes_matches = re.findall(r'Number of nodes for computation during this epoch:\s+(\d+)', content)
        if len(nodes_matches) >= 1:
            data['epoch0']['nodes'] = int(nodes_matches[0])
        if len(nodes_matches) >= 2:
            data['epoch1']['nodes'] = int(nodes_matches[1])

    return data

# Process log files
log_files = [
    'cherry_gmfg_test.log',
    'vanilla_test.log',
    'cherry_8batch_test.log',
]

results = []

for log_file in log_files:
    log_path = os.path.join(script_dir, log_file)
    if os.path.exists(log_path):
        data = extract_time_from_log(log_path)
        results.append(data)
        print(f"Processed: {log_file}")
        print(f"  Method: {data['method']}, Batches: {data['num_batch']}")
        if data['epoch0']:
            print(f"  Epoch 0 Total: {data['epoch0'].get('total', 0):.4f}s")
        if data['epoch1']:
            print(f"  Epoch 1 Total: {data['epoch1'].get('total', 0):.4f}s")
        print()

# Save to CSV
csv_file = os.path.join(script_dir, 'computation_time_data.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['method', 'num_batch', 'epoch', 'total_time', 'load_block', 'block_move', 'model', 'loss', 'optimizer', 'nodes'])
    for r in results:
        for epoch_name, epoch_data in [('epoch0', r['epoch0']), ('epoch1', r['epoch1'])]:
            if epoch_data:
                writer.writerow([
                    r['method'],
                    r['num_batch'],
                    epoch_name,
                    epoch_data.get('total', 0),
                    epoch_data.get('load_block', 0),
                    epoch_data.get('block_move', 0),
                    epoch_data.get('model', 0),
                    epoch_data.get('loss', 0),
                    epoch_data.get('optimizer', 0),
                    epoch_data.get('nodes', 0),
                ])

print(f"Saved: {csv_file}")

# ============================================================
# Create Visualizations
# ============================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11

labels = ['Cherry\n(4 batches)', 'Vanilla\n(4 batches)', 'Cherry\n(8 batches)']
colors = ['#2ecc71', '#e74c3c', '#3498db']
x = np.arange(len(labels))
width = 0.6

# Extract data for plotting
epoch1_totals = [r['epoch1']['total'] for r in results]
epoch1_load_block = [r['epoch1']['load_block'] for r in results]
epoch1_block_move = [r['epoch1']['block_move'] for r in results]
epoch1_model = [r['epoch1']['model'] for r in results]
epoch1_loss = [r['epoch1']['loss'] for r in results]
epoch1_optimizer = [r['epoch1']['optimizer'] for r in results]

# Calculate throughput (nodes/sec)
throughputs = []
for r in results:
    if r['epoch1'].get('nodes') and r['epoch1'].get('total'):
        throughput = r['epoch1']['nodes'] / r['epoch1']['total']
    else:
        throughput = 0
    throughputs.append(throughput)

# ============================================================
# Figure 1: Epoch 1 Total Training Time
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
bars1 = ax1.bar(x, epoch1_totals, width, color=colors, edgecolor='black', linewidth=1.2)

ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Epoch 1 Training Time Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

for bar, val in zip(bars1, epoch1_totals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylim(0, max(epoch1_totals) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'training_time.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_time.png")

# ============================================================
# Figure 2: Time Breakdown (Stacked Bar)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Prepare data for stacked bar
categories = ['load_block', 'block_move', 'model', 'loss', 'optimizer']
cat_colors = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']

bottom = np.zeros(len(labels))
for cat, cat_color in zip(categories, cat_colors):
    cat_data = []
    for r in results:
        if cat == 'load_block':
            cat_data.append(r['epoch1'].get('load_block', 0))
        elif cat == 'block_move':
            cat_data.append(r['epoch1'].get('block_move', 0))
        elif cat == 'model':
            cat_data.append(r['epoch1'].get('model', 0))
        elif cat == 'loss':
            cat_data.append(r['epoch1'].get('loss', 0))
        elif cat == 'optimizer':
            cat_data.append(r['epoch1'].get('optimizer', 0))
    ax2.bar(x, cat_data, width, bottom=bottom, label=cat, color=cat_color, edgecolor='black', linewidth=0.5)
    bottom += cat_data

ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_title('Epoch 1 Time Breakdown', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'time_breakdown.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: time_breakdown.png")

# ============================================================
# Figure 3: Throughput Comparison
# ============================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))
bars3 = ax3.bar(x, throughputs, width, color=colors, edgecolor='black', linewidth=1.2)

ax3.set_xlabel('Method', fontsize=12, fontweight='bold')
ax3.set_ylabel('Throughput (nodes/sec)', fontsize=12, fontweight='bold')
ax3.set_title('Throughput Comparison (Nodes per Second)', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)

for bar, val in zip(bars3, throughputs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
             f'{val/1000:.1f}K/s', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax3.set_ylim(0, max(throughputs) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'throughput.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: throughput.png")

# ============================================================
# Figure 4: Time Percentage Breakdown
# ============================================================
fig4, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (r, color) in enumerate(zip(results, colors)):
    ax = axes[idx]
    epoch = r['epoch1']
    total = epoch.get('total', 1)

    sizes = [
        epoch.get('load_block', 0) / total * 100,
        epoch.get('block_move', 0) / total * 100,
        epoch.get('model', 0) / total * 100,
        epoch.get('loss', 0) / total * 100,
        epoch.get('optimizer', 0) / total * 100,
    ]

    labels_pie = ['Load Block', 'Block Move', 'Model', 'Loss', 'Optimizer']
    explode = (0.02, 0.02, 0.05, 0.02, 0.02)

    ax.pie(sizes, explode=explode, labels=labels_pie, autopct='%1.1f%%',
           colors=cat_colors, shadow=True, startangle=90)
    ax.set_title(f'{labels[idx].replace(chr(10), " ")}', fontsize=12, fontweight='bold')

fig4.suptitle('Epoch 1 Time Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'time_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: time_distribution.png")

# ============================================================
# Figure 5: Comprehensive Dashboard
# ============================================================
fig5, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training Time
ax = axes[0, 0]
bars = ax.bar(x, epoch1_totals, width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Epoch 1 Training Time', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Cherry\n(4b)', 'Vanilla\n(4b)', 'Cherry\n(8b)'])
for bar, val in zip(bars, epoch1_totals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}s', ha='center', va='bottom', fontsize=9)

# Throughput
ax = axes[0, 1]
bars = ax.bar(x, [t/1000 for t in throughputs], width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Throughput (K nodes/sec)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Cherry\n(4b)', 'Vanilla\n(4b)', 'Cherry\n(8b)'])
for bar, val in zip(bars, [t/1000 for t in throughputs]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val:.1f}K', ha='center', va='bottom', fontsize=9)

# Model time (largest component)
ax = axes[1, 0]
bars = ax.bar(x, epoch1_model, width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Model Forward Time', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Cherry\n(4b)', 'Vanilla\n(4b)', 'Cherry\n(8b)'])
for bar, val in zip(bars, epoch1_model):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}s', ha='center', va='bottom', fontsize=9)

# Load block time
ax = axes[1, 1]
bars = ax.bar(x, epoch1_load_block, width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Block Loading Time', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Cherry\n(4b)', 'Vanilla\n(4b)', 'Cherry\n(8b)'])
for bar, val in zip(bars, epoch1_load_block):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}s', ha='center', va='bottom', fontsize=9)

fig5.suptitle('Computational Efficiency Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'computation_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: computation_dashboard.png")

print("\n" + "="*60)
print("Computational Efficiency Data Collection Complete")
print("="*60)
