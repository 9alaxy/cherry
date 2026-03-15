#!/usr/bin/env python3
"""
Memory Efficiency Data Collection Script
Extracts memory data from experiment logs for analysis
"""

import os
import re
import csv
import matplotlib.pyplot as plt
import numpy as np

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def extract_memory_from_log(log_file):
    """Extract memory metrics from a log file"""
    memory_data = {
        'method': '',
        'num_batch': 0,
        'peak_memory_gb': 0.0,
        'micro_batch_memory': [],
        'initial_gpu_memory_gb': 0.0,
    }

    # Determine method and num_batch from filename
    basename = os.path.basename(log_file)
    if 'cherry_8batch' in basename:
        memory_data['method'] = 'Cherry'
        memory_data['num_batch'] = 8
    elif 'cherry_gmfg' in basename:
        memory_data['method'] = 'Cherry'
        memory_data['num_batch'] = 4
    elif 'vanilla' in basename:
        memory_data['method'] = 'Vanilla'
        memory_data['num_batch'] = 4

    with open(log_file, 'r') as f:
        for line in f:
            # Initial GPU memory
            if 'Nvidia-smi:' in line and 'GB' in line and 'before load data' in open(log_file).read():
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'GB' in part and i > 0:
                        try:
                            memory_data['initial_gpu_memory_gb'] = float(parts[i-1])
                        except:
                            pass

            # Micro-batch max memory allocated
            match = re.search(r'Micro-batch-\s*\d+\s+max memory allocated:\s+([\d.]+)\s+GB', line)
            if match:
                memory_data['micro_batch_memory'].append(float(match.group(1)))

            # Peak memory allocated
            match = re.search(r'Max Memory Allocated:\s+([\d.]+)\s+GigaBytes', line)
            if match:
                memory_data['peak_memory_gb'] = float(match.group(1))

    return memory_data

# Process all log files
log_files = [
    'cherry_gmfg_test.log',
    'vanilla_test.log',
    'cherry_8batch_test.log',
]

results = []

for log_file in log_files:
    log_path = os.path.join(script_dir, log_file)
    if os.path.exists(log_path):
        data = extract_memory_from_log(log_path)
        results.append(data)
        print(f"Processed: {log_file}")
        print(f"  Method: {data['method']}, Batches: {data['num_batch']}")
        print(f"  Peak Memory: {data['peak_memory_gb']:.4f} GB")
        print(f"  Micro-batch Memory: {data['micro_batch_memory']}")
        print()

# Save to CSV
csv_file = os.path.join(script_dir, 'memory_data.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['method', 'num_batch', 'peak_memory_gb', 'micro_batch_0', 'micro_batch_1', 'micro_batch_2', 'micro_batch_3'])
    for r in results:
        row = [r['method'], r['num_batch'], r['peak_memory_gb']]
        # Pad with empty values if less than 4 micro-batches
        row.extend(r['micro_batch_memory'] + [''] * (4 - len(r['micro_batch_memory'])))
        writer.writerow(row)

print(f"Saved: {csv_file}")

# ============================================================
# Create Visualization
# ============================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11

labels = ['Cherry\n(4 batches)', 'Vanilla\n(4 batches)', 'Cherry\n(8 batches)']
colors = ['#2ecc71', '#e74c3c', '#3498db']
x = np.arange(len(labels))
width = 0.6

# Figure 1: Peak Memory Comparison
fig1, ax1 = plt.subplots(figsize=(10, 6))
peak_memories = [r['peak_memory_gb'] for r in results]
bars1 = ax1.bar(x, peak_memories, width, color=colors, edgecolor='black', linewidth=1.2)

ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
ax1.set_ylabel('Peak Memory (GB)', fontsize=12, fontweight='bold')
ax1.set_title('Peak GPU Memory Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

for bar, val in zip(bars1, peak_memories):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.4f} GB', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylim(0, max(peak_memories) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'peak_memory.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: peak_memory.png")

# Figure 2: Micro-batch Memory Distribution (per epoch)
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# First epoch micro-batch memory
ax = axes2[0]
micro_batch_epoch0 = []
for r in results:
    # First epoch: first num_batch entries
    n_batch = r['num_batch']
    if len(r['micro_batch_memory']) >= n_batch:
        micro_batch_epoch0.append(r['micro_batch_memory'][:n_batch])
    else:
        micro_batch_epoch0.append([])

mb_x = np.arange(4)
mb_width = 0.25

for i, (data, color, label) in enumerate(zip(micro_batch_epoch0, colors, labels)):
    if len(data) == 4:
        bars = ax.bar(mb_x + i * mb_width, data, mb_width, color=color, edgecolor='black', label=label.replace('\n', ' '))

ax.set_xlabel('Micro-batch Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Memory Allocated (GB)', fontsize=12, fontweight='bold')
ax.set_title('Epoch 0: Memory per Micro-batch', fontsize=12, fontweight='bold')
ax.set_xticks(mb_x + mb_width)
ax.set_xticklabels(['MB 0', 'MB 1', 'MB 2', 'MB 3'])
ax.legend(fontsize=9)

# Second epoch micro-batch memory
ax = axes2[1]
micro_batch_epoch1 = []
for r in results:
    n_batch = r['num_batch']
    if len(r['micro_batch_memory']) >= n_batch * 2:
        micro_batch_epoch1.append(r['micro_batch_memory'][n_batch:n_batch*2])
    else:
        micro_batch_epoch1.append([])

for i, (data, color, label) in enumerate(zip(micro_batch_epoch1, colors, labels)):
    if len(data) == 4:
        bars = ax.bar(mb_x + i * mb_width, data, mb_width, color=color, edgecolor='black', label=label.replace('\n', ' '))

ax.set_xlabel('Micro-batch Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Memory Allocated (GB)', fontsize=12, fontweight='bold')
ax.set_title('Epoch 1: Memory per Micro-batch', fontsize=12, fontweight='bold')
ax.set_xticks(mb_x + mb_width)
ax.set_xticklabels(['MB 0', 'MB 1', 'MB 2', 'MB 3'])
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'micro_batch_memory.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: micro_batch_memory.png")

# Figure 3: Memory Efficiency Dashboard
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

# Peak Memory
ax = axes[0]
bars = ax.bar(x, peak_memories, width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Peak Memory (GB)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Cherry\n(4b)', 'Vanilla\n(4b)', 'Cherry\n(8b)'])
for bar, val in zip(bars, peak_memories):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# Average Micro-batch Memory
ax = axes[1]
avg_mb_memories = [np.mean(r['micro_batch_memory']) if r['micro_batch_memory'] else 0 for r in results]
bars = ax.bar(x, avg_mb_memories, width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Avg Micro-batch Memory (GB)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Cherry\n(4b)', 'Vanilla\n(4b)', 'Cherry\n(8b)'])
for bar, val in zip(bars, avg_mb_memories):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

fig3.suptitle('Memory Efficiency Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'memory_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: memory_dashboard.png")

print("\n" + "="*60)
print("Memory Efficiency Data Collection Complete")
print("="*60)
