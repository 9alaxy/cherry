#!/usr/bin/env python3
"""
Scalability Analysis Script
Analyzes how performance scales with different num-batch configurations.
"""

import os
import re
import csv
import matplotlib.pyplot as plt
import numpy as np

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def extract_scalability_from_log(log_file):
    """Extract scalability metrics from log file."""
    data = {
        'method': '',
        'num_batch': 0,
        'dataset': '',
        'num_layers': 0,
        'peak_memory_gb': 0.0,
        'epoch0_time': 0.0,
        'epoch1_time': 0.0,
        'epoch0_throughput': 0.0,  # nodes/sec
        'epoch1_throughput': 0.0,
        'partition_time': 0.0,
        'replication_factor': 0.0,
        'edge_cut_ratio': 0.0,
        'total_nodes': 0,
        'train_nodes': 0,
    }

    # Determine method and num_batch from filename
    basename = os.path.basename(log_file)

    # Parse method and num_batch from filename
    # Format: METHOD_XXbatch_XXlayer_XXhid_MODEL_DATASET.log
    # Example: Cherry_2batch_2layer_64hid_GCN_ogbn-arxiv.log
    basename_no_ext = os.path.splitext(basename)[0]

    # Extract method (first part before _XXbatch)
    method_match = re.search(r'^(\w+)_(\d+)batch', basename_no_ext)
    if method_match:
        data['method'] = method_match.group(1).capitalize()
        data['num_batch'] = int(method_match.group(2))

    # Extract dataset (last part after last _)
    parts = basename_no_ext.split('_')
    for ds in ['ogbn-arxiv', 'ogbn-products', 'reddit', 'amazon', 'ogbn-papers100M']:
        if ds in basename:
            data['dataset'] = ds
            break

    # Extract num_layers
    layer_match = re.search(r'(\d+)layer', basename_no_ext)
    if layer_match:
        data['num_layers'] = int(layer_match.group(1))

    # Extract num_hidden
    hidden_match = re.search(r'(\d+)hid', basename_no_ext)
    if hidden_match:
        data['num_hidden'] = int(hidden_match.group(1))

    # Extract model type
    for model in ['GCN', 'SAGE', 'GAT']:
        if model in basename_no_ext:
            data['model'] = model
            break

    with open(log_file, 'r') as f:
        content = f.read()
        lines = content.split('\n')

        # Extract dataset info
        for line in lines:
            if '# Train:' in line:
                match = re.search(r'# Train:\s+(\d+)', line)
                if match:
                    data['train_nodes'] = int(match.group(1))
            if '# Nodes:' in line:
                match = re.search(r'# Nodes:\s+(\d+)', line)
                if match:
                    data['total_nodes'] = int(match.group(1))

        # Extract peak memory
        mem_match = re.search(r'Max Memory Allocated:\s+([\d.]+)\s+GigaBytes', content)
        if mem_match:
            data['peak_memory_gb'] = float(mem_match.group(1))

        # Also check nvidia-smi for peak memory
        nvidia_matches = re.findall(r'Nvidia-smi:\s+([\d.]+)\s+GB', content)
        if nvidia_matches:
            data['peak_memory_gb'] = max([float(m) for m in nvidia_matches])

        # Extract partition time
        part_match = re.search(r'one partition time:\s+([\d.]+)', content)
        if part_match:
            data['partition_time'] = float(part_match.group(1))

        # Extract replication factor
        rf_match = re.search(r'Replication Factor:\s+([\d.]+)', content)
        if rf_match:
            data['replication_factor'] = float(rf_match.group(1))

        # Extract edge cut ratio
        ec_match = re.search(r'Edge cut ratio:\s+([\d.]+)', content)
        if ec_match:
            data['edge_cut_ratio'] = float(ec_match.group(1))

        # Extract epoch times
        epoch_times = re.findall(
            r'Run \d+ \| Epoch (\d+) \|.*?total_time:\s+([\d.]+)',
            content, re.DOTALL
        )
        for epoch, total_time in epoch_times:
            epoch_num = int(epoch)
            time_val = float(total_time)
            if epoch_num == 0:
                data['epoch0_time'] = time_val
                # Calculate throughput
                if data['train_nodes'] > 0 and time_val > 0:
                    data['epoch0_throughput'] = data['train_nodes'] / time_val
            elif epoch_num == 1:
                data['epoch1_time'] = time_val
                if data['train_nodes'] > 0 and time_val > 0:
                    data['epoch1_throughput'] = data['train_nodes'] / time_val

    return data


def normalize_scalability_data(results, baseline_key='num_batch=4'):
    """Normalize data relative to baseline for comparison."""
    # Find baseline
    baseline = None
    for r in results:
        if r[baseline_key] == 4 and r.get('method') == 'Cherry':
            baseline = r
            break

    if not baseline:
        return results

    normalized = []
    for r in results:
        norm_r = r.copy()
        if baseline.get('peak_memory_gb', 0) > 0:
            norm_r['norm_memory'] = r.get('peak_memory_gb', 0) / baseline.get('peak_memory_gb', 1)
        if baseline.get('epoch1_time', 0) > 0:
            norm_r['norm_time'] = r.get('epoch1_time', 0) / baseline.get('epoch1_time', 1)
        normalized.append(norm_r)

    return normalized


# ============================================================
# Main Analysis
# ============================================================

# Collect all log files from subdirectories
log_files = []
search_dirs = [
    script_dir,
    os.path.join(script_dir, '..'),
    os.path.join(script_dir, 'partitioning_quality'),
    os.path.join(script_dir, 'memory_efficiency'),
    os.path.join(script_dir, 'computational_efficiency'),
]

for search_dir in search_dirs:
    if os.path.exists(search_dir):
        for f in os.listdir(search_dir):
            if f.endswith('.log'):
                log_files.append(os.path.join(search_dir, f))

results = []

for log_file in log_files:
    data = extract_scalability_from_log(log_file)
    if data['num_batch'] > 0:  # Only add if we successfully extracted data
        results.append(data)
        print(f"Processed: {os.path.basename(log_file)}")
        print(f"  Method: {data['method']}, Batches: {data['num_batch']}")
        print(f"  Epoch 1 Time: {data['epoch1_time']:.4f}s")
        print(f"  Peak Memory: {data['peak_memory_gb']:.4f} GB")
        print()

# Group by method for analysis
cherry_results = [r for r in results if r['method'] == 'Cherry']
vanilla_results = [r for r in results if r['method'] == 'Vanilla']
betty_results = [r for r in results if r['method'] == 'Betty']

# Sort by num_batch
cherry_results.sort(key=lambda x: x['num_batch'])

# Save to CSV
csv_file = os.path.join(script_dir, 'scalability_data.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['method', 'num_batch', 'dataset', 'num_layers', 'peak_memory_gb',
                     'epoch0_time', 'epoch1_time', 'epoch0_throughput', 'epoch1_throughput',
                     'partition_time', 'replication_factor', 'edge_cut_ratio', 'train_nodes'])
    for r in results:
        writer.writerow([
            r['method'],
            r['num_batch'],
            r.get('dataset', ''),
            r.get('num_layers', 0),
            r['peak_memory_gb'],
            r['epoch0_time'],
            r['epoch1_time'],
            r['epoch0_throughput'],
            r['epoch1_throughput'],
            r['partition_time'],
            r['replication_factor'],
            r['edge_cut_ratio'],
            r['train_nodes'],
        ])

print(f"Saved: {csv_file}")

# ============================================================
# Create Visualizations
# ============================================================

if not cherry_results:
    print("No Cherry results to analyze for scalability")
    exit(0)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11

# Extract Cherry data for plotting
num_batches = [r['num_batch'] for r in cherry_results]
epoch1_times = [r['epoch1_time'] for r in cherry_results]
peak_memories = [r['peak_memory_gb'] for r in cherry_results]
throughputs = [r['epoch1_throughput'] for r in cherry_results]
partition_times = [r['partition_time'] for r in cherry_results]
replication_factors = [r['replication_factor'] for r in cherry_results]
edge_cut_ratios = [r['edge_cut_ratio'] for r in cherry_results]

# Figure 1: Training Time vs Num Batch
fig1, ax = plt.subplots(figsize=(10, 6))
ax.plot(num_batches, epoch1_times, 'o-', color='#2ecc71', linewidth=2, markersize=10, label='Cherry')
ax.fill_between(num_batches, [t * 0.9 for t in epoch1_times], [t * 1.1 for t in epoch1_times],
                alpha=0.2, color='#2ecc71')
ax.set_xlabel('Number of Micro-batches', fontsize=12, fontweight='bold')
ax.set_ylabel('Epoch 1 Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Training Time Scalability', fontsize=14, fontweight='bold')
ax.set_xticks(num_batches)
ax.grid(True, alpha=0.3)

# Add data labels
for x, y in zip(num_batches, epoch1_times):
    ax.annotate(f'{y:.3f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'scalability_time.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: scalability_time.png")

# Figure 2: Memory vs Num Batch
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(num_batches, peak_memories, 's-', color='#e74c3c', linewidth=2, markersize=10, label='Cherry')
ax.fill_between(num_batches, [m * 0.9 for m in peak_memories], [m * 1.1 for m in peak_memories],
                alpha=0.2, color='#e74c3c')
ax.set_xlabel('Number of Micro-batches', fontsize=12, fontweight='bold')
ax.set_ylabel('Peak GPU Memory (GB)', fontsize=12, fontweight='bold')
ax.set_title('Memory Scalability', fontsize=14, fontweight='bold')
ax.set_xticks(num_batches)
ax.grid(True, alpha=0.3)

for x, y in zip(num_batches, peak_memories):
    ax.annotate(f'{y:.3f}GB', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'scalability_memory.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: scalability_memory.png")

# Figure 3: Throughput vs Num Batch
fig3, ax = plt.subplots(figsize=(10, 6))
ax.plot(num_batches, [t/1000 for t in throughputs], '^-', color='#3498db', linewidth=2, markersize=10, label='Cherry')
ax.set_xlabel('Number of Micro-batches', fontsize=12, fontweight='bold')
ax.set_ylabel('Throughput (K nodes/sec)', fontsize=12, fontweight='bold')
ax.set_title('Throughput Scalability', fontsize=14, fontweight='bold')
ax.set_xticks(num_batches)
ax.grid(True, alpha=0.3)

for x, y in zip(num_batches, [t/1000 for t in throughputs]):
    ax.annotate(f'{y:.2f}K', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'scalability_throughput.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: scalability_throughput.png")

# Figure 4: Partition Quality vs Num Batch
fig4, axes = plt.subplots(1, 2, figsize=(14, 5))

# Replication factor
ax = axes[0]
ax.plot(num_batches, replication_factors, 'o-', color='#9b59b6', linewidth=2, markersize=10)
ax.set_xlabel('Number of Micro-batches', fontsize=12, fontweight='bold')
ax.set_ylabel('Replication Factor', fontsize=12, fontweight='bold')
ax.set_title('Replication Factor vs Num Batch', fontsize=12, fontweight='bold')
ax.set_xticks(num_batches)
ax.grid(True, alpha=0.3)

for x, y in zip(num_batches, replication_factors):
    ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

# Edge cut ratio
ax = axes[1]
ax.plot(num_batches, edge_cut_ratios, 's-', color='#f39c12', linewidth=2, markersize=10)
ax.set_xlabel('Number of Micro-batches', fontsize=12, fontweight='bold')
ax.set_ylabel('Edge Cut Ratio', fontsize=12, fontweight='bold')
ax.set_title('Edge Cut Ratio vs Num Batch', fontsize=12, fontweight='bold')
ax.set_xticks(num_batches)
ax.grid(True, alpha=0.3)

for x, y in zip(num_batches, edge_cut_ratios):
    ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'scalability_partition.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: scalability_partition.png")

# Figure 5: Comprehensive Scalability Dashboard
fig5, axes = plt.subplots(2, 3, figsize=(15, 10))

# Time
ax = axes[0, 0]
ax.plot(num_batches, epoch1_times, 'o-', color='#2ecc71', linewidth=2, markersize=8)
ax.set_xlabel('Num Batch')
ax.set_ylabel('Time (s)')
ax.set_title('Training Time')
ax.set_xticks(num_batches)
ax.grid(True, alpha=0.3)

# Memory
ax = axes[0, 1]
ax.plot(num_batches, peak_memories, 's-', color='#e74c3c', linewidth=2, markersize=8)
ax.set_xlabel('Num Batch')
ax.set_ylabel('Memory (GB)')
ax.set_title('Peak Memory')
ax.set_xticks(num_batches)
ax.grid(True, alpha=0.3)

# Throughput
ax = axes[0, 2]
ax.plot(num_batches, [t/1000 for t in throughputs], '^-', color='#3498db', linewidth=2, markersize=8)
ax.set_xlabel('Num Batch')
ax.set_ylabel('Throughput (K/s)')
ax.set_title('Throughput')
ax.set_xticks(num_batches)
ax.grid(True, alpha=0.3)

# Partition time
ax = axes[1, 0]
ax.plot(num_batches, partition_times, 'd-', color='#1abc9c', linewidth=2, markersize=8)
ax.set_xlabel('Num Batch')
ax.set_ylabel('Time (s)')
ax.set_title('Partition Time')
ax.set_xticks(num_batches)
ax.grid(True, alpha=0.3)

# Replication factor
ax = axes[1, 1]
ax.plot(num_batches, replication_factors, 'p-', color='#9b59b6', linewidth=2, markersize=8)
ax.set_xlabel('Num Batch')
ax.set_ylabel('Repl. Factor')
ax.set_title('Replication Factor')
ax.set_xticks(num_batches)
ax.grid(True, alpha=0.3)

# Edge cut
ax = axes[1, 2]
ax.plot(num_batches, edge_cut_ratios, 'h-', color='#f39c12', linewidth=2, markersize=8)
ax.set_xlabel('Num Batch')
ax.set_ylabel('Edge Cut Ratio')
ax.set_title('Edge Cut')
ax.set_xticks(num_batches)
ax.grid(True, alpha=0.3)

fig5.suptitle('Cherry Scalability Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'scalability_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: scalability_dashboard.png")

# Figure 6: Efficiency Metrics (normalized to baseline)
if len(num_batches) >= 2:
    baseline_time = epoch1_times[0]
    baseline_mem = peak_memories[0]

    if baseline_time > 0 and baseline_mem > 0:
        norm_times = [t / baseline_time for t in epoch1_times]
        norm_mems = [m / baseline_mem for m in peak_memories]

        fig6, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(num_batches))
        width = 0.35

        bars1 = ax.bar(x - width/2, norm_times, width, label='Normalized Time', color='#2ecc71', edgecolor='black')
        bars2 = ax.bar(x + width/2, norm_mems, width, label='Normalized Memory', color='#e74c3c', edgecolor='black')

        ax.set_xlabel('Number of Micro-batches', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Value (relative to 4 batches)', fontsize=12, fontweight='bold')
        ax.set_title('Normalized Scalability (Lower is Better for Both)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(num_batches)
        ax.legend()
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, 'scalability_normalized.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: scalability_normalized.png")

# ============================================================
# Print Summary Statistics
# ============================================================
print("\n" + "="*60)
print("Scalability Analysis Summary (Cherry)")
print("="*60)

if len(num_batches) >= 2:
    # Calculate scaling efficiency
    time_increase = (epoch1_times[-1] - epoch1_times[0]) / epoch1_times[0] * 100 if epoch1_times[0] > 0 else 0
    mem_decrease = (peak_memories[0] - peak_memories[-1]) / peak_memories[0] * 100 if peak_memories[0] > 0 else 0

    print(f"Num Batch Range: {min(num_batches)} -> {max(num_batches)}")
    print(f"Time Increase: {time_increase:.2f}%")
    print(f"Memory Decrease: {mem_decrease:.2f}%")
    print(f"Partition Time Range: {min(partition_times):.3f}s -> {max(partition_times):.3f}s")

    # Calculate ideal vs actual scaling
    batch_ratio = max(num_batches) / min(num_batches)
    ideal_time_ratio = batch_ratio
    actual_time_ratio = epoch1_times[-1] / epoch1_times[0] if epoch1_times[0] > 0 else 0
    time_efficiency = ideal_time_ratio / actual_time_ratio if actual_time_ratio > 0 else 0

    print(f"\nScaling Efficiency:")
    print(f"  Ideal time increase: {batch_ratio}x")
    print(f"  Actual time increase: {actual_time_ratio:.2f}x")
    print(f"  Time scaling efficiency: {time_efficiency:.2f}")

print("\n" + "="*60)
print("Scalability Analysis Complete")
print("="*60)
