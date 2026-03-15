#!/usr/bin/env python3
"""
Plotting script for partitioning quality experiments
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os

# Get script directory for saving files
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (12, 8)

# Data from summary.csv
data = {
    'method': ['Cherry', 'Vanilla', 'Cherry'],
    'num_batch': [4, 4, 8],
    'replication_factor': [1.2283, 1.1959, 1.6051],
    'edge_cut': [27762, 23972, 92714],
    'edge_cut_ratio': [0.0211, 0.0322, 0.0706],
    'partition_std': [136.26, 56.57, 29.60],
    'partition_cv': [0.0060, 0.0025, 0.0026],
    'peak_memory_gb': [0.2352, 0.2342, 0.1625],
    'epoch0_time': [0.8505, 4.0774, 0.9037],
    'epoch1_time': [0.1333, 3.2822, 0.1760],
}

df = pd.DataFrame(data)

# Create labels
labels = ['Cherry\n(4 batches)', 'Vanilla\n(4 batches)', 'Cherry\n(8 batches)']
x = np.arange(len(labels))
width = 0.6

# Color scheme
colors = ['#2ecc71', '#e74c3c', '#3498db']

# ============================================================
# Figure 1: Replication Factor Comparison
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

bars1 = ax1.bar(x, df['replication_factor'], width, color=colors, edgecolor='black', linewidth=1.2)

ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
ax1.set_ylabel('Replication Factor', fontsize=12, fontweight='bold')
ax1.set_title('Replication Factor Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# Add value labels on bars
for bar, val in zip(bars1, df['replication_factor']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylim(0, max(df['replication_factor']) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'replication_factor.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: replication_factor.png")

# ============================================================
# Figure 2: Edge Cut Comparison
# ============================================================
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

# Edge Cut (absolute)
bars2a = ax2a.bar(x, df['edge_cut'], width, color=colors, edgecolor='black', linewidth=1.2)
ax2a.set_xlabel('Method', fontsize=12, fontweight='bold')
ax2a.set_ylabel('Edge Cut (count)', fontsize=12, fontweight='bold')
ax2a.set_title('Edge Cut Comparison', fontsize=14, fontweight='bold')
ax2a.set_xticks(x)
ax2a.set_xticklabels(labels)
for bar, val in zip(bars2a, df['edge_cut']):
    ax2a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
             f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Edge Cut Ratio
bars2b = ax2b.bar(x, df['edge_cut_ratio'] * 100, width, color=colors, edgecolor='black', linewidth=1.2)
ax2b.set_xlabel('Method', fontsize=12, fontweight='bold')
ax2b.set_ylabel('Edge Cut Ratio (%)', fontsize=12, fontweight='bold')
ax2b.set_title('Edge Cut Ratio Comparison', fontsize=14, fontweight='bold')
ax2b.set_xticks(x)
ax2b.set_xticklabels(labels)
for bar, val in zip(bars2b, df['edge_cut_ratio'] * 100):
    ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'edge_cut.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: edge_cut.png")

# ============================================================
# Figure 3: Memory Usage Comparison
# ============================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))

bars3 = ax3.bar(x, df['peak_memory_gb'], width, color=colors, edgecolor='black', linewidth=1.2)

ax3.set_xlabel('Method', fontsize=12, fontweight='bold')
ax3.set_ylabel('Peak Memory (GB)', fontsize=12, fontweight='bold')
ax3.set_title('Peak Memory Usage Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)

for bar, val in zip(bars3, df['peak_memory_gb']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.4f} GB', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax3.set_ylim(0, max(df['peak_memory_gb']) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'memory_usage.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: memory_usage.png")

# ============================================================
# Figure 4: Training Time Comparison
# ============================================================
fig4, ax4 = plt.subplots(figsize=(10, 6))

bars4 = ax4.bar(x, df['epoch1_time'], width, color=colors, edgecolor='black', linewidth=1.2)

ax4.set_xlabel('Method', fontsize=12, fontweight='bold')
ax4.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax4.set_title('Epoch 1 Training Time Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(labels)

for bar, val in zip(bars4, df['epoch1_time']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4.set_ylim(0, max(df['epoch1_time']) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'training_time.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_time.png")

# ============================================================
# Figure 5: Comprehensive Dashboard
# ============================================================
fig5, axes = plt.subplots(2, 2, figsize=(14, 10))

# Replication Factor
ax = axes[0, 0]
bars = ax.bar(x, df['replication_factor'], width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Replication Factor', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Cherry\n(4b)', 'Vanilla\n(4b)', 'Cherry\n(8b)'])
for bar, val in zip(bars, df['replication_factor']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# Edge Cut Ratio
ax = axes[0, 1]
bars = ax.bar(x, df['edge_cut_ratio'] * 100, width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Edge Cut Ratio (%)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Cherry\n(4b)', 'Vanilla\n(4b)', 'Cherry\n(8b)'])
for bar, val in zip(bars, df['edge_cut_ratio'] * 100):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

# Memory Usage
ax = axes[1, 0]
bars = ax.bar(x, df['peak_memory_gb'], width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Peak Memory (GB)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Cherry\n(4b)', 'Vanilla\n(4b)', 'Cherry\n(8b)'])
for bar, val in zip(bars, df['peak_memory_gb']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}GB', ha='center', va='bottom', fontsize=9)

# Training Time (Epoch 1)
ax = axes[1, 1]
bars = ax.bar(x, df['epoch1_time'], width, color=colors, edgecolor='black', linewidth=1)
ax.set_title('Epoch 1 Training Time (s)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Cherry\n(4b)', 'Vanilla\n(4b)', 'Cherry\n(8b)'])
for bar, val in zip(bars, df['epoch1_time']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}s', ha='center', va='bottom', fontsize=9)

fig5.suptitle('Graph Partitioning Quality - Comprehensive Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'comprehensive_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: comprehensive_dashboard.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("All plots saved to partitioning_quality/")
print("="*60)
print("\nGenerated files:")
print("  - replication_factor.png")
print("  - edge_cut.png")
print("  - memory_usage.png")
print("  - training_time.png")
print("  - comprehensive_dashboard.png")
