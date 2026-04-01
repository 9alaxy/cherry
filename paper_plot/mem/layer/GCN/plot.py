import csv
import math
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
})

csv_path = "mem_cost.csv"
out_pdf = "mem_cost.pdf"

labels = []
mems = []

with open(csv_path, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    values = next(reader)

for label, v in zip(header[1:], values[1:]):
    labels.append(label)
    if v.strip().upper() == "OOM":
        mems.append(float("nan"))
    else:
        mems.append(float(v))

max_mem = max(m for m in mems if not math.isnan(m))
ylim_top = max_mem * 1.15

plt.figure(figsize=(6, 4))

# ColorBrewer Set2 (muted qualitative palette)
bar_color = "#66c2a5"
oom_color = "#d1d5db"

for i, (label, m) in enumerate(zip(labels, mems)):
    if math.isnan(m):
        oom_height = ylim_top
        plt.bar(i, oom_height, color=oom_color, edgecolor=oom_color, linewidth=1)
        plt.text(i, oom_height * 0.98, "OOM", ha="center", va="top", color="#ef4444", fontsize=12, fontweight="bold")
    else:
        plt.bar(i, m, color=bar_color)

plt.xlabel("Layer")
plt.ylabel("Memory (GB)")
plt.xticks(range(len(labels)), labels)
plt.ylim(0, ylim_top)
plt.tight_layout()
plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
plt.savefig("mem_cost.png", dpi=300, bbox_inches="tight")
