import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

plt.rcParams.update({
    "font.size": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "axes.unicode_minus": False,
})

# Try to force a CJK font so Chinese ylabel can be rendered in paper figures.
font_candidates = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
]
for fp in font_candidates:
    if fp.is_file():
        font_name = fm.FontProperties(fname=str(fp)).get_name()
        plt.rcParams["font.family"] = font_name
        break

csv_path = "replication-factor-plot-ogbn-arxiv.csv"
out_pdf = "replication_factor.pdf"
out_png = "replication_factor.png"

# Fixed x-axis categories for paper plot (uniform spacing)
x_order = [2, 4, 8, 32, 64]
x_to_pos = {x: i for i, x in enumerate(x_order)}

series = {}

with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        method = row["method"].strip()
        mb = int(row["micro_batch"])
        rf = float(row["replication_factor_for_plot"])
        if mb in x_to_pos:
            series.setdefault(method, []).append((mb, rf))

for method in series:
    series[method].sort(key=lambda x: x_order.index(x[0]))

plt.figure(figsize=(6, 4))

# ColorBrewer Set2 (muted qualitative palette), consistent with Evaluation/mem style.
method_colors = {
    "IODG-Full": "#66c2a5",
    "Betty": "#fc8d62",
}

for method in sorted(series.keys()):
    xs = [x_to_pos[x] for x, _ in series[method]]
    ys = [y for _, y in series[method]]
    plt.plot(
        xs,
        ys,
        marker="o",
        linewidth=2.2,
        markersize=7,
        label=method,
        color=method_colors.get(method, "#8da0cb"),
    )

plt.xlabel("微批次数")
plt.ylabel("冗余率")
plt.xticks(list(range(len(x_order))), [str(x) for x in x_order])
plt.legend(frameon=False, fontsize=16)
plt.tight_layout()
plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
plt.savefig(out_png, dpi=600, bbox_inches="tight")
