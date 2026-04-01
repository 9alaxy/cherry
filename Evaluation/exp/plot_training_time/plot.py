import argparse
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

# Keep Chinese labels renderable in paper figures.
font_candidates = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
]
for fp in font_candidates:
    if fp.is_file():
        font_name = fm.FontProperties(fname=str(fp)).get_name()
        plt.rcParams["font.family"] = font_name
        break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training time bar chart from summary CSV.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default="epoch-train-time-summary-ogbn-arxiv.csv",
        help="Input CSV path. Must contain method, micro_batch, mean_after_warmup_sec.",
    )
    parser.add_argument("--out-pdf", default="training_time_bar.pdf")
    parser.add_argument("--out-png", default="training_time_bar.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"].strip()
            mb = int(row["micro_batch"])
            t = float(row["mean_after_warmup_sec"])
            rows.append((method, mb, t))

    x_order = [2, 4, 8, 16, 32, 64]

    # method -> {micro_batch: value}
    table = {"Berry": {}, "REG": {}}
    for method, mb, t in rows:
        if method in table:
            table[method][mb] = t

    x = list(range(len(x_order)))
    width = 0.36

    plt.figure(figsize=(6, 4))

    # ColorBrewer Set2, same style family as plot_replication_factor.
    method_colors = {
        "Berry": "#66c2a5",
        "REG": "#fc8d62",
    }

    berry_vals = [table["Berry"].get(mb, float("nan")) for mb in x_order]
    reg_vals = [table["REG"].get(mb, float("nan")) for mb in x_order]

    # Left bar: REG (Betty), Right bar: Berry (IODG-full)
    plt.bar(
        [i - width / 2 for i in x],
        reg_vals,
        width=width,
        color=method_colors["REG"],
        label="Betty",
    )
    plt.bar(
        [i + width / 2 for i in x],
        berry_vals,
        width=width,
        color=method_colors["Berry"],
        label="IODG-full",
    )

    plt.xlabel("微批次数")
    plt.ylabel("时间（秒）")
    plt.xticks(x, [str(v) for v in x_order])
    plt.legend(frameon=False, fontsize=16)
    plt.tight_layout()
    plt.savefig(args.out_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(args.out_png, dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
