import os
import re
import csv

# 目标目录
log_dir = "./Evaluation/renduancy/ogbn-arxiv"
# 匹配Berry开头的log文件
log_files = [f for f in os.listdir(log_dir) if f.startswith("Berry") and f.endswith(".log")]

# 匹配Replication Factor的正则表达式
replication_pattern = re.compile(r"Replication Factor\s*[:=]\s*([\d.]+)", re.IGNORECASE)

results = []

for log_file in sorted(log_files):
    log_path = os.path.join(log_dir, log_file)
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
        match = replication_pattern.search(content)
        if match:
            replication_factor = float(match.group(1))
        else:
            replication_factor = None
    # 可从文件名中提取batch等参数
    results.append({
        "file": log_file,
        "replication_factor": replication_factor
    })

# 导出到CSV
csv_path = os.path.join(log_dir, "berry_replication_factor.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["file", "replication_factor"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"结果已导出到: {csv_path}")
