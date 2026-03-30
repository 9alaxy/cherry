#!/usr/bin/env /root/miniconda3/envs/cherry/bin/python
import argparse
import csv
import math
import os
import re
from typing import Dict, List, Optional

RUN_LINE_RE = re.compile(
    r"Run\s+\d+\s*\|\s*Epoch\s+\d+\s*\|\s*Loss\s+[-+0-9.eE]+\s*\|\s*Train\s+([-+0-9.eE]+)\s*\|\s*Val\s+([-+0-9.eE]+)\s*\|\s*Test\s+([-+0-9.eE]+)"
)
PEAK_RE = re.compile(r"max memory allocated:\s*([-+0-9.eE]+)\s*GB")
TIME_RE = re.compile(r"total_time:\s*([-+0-9.eE]+)")
FINAL_BATCH_RE = re.compile(r"Final num_batch after partition:\s*(\d+)\s*\(initial\s*(\d+)\)")
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")
CUDA_OOM_RE = re.compile(r"out of memory", re.IGNORECASE)


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def parse_main_log(path: str) -> Dict[str, object]:
    result: Dict[str, object] = {
        "peak_train_gb": math.nan,
        "time_sum_s": math.nan,
        "final_train_acc": math.nan,
        "final_val_acc": math.nan,
        "final_test_acc": math.nan,
        "final_num_batch": math.nan,
        "initial_num_batch": math.nan,
        "has_traceback": False,
        "has_oom": False,
        "run_finished": False,
    }
    if not path or not os.path.exists(path):
        return result

    peaks: List[float] = []
    times: List[float] = []
    last_acc = None
    final_batch = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[INFO] Train run finished" in line:
                result["run_finished"] = True
            if TRACEBACK_RE.search(line):
                result["has_traceback"] = True
            if CUDA_OOM_RE.search(line):
                result["has_oom"] = True

            m = PEAK_RE.search(line)
            if m:
                peaks.append(_to_float(m.group(1)))

            m = TIME_RE.search(line)
            if m:
                times.append(_to_float(m.group(1)))

            m = RUN_LINE_RE.search(line)
            if m:
                last_acc = (_to_float(m.group(1)), _to_float(m.group(2)), _to_float(m.group(3)))

            m = FINAL_BATCH_RE.search(line)
            if m:
                final_batch = (_to_float(m.group(1)), _to_float(m.group(2)))

    if peaks:
        result["peak_train_gb"] = max(peaks)
    if times:
        result["time_sum_s"] = sum(times)
    if last_acc is not None:
        result["final_train_acc"], result["final_val_acc"], result["final_test_acc"] = last_acc
    if final_batch is not None:
        result["final_num_batch"], result["initial_num_batch"] = final_batch

    return result


def parse_gpu_log(path: str) -> Dict[str, object]:
    result: Dict[str, object] = {
        "gpu_peak_mib": math.nan,
        "gpu_peak_gb": math.nan,
    }
    if not path or not os.path.exists(path):
        return result

    peak_mib = 0.0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "memory.used" in line:
                continue
            m = re.search(r"(\d+)\s*MiB", line)
            if not m:
                continue
            value = float(m.group(1))
            if value > peak_mib:
                peak_mib = value

    if peak_mib > 0:
        result["gpu_peak_mib"] = peak_mib
        result["gpu_peak_gb"] = peak_mib / 1024.0

    return result


def pct(delta: float, base: float) -> float:
    if base == 0 or math.isnan(base) or math.isnan(delta):
        return math.nan
    return 100.0 * delta / base


def pass_flag(cond: Optional[bool]) -> str:
    if cond is None:
        return "N/A"
    return "PASS" if cond else "FAIL"


def bool_or_none(v: bool) -> Optional[bool]:
    return v


def format_float(v: float, digits: int = 4) -> str:
    if v is None or math.isnan(v):
        return ""
    return f"{v:.{digits}f}"


def summarize_model(rows: List[Dict[str, object]]) -> Dict[str, object]:
    indexed = {r["run_id"]: r for r in rows}

    cherry = None
    metis = None
    berry_relaxed = None
    berry_tight = None

    for r in rows:
        run_id = str(r["run_id"])
        if run_id.endswith("_cherry_baseline"):
            cherry = r
        elif run_id.endswith("_metis_baseline"):
            metis = r
        elif run_id.endswith("_berry_relaxed"):
            berry_relaxed = r
        elif run_id.endswith("_berry_tight"):
            berry_tight = r

    out: Dict[str, object] = {}
    out["model"] = rows[0]["model"] if rows else ""

    cherry_peak = float(cherry.get("peak_train_gb", math.nan)) if cherry else math.nan
    metis_peak = float(metis.get("peak_train_gb", math.nan)) if metis else math.nan
    berry_relaxed_peak = float(berry_relaxed.get("peak_train_gb", math.nan)) if berry_relaxed else math.nan
    berry_tight_peak = float(berry_tight.get("peak_train_gb", math.nan)) if berry_tight else math.nan
    berry_tight_budget = float(berry_tight.get("budget_gb", math.nan)) if berry_tight else math.nan

    out["cherry_peak_gb"] = cherry_peak
    out["metis_peak_gb"] = metis_peak
    out["berry_relaxed_peak_gb"] = berry_relaxed_peak
    out["berry_tight_peak_gb"] = berry_tight_peak
    out["berry_tight_budget_gb"] = berry_tight_budget

    out["berry_relaxed_mem_drop_vs_cherry_pct"] = pct(cherry_peak - berry_relaxed_peak, cherry_peak)
    out["berry_tight_mem_drop_vs_cherry_pct"] = pct(cherry_peak - berry_tight_peak, cherry_peak)

    cherry_acc = float(cherry.get("final_test_acc", math.nan)) if cherry else math.nan
    berry_relaxed_acc = float(berry_relaxed.get("final_test_acc", math.nan)) if berry_relaxed else math.nan
    out["delta_test_acc_pct_point"] = (berry_relaxed_acc - cherry_acc) * 100.0 if not math.isnan(cherry_acc) and not math.isnan(berry_relaxed_acc) else math.nan

    cherry_time = float(cherry.get("time_sum_s", math.nan)) if cherry else math.nan
    berry_relaxed_time = float(berry_relaxed.get("time_sum_s", math.nan)) if berry_relaxed else math.nan
    out["time_ratio_berry_relaxed_vs_cherry"] = (
        berry_relaxed_time / cherry_time if cherry_time and not math.isnan(cherry_time) and not math.isnan(berry_relaxed_time) else math.nan
    )

    baseline_over_budget = None
    if not math.isnan(berry_tight_budget):
        baseline_peaks = [v for v in [cherry_peak, metis_peak] if not math.isnan(v)]
        if baseline_peaks:
            baseline_over_budget = any(v > berry_tight_budget for v in baseline_peaks)

    berry_tight_within_budget = None
    if not math.isnan(berry_tight_budget) and not math.isnan(berry_tight_peak):
        berry_tight_within_budget = berry_tight_peak <= berry_tight_budget

    c1 = None
    mem_drop = out["berry_tight_mem_drop_vs_cherry_pct"]
    if not math.isnan(mem_drop):
        c1 = mem_drop >= 10.0

    c2 = None
    if baseline_over_budget is not None and berry_tight_within_budget is not None:
        c2 = baseline_over_budget and berry_tight_within_budget

    c3 = None
    delta_acc = out["delta_test_acc_pct_point"]
    if not math.isnan(delta_acc):
        c3 = delta_acc >= -1.0

    c4 = None
    ratio = out["time_ratio_berry_relaxed_vs_cherry"]
    if not math.isnan(ratio):
        c4 = ratio <= 1.25

    out["criterion_1_mem_drop_ge_10pct"] = pass_flag(c1)
    out["criterion_2_tight_budget_trainable"] = pass_flag(c2)
    out["criterion_3_acc_drop_le_1pp"] = pass_flag(c3)
    out["criterion_4_time_ratio_le_1_25"] = pass_flag(c4)

    return out


def write_markdown(path: str, rows: List[Dict[str, object]], model_summary: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Memory-aware Partition Effectiveness Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append("| run_id | model | method | memory_aware | budget_gb | init_batch | final_batch | peak_train_gb | gpu_peak_gb | time_sum_s | final_test_acc | status |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        status = "ok"
        if r.get("has_traceback") or r.get("has_oom"):
            status = "error"
        elif not r.get("run_finished"):
            status = "incomplete"

        lines.append(
            "| {run_id} | {model} | {method} | {memory_aware} | {budget} | {init_batch} | {final_batch} | {peak} | {gpu_peak} | {time_sum} | {test_acc} | {status} |".format(
                run_id=r.get("run_id", ""),
                model=r.get("model", ""),
                method=r.get("method", ""),
                memory_aware=r.get("memory_aware", ""),
                budget=format_float(float(r.get("budget_gb", math.nan)), 4),
                init_batch=format_float(float(r.get("num_batch_init", math.nan)), 0),
                final_batch=format_float(float(r.get("final_num_batch", math.nan)), 0),
                peak=format_float(float(r.get("peak_train_gb", math.nan)), 4),
                gpu_peak=format_float(float(r.get("gpu_peak_gb", math.nan)), 4),
                time_sum=format_float(float(r.get("time_sum_s", math.nan)), 4),
                test_acc=format_float(float(r.get("final_test_acc", math.nan)), 4),
                status=status,
            )
        )

    lines.append("")
    lines.append("## Criteria Check")
    lines.append("")
    lines.append("| model | mem_drop_tight_vs_cherry_pct | tight_budget_trainable | delta_test_acc_pp | time_ratio_relaxed_vs_cherry | C1(mem>=10%) | C2(budget) | C3(acc) | C4(time) |")
    lines.append("|---|---:|---|---:|---:|---|---|---|---|")

    for s in model_summary:
        tight_peak = float(s.get("berry_tight_peak_gb", math.nan))
        tight_budget = float(s.get("berry_tight_budget_gb", math.nan))
        budget_ok = ""
        if not math.isnan(tight_peak) and not math.isnan(tight_budget):
            budget_ok = "yes" if tight_peak <= tight_budget else "no"

        lines.append(
            "| {model} | {mem_drop} | {budget_ok} | {delta_acc} | {time_ratio} | {c1} | {c2} | {c3} | {c4} |".format(
                model=s.get("model", ""),
                mem_drop=format_float(float(s.get("berry_tight_mem_drop_vs_cherry_pct", math.nan)), 2),
                budget_ok=budget_ok,
                delta_acc=format_float(float(s.get("delta_test_acc_pct_point", math.nan)), 2),
                time_ratio=format_float(float(s.get("time_ratio_berry_relaxed_vs_cherry", math.nan)), 3),
                c1=s.get("criterion_1_mem_drop_ge_10pct", "N/A"),
                c2=s.get("criterion_2_tight_budget_trainable", "N/A"),
                c3=s.get("criterion_3_acc_drop_le_1pp", "N/A"),
                c4=s.get("criterion_4_time_ratio_le_1_25", "N/A"),
            )
        )

    lines.append("")
    all_pass = all(
        s.get("criterion_1_mem_drop_ge_10pct") == "PASS"
        and s.get("criterion_2_tight_budget_trainable") == "PASS"
        and s.get("criterion_3_acc_drop_le_1pp") == "PASS"
        and s.get("criterion_4_time_ratio_le_1_25") == "PASS"
        for s in model_summary
    ) if model_summary else False
    lines.append("## Final Verdict")
    lines.append("")
    lines.append("PASS" if all_pass else "FAIL")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize memory-aware effectiveness experiments.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row: Dict[str, object] = dict(r)
            row["memory_aware"] = int(r["memory_aware"])
            row["budget_gb"] = _to_float(r["budget_gb"])
            row["num_batch_init"] = _to_float(r["num_batch_init"])
            row["num_epochs"] = _to_float(r["num_epochs"])

            main_metrics = parse_main_log(r["main_log"])
            gpu_metrics = parse_gpu_log(r["gpu_log"])

            row.update(main_metrics)
            row.update(gpu_metrics)
            rows.append(row)

    fieldnames = [
        "run_id",
        "model",
        "method",
        "memory_aware",
        "budget_gb",
        "num_batch_init",
        "num_epochs",
        "role",
        "main_log",
        "gpu_log",
        "peak_train_gb",
        "gpu_peak_gb",
        "time_sum_s",
        "final_train_acc",
        "final_val_acc",
        "final_test_acc",
        "initial_num_batch",
        "final_num_batch",
        "has_traceback",
        "has_oom",
        "run_finished",
    ]

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        grouped.setdefault(str(r["model"]), []).append(r)

    model_summary = [summarize_model(grouped[m]) for m in sorted(grouped.keys())]

    # Write model summary next to detailed csv for convenient script consumption.
    summary_sidecar = os.path.splitext(args.output_csv)[0] + "_model_summary.csv"
    with open(summary_sidecar, "w", encoding="utf-8", newline="") as f:
        fieldnames_summary = [
            "model",
            "cherry_peak_gb",
            "metis_peak_gb",
            "berry_relaxed_peak_gb",
            "berry_tight_peak_gb",
            "berry_tight_budget_gb",
            "berry_relaxed_mem_drop_vs_cherry_pct",
            "berry_tight_mem_drop_vs_cherry_pct",
            "delta_test_acc_pct_point",
            "time_ratio_berry_relaxed_vs_cherry",
            "criterion_1_mem_drop_ge_10pct",
            "criterion_2_tight_budget_trainable",
            "criterion_3_acc_drop_le_1pp",
            "criterion_4_time_ratio_le_1_25",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames_summary)
        writer.writeheader()
        for s in model_summary:
            writer.writerow({k: s.get(k, "") for k in fieldnames_summary})

    write_markdown(args.output_md, rows, model_summary)


if __name__ == "__main__":
    main()
