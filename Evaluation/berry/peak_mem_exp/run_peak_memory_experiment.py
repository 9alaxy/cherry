#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
LOG_DIR = EXP_DIR / "log"
PYTHON_BIN_DEFAULT = "/root/miniconda3/envs/cherry/bin/python"

METHOD_MAP = {
    "Berry": {"entry": "micro_batch_train_berry.py", "args": ["--selection-method", "Berry"]},
    "DGL_random": {"entry": "micro_batch_train_berry.py", "args": ["--selection-method", "Random"]},
    "DGL_metis": {"entry": "micro_batch_train_berry.py", "args": ["--selection-method", "Metis"]},
    "Betty": {"entry": "Betty.py", "args": ["--selection-method", "REG", "--re-partition-method", "REG"]},
}

DATASETS = ["reddit", "ogbn-arxiv", "ogbn-products", "amazon", "ogbn-papers100M", "cora"]
MODELS = ["SAGE", "GCN", "GAT"]
METHODS = ["Berry", "Betty", "DGL_random", "DGL_metis"]
FANOUT_MAP = {
    "reddit": "10,25,30",
    "ogbn-arxiv": "10,25,30",
    "ogbn-products": "10,25,30",
    "amazon": "10,25,30",
    "ogbn-papers100M": "10,25,30",
    "cora": "10,25,30",
    "karate": "5,5,5",
}


def now_tag() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def build_model_cfg(model: str) -> Dict[str, str]:
    if model == "SAGE":
        return {"num_hidden": "256", "aggre": "pool", "num_heads": "4"}
    if model == "GCN":
        return {"num_hidden": "256", "aggre": "mean", "num_heads": "4"}
    if model == "GAT":
        return {"num_hidden": "128", "aggre": "mean", "num_heads": "4"}
    raise ValueError(f"Unsupported model: {model}")


def is_oom_text(text: str) -> bool:
    t = text.lower()
    keys = [
        "out of memory",
        "cuda error: out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
    ]
    return any(k in t for k in keys)


def start_gpu_sampler(gpu_log_path: Path, device_number: int):
    with gpu_log_path.open("w", encoding="utf-8") as gpu_f:
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                str(device_number),
                "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv",
                "-l",
                "1",
            ],
            stdout=gpu_f,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
        )
    return sampler


def stop_process(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def run_one(
    task: Dict[str, str],
    python_bin: str,
    device_number: int,
    timeout_seconds: int,
    dry_run: bool,
) -> Dict[str, object]:
    method = task["method"]
    model = task["model"]
    dataset = task["dataset"]
    seed = int(task["seed"])
    fan_out = task["fan_out"]
    num_epochs = int(task["num_epochs"])
    num_batch = int(task["num_batch"])


    model_cfg = build_model_cfg(model)
    method_cfg = METHOD_MAP[method]

    tag = f"{method}_{model}_{dataset}_seed{seed}_{now_tag()}"
    main_log = LOG_DIR / f"{tag}.train.log"
    gpu_log = LOG_DIR / f"{tag}.gpu.log"
    meta_json = LOG_DIR / f"{tag}.meta.json"

    entry_script = ROOT / method_cfg["entry"]
    cmd = [
        python_bin,
        str(entry_script),
        "--dataset",
        dataset,
        "--model",
        model,
        "--aggre",
        model_cfg["aggre"],
        "--seed",
        str(seed),
        "--setseed",
        "True",
        "--GPUmem",
        "True",
        "--num-batch",
        str(num_batch),
        "--num-runs",
        "1",
        "--num-epochs",
        str(num_epochs),
        "--num-layers",
        "3",
        "--num-hidden",
        model_cfg["num_hidden"],
        "--dropout",
        "0.5",
        "--fan-out",
        fan_out,
        "--device-number",
        str(device_number),
        "--num-heads",
        model_cfg["num_heads"],
    ] + method_cfg["args"]

    if method != "Betty":
        cmd += ["--num-workers", "0"]

    start_utc = dt.datetime.utcnow().isoformat() + "Z"
    t0 = time.time()
    result = {
        "dataset": dataset,
        "model": model,
        "method": method,
        "seed": seed,
        "status": "not_run",
        "main_log": str(main_log),
        "gpu_log": str(gpu_log),
        "meta_json": str(meta_json),
        "start_utc": start_utc,
        "end_utc": "",
        "duration_s": None,
        "exit_code": None,
        "is_timeout": False,
        "is_oom": False,
        "cmd": cmd,
    }

    if dry_run:
        result["status"] = "dry_run"
        result["end_utc"] = dt.datetime.utcnow().isoformat() + "Z"
        result["duration_s"] = 0.0
        meta_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    sampler = None
    proc = None
    try:
        try:
            sampler = start_gpu_sampler(gpu_log, device_number)
        except Exception as e:
            gpu_log.write_text(f"nvidia-smi start failed: {e}\n", encoding="utf-8")

        with main_log.open("w", encoding="utf-8") as f:
            f.write(f"[INFO] START {tag}\n")
            f.write(f"[INFO] CMD {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
                preexec_fn=os.setsid,
            )
            try:
                exit_code = proc.wait(timeout=timeout_seconds)
                result["exit_code"] = exit_code
                result["status"] = "success" if exit_code == 0 else "failed"
            except subprocess.TimeoutExpired:
                result["is_timeout"] = True
                result["status"] = "timeout"
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    time.sleep(2)
                    if proc.poll() is None:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                result["exit_code"] = -9

        if main_log.exists():
            txt = main_log.read_text(encoding="utf-8", errors="ignore")
            result["is_oom"] = is_oom_text(txt)
            if result["is_oom"] and result["status"] == "failed":
                result["status"] = "oom"
    finally:
        if sampler is not None:
            stop_process(sampler)

    result["duration_s"] = round(time.time() - t0, 3)
    result["end_utc"] = dt.datetime.utcnow().isoformat() + "Z"
    meta_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_manifest(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "dataset",
        "model",
        "method",
        "seed",
        "status",
        "start_utc",
        "end_utc",
        "duration_s",
        "exit_code",
        "is_timeout",
        "is_oom",
        "main_log",
        "gpu_log",
        "meta_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k) for k in headers}
            writer.writerow(out)


def build_tasks(phase: str, runs: int, epochs: int, num_batch: int, datasets: List[str]) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []
    if phase == "smoke":
        # Tiny graphs (e.g., karate) can fail under metis when k > #train nodes.
        # Keep smoke partition count conservative so all method pipelines can run through.
        smoke_num_batch = min(num_batch, 2)
        for method in METHODS:
            tasks.append(
                {
                    "dataset": "karate",
                    "model": "GCN",
                    "method": method,
                    "seed": "1236",
                    "fan_out": FANOUT_MAP["karate"],
                    "num_epochs": str(epochs),
                    "num_batch": str(smoke_num_batch),
                }
            )
        return tasks

    for dataset in datasets:
        for model in MODELS:
            for method in METHODS:
                for run_idx in range(runs):
                    tasks.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "method": method,
                            "seed": str(1236 + run_idx),
                            "fan_out": FANOUT_MAP[dataset],
                            "num_epochs": str(epochs),
                            "num_batch": str(num_batch),
                        }
                    )
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Peak memory experiment controller")
    parser.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-batch", type=int, default=8)
    parser.add_argument("--device-number", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--python-bin", type=str, default=PYTHON_BIN_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = parser.parse_args()

    selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.phase == "full" and not selected_datasets:
        raise ValueError("--datasets is empty for full phase")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args.phase, args.runs, args.epochs, args.num_batch, selected_datasets)

    all_rows: List[Dict[str, object]] = []
    print(f"[INFO] phase={args.phase} tasks={len(tasks)} device={args.device_number}")
    for idx, task in enumerate(tasks, 1):
        print(
            "[RUN {}/{}] {} {} {} seed={}".format(
                idx,
                len(tasks),
                task["method"],
                task["model"],
                task["dataset"],
                task["seed"],
            )
        )
        row = run_one(
            task=task,
            python_bin=args.python_bin,
            device_number=args.device_number,
            timeout_seconds=args.timeout_seconds,
            dry_run=args.dry_run,
        )
        print(
            "[DONE] status={} exit={} dur={}s".format(
                row["status"],
                row["exit_code"],
                row["duration_s"],
            )
        )
        all_rows.append(row)

    stamp = now_tag()
    manifest_path = LOG_DIR / f"manifest_{args.phase}_{stamp}.csv"
    fail_path = LOG_DIR / f"fail_manifest_{args.phase}_{stamp}.csv"
    write_manifest(all_rows, manifest_path)
    write_manifest([r for r in all_rows if r["status"] != "success"], fail_path)

    print(f"[RESULT] manifest={manifest_path}")
    print(f"[RESULT] fail_manifest={fail_path}")


if __name__ == "__main__":
    main()
