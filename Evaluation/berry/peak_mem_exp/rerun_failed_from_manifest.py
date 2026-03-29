#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
from pathlib import Path

import run_peak_memory_experiment as rpe


def now_tag() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser(description="Rerun failed rows from an existing manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device-number", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--python-bin", type=str, default=rpe.PYTHON_BIN_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))

    failed_rows = [r for r in rows if r.get("status") != "success"]

    tasks = []
    for r in failed_rows:
        dataset = r["dataset"]
        tasks.append(
            {
                "dataset": dataset,
                "model": r["model"],
                "method": r["method"],
                "seed": r["seed"],
                "fan_out": rpe.FANOUT_MAP[dataset],
                "num_epochs": "1",
                "num_batch": "8",
            }
        )

    print(f"[INFO] failed_rows={len(failed_rows)} rerun_tasks={len(tasks)} device={args.device_number}")

    out_rows = []
    for i, t in enumerate(tasks, 1):
        print(f"[RERUN {i}/{len(tasks)}] {t['method']} {t['model']} {t['dataset']} seed={t['seed']}")
        row = rpe.run_one(
            task=t,
            python_bin=args.python_bin,
            device_number=args.device_number,
            timeout_seconds=args.timeout_seconds,
            dry_run=args.dry_run,
        )
        print(f"[DONE] status={row['status']} exit={row['exit_code']} dur={row['duration_s']}s")
        out_rows.append(row)

    stamp = now_tag()
    out_manifest = rpe.LOG_DIR / f"manifest_rerun_failed_{stamp}.csv"
    out_fail = rpe.LOG_DIR / f"fail_manifest_rerun_failed_{stamp}.csv"
    rpe.write_manifest(out_rows, out_manifest)
    rpe.write_manifest([r for r in out_rows if r["status"] != "success"], out_fail)

    print(f"[RESULT] manifest={out_manifest}")
    print(f"[RESULT] fail_manifest={out_fail}")


if __name__ == "__main__":
    main()
