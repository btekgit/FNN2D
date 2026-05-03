import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate per-run benchmark JSON files into CSV and Markdown summaries.")
    parser.add_argument("--input-dir", type=Path, default=Path("./results_spatial_alc2d_test1/"), help="Directory containing per-run JSON files.")
    parser.add_argument("--output-prefix", type=Path, default=Path("./results_spatial_alc2d_test1/summary"))
    return parser.parse_args()


def safe_std(values):
    return stdev(values) if len(values) > 1 else 0.0


def load_runs(input_dir):
    runs = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("summary"):
            continue
        runs.append(json.loads(path.read_text()))
    if not runs:
        raise FileNotFoundError(f"No run JSON files found in {input_dir}")
    return runs


def aggregate_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["dataset"], run["method"])].append(run)

    rows = []
    for (dataset, method), items in sorted(grouped.items()):
        best_accs = [item["best_test_acc"] for item in items]
        final_accs = [item["final_test_acc"] for item in items]
        params = [item["parameter_count"] for item in items]
        epoch_times = [item["epoch_seconds_mean"] for item in items]
        train_times = [item["train_seconds_total"] for item in items]
        eval_times = [item["eval_seconds_mean"] for item in items if "eval_seconds_mean" in item]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(items),
                "best_acc_mean": mean(best_accs),
                "best_acc_std": safe_std(best_accs),
                "final_acc_mean": mean(final_accs),
                "final_acc_std": safe_std(final_accs),
                "best_single_acc": max(best_accs),
                "params_mean": int(mean(params)),
                "epoch_seconds_mean": mean(epoch_times),
                "train_seconds_mean": mean(train_times),
                "inference_seconds_mean": mean(eval_times) if eval_times else 0.0,
                "inference_seconds_std": safe_std(eval_times) if eval_times else 0.0,
            }
        )
    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "runs",
        "best_acc_mean",
        "best_acc_std",
        "final_acc_mean",
        "final_acc_std",
        "best_single_acc",
        "params_mean",
        "epoch_seconds_mean",
        "train_seconds_mean",
        "inference_seconds_mean",
        "inference_seconds_std",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    lines = ["# Spatial Benchmark Summary", ""]
    for dataset, dataset_rows in sorted(grouped.items()):
        dataset_rows = sorted(dataset_rows, key=lambda item: item["best_acc_mean"], reverse=True)
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| method | runs | best acc mean±std | final acc mean±std | best single | params | sec/epoch | total sec | infer sec/eval mean±std |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in dataset_rows:
            lines.append(
                "| {method} | {runs} | {best_acc_mean:.2f} ± {best_acc_std:.2f} | "
                "{final_acc_mean:.2f} ± {final_acc_std:.2f} | {best_single_acc:.2f} | {params_mean} | "
                "{epoch_seconds_mean:.2f} | {train_seconds_mean:.1f} | "
                "{inference_seconds_mean:.2f} ± {inference_seconds_std:.2f} |".format(**row)
            )
        lines.append("")
    output_path.write_text("\n".join(lines))


def main():
    args = parse_args()
    runs = load_runs(args.input_dir)
    rows = aggregate_runs(runs)
    write_csv(rows, args.output_prefix.with_suffix(".csv"))
    write_markdown(rows, args.output_prefix.with_suffix(".md"))
    print(f"Wrote {args.output_prefix.with_suffix('.csv')}")
    print(f"Wrote {args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()