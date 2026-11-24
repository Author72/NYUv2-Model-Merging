import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from benchmark_metrics import METHOD_ORDER, PRIMARY_METRICS


PERCENT_METRICS = {
    "seg_mIoU",
    "seg_pixAcc",
}


def parse_seeds(value):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("Seeds must be unique")
    return seeds


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No method rows found in {path}")
    methods = [row["method"] for row in rows]
    if len(methods) != len(set(methods)):
        raise RuntimeError(f"Duplicate methods found in {path}")
    missing = set(METHOD_ORDER) - set(methods)
    if missing:
        raise RuntimeError(f"{path} is missing methods: {sorted(missing)}")
    result = {}
    for row in rows:
        method = row["method"]
        if method not in METHOD_ORDER:
            continue
        result[method] = {
            metric: float(row[metric]) for metric in PRIMARY_METRICS
        }
        result[method]["delta_m_pct"] = float(row["delta_m_pct"])
    return result


def mean_std(values):
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def format_value(metric, mean, std):
    if metric in PERCENT_METRICS:
        mean *= 100.0
        std *= 100.0
        return f"{mean:.2f} ± {std:.2f}%"
    if metric == "delta_m_pct":
        return f"{mean:.2f} ± {std:.2f}%"
    return f"{mean:.4f} ± {std:.4f}"


def markdown_report(aggregate, seeds, epochs, methods):
    headers = [
        "Method",
        "Seg mIoU",
        "Seg Pix Acc",
        "Depth Abs",
        "Depth Rel",
        "Normal Mean",
        "Normal Median",
        "Δm%",
    ]
    lines = [
        "# NYUv2 Five-Method Benchmark",
        "",
        f"- Training seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- Epoch budget: `{epochs}` for each trained method",
        "- Values are mean ± sample standard deviation across seeds.",
        (
            "- Δm% is the mean relative improvement over STL across all "
            "six metrics; positive values are better."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    metric_labels = (
        "seg_mIoU",
        "seg_pixAcc",
        "depth_abs_err",
        "depth_rel_err",
        "normal_mean",
        "normal_median",
        "delta_m_pct",
    )
    for method in methods:
        cells = [method]
        for metric in metric_labels:
            mean, std = aggregate[method][metric]
            cells.append(format_value(metric, mean, std))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Higher is better for segmentation metrics and Δm%; lower is better "
        "for depth errors and normal angular errors.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--seeds", required=True, type=parse_seeds)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    seed_results = {}
    for seed in args.seeds:
        path = run_root / f"seed_{seed}" / "benchmark_results.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        seed_results[seed] = read_csv(path)

    aggregate = {}
    for method in METHOD_ORDER:
        aggregate[method] = {}
        for metric in (*PRIMARY_METRICS, "delta_m_pct"):
            values = [
                seed_results[seed][method][metric] for seed in args.seeds
            ]
            if any(not math.isfinite(value) for value in values):
                raise RuntimeError(f"Non-finite {metric} for {method}")
            aggregate[method][metric] = mean_std(values)

    csv_path = Path(args.csv or run_root / "aggregate_results.csv")
    json_path = Path(args.json or run_root / "aggregate_results.json")
    markdown_path = Path(
        args.markdown or run_root / "three_seed_report.md"
    )
    seed_csv_path = run_root / "all_seed_results.csv"
    for path in (csv_path, json_path, markdown_path, seed_csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    metric_columns = (*PRIMARY_METRICS, "delta_m_pct")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["method", "n_seeds"]
        for metric in metric_columns:
            fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHOD_ORDER:
            row = {"method": method, "n_seeds": len(args.seeds)}
            for metric in metric_columns:
                mean, std = aggregate[method][metric]
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            writer.writerow(row)

    with seed_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["method", "seed", *metric_columns]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHOD_ORDER:
            for seed in args.seeds:
                writer.writerow({
                    "method": method,
                    "seed": seed,
                    **seed_results[seed][method],
                })

    payload = {
        "dataset": "NYUv2",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "delta_m_definition": (
            "Mean relative improvement over STL across the six reported "
            "metrics, with sign normalized so positive is better."
        ),
        "metrics": list(metric_columns),
        "results": {
            method: {
                metric: {
                    "mean": aggregate[method][metric][0],
                    "std": aggregate[method][metric][1],
                }
                for metric in metric_columns
            }
            for method in METHOD_ORDER
        },
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        markdown_report(
            aggregate, args.seeds, args.epochs, METHOD_ORDER
        ),
        encoding="utf-8",
    )

    print(f"Saved aggregate CSV: {csv_path}")
    print(f"Saved aggregate JSON: {json_path}")
    print(f"Saved report: {markdown_path}")
    print(f"Saved all-seed CSV: {seed_csv_path}")


if __name__ == "__main__":
    main()
