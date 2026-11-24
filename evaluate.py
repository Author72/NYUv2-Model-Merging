import argparse
import csv
import os
import gc
from runtime import (
    configure_device,
    device_memory_gb,
    release_accelerator_cache,
    shutdown_loader,
)
import torch
from torch.utils.data import DataLoader

from nyuv2_dataset import NYUv2
from models import NYUModel
from train import evaluate, TASKS
from benchmark_metrics import (
    METHOD_ORDER,
    PRIMARY_METRICS,
    delta_m_pct,
)

def load_model(path, tasks, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = NYUModel(tasks, pretrained_encoder=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device), ckpt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--mode", choices=["all"], default="all")
    p.add_argument("--run_dir", default="./runs")
    p.add_argument("--merged", default="./runs/merged_average.pth")
    p.add_argument("--distilled", default=None)
    p.add_argument("--output_distilled", default=None)
    p.add_argument("--intermediate_distilled", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cuda")
    p.add_argument("--cpu_threads", type=int, default=8)
    p.add_argument(
        "--gpu_memory_fraction", type=float, default=0.90,
        help="PyTorch CUDA allocator cap; ignored on CPU and MPS."
    )
    p.add_argument(
        "--image_normalization", choices=["imagenet", "none"],
        default="imagenet",
        help="Must match the normalization used during training."
    )
    p.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use FP16 autocast on CUDA or MPS."
    )
    p.add_argument("--csv", default="./runs/benchmark_results.csv")
    p.add_argument(
        "--task_csv", default=None,
        help="Optional diagnostic CSV with one row per method and task."
    )
    args = p.parse_args()

    device = configure_device(
        args.device, args.cpu_threads, args.gpu_memory_fraction
    )
    use_amp = args.amp and device.type in ("cuda", "mps")
    print(
        f"[resource] device={device}; threads={args.cpu_threads}; "
        f"batch_size={args.batch_size}; workers={args.workers}; amp={use_amp}"
    )
    dataset = NYUv2(
        args.data_root, "val",
        image_normalization=args.image_normalization
    )
    loader_options = {}
    if args.workers > 0:
        loader_options = {
            "prefetch_factor": 4,
            "persistent_workers": True,
        }
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        **loader_options
    )

    method_rows = []
    task_rows = []

    def add_method_metrics(method, metrics):
        method_rows.append({
            "method": method,
            "task": "all",
            "seed": args.seed,
            **{metric: metrics[metric] for metric in PRIMARY_METRICS},
        })

    # STL: one independently trained expert per task.
    stl_method_metrics = {}
    for task in TASKS:
        path = os.path.join(args.run_dir, "stl_" + task, "best.pth")
        model, _ = load_model(path, [task], device)
        metrics = evaluate(model, loader, [task], device, use_amp=use_amp)
        task_rows.append({
            "method": "STL", "task": task, "seed": args.seed, **metrics,
        })
        stl_method_metrics.update({
            metric: metrics[metric]
            for metric in PRIMARY_METRICS
            if metric in metrics
        })
        print("STL", task, metrics)
        del model
        gc.collect()
        release_accelerator_cache(device)
    missing_stl = set(PRIMARY_METRICS) - set(stl_method_metrics)
    if missing_stl:
        raise RuntimeError(f"Missing STL baseline metrics: {sorted(missing_stl)}")
    add_method_metrics("STL", stl_method_metrics)

    # MTL: one shared encoder and three task heads.
    mtl_path = os.path.join(args.run_dir, "mtl", "best.pth")
    model, _ = load_model(mtl_path, TASKS, device)
    metrics = evaluate(model, loader, TASKS, device, use_amp=use_amp)
    for task in TASKS:
        task_rows.append({
            "method": "MTL", "task": task, "seed": args.seed, **metrics,
        })
    add_method_metrics("MTL", metrics)
    print("MTL", metrics)
    del model
    gc.collect()
    release_accelerator_cache(device)

    # Encoder weight averaging + task-specific heads.
    model, _ = load_model(args.merged, TASKS, device)
    metrics = evaluate(model, loader, TASKS, device, use_amp=use_amp)
    for task in TASKS:
        task_rows.append({
            "method": "EncoderAvg", "task": task, "seed": args.seed, **metrics,
        })
    add_method_metrics("EncoderAvg", metrics)
    print("EncoderAvg", metrics)
    del model
    gc.collect()
    release_accelerator_cache(device)

    if args.distilled:
        model, _ = load_model(args.distilled, TASKS, device)
        metrics = evaluate(model, loader, TASKS, device, use_amp=use_amp)
        for task in TASKS:
            task_rows.append({
                "method": "EncoderAvg+KD", "task": task,
                "seed": args.seed, **metrics,
            })
        print("EncoderAvg+KD", metrics)
        del model
        gc.collect()
        release_accelerator_cache(device)

    if args.output_distilled:
        model, _ = load_model(args.output_distilled, TASKS, device)
        metrics = evaluate(model, loader, TASKS, device, use_amp=use_amp)
        for task in TASKS:
            task_rows.append({
                "method": "EncoderAvg+OutputKD",
                "task": task,
                "seed": args.seed,
                **metrics,
            })
        add_method_metrics("EncoderAvg+OutputKD", metrics)
        print("EncoderAvg+OutputKD", metrics)
        del model
        gc.collect()
        release_accelerator_cache(device)

    if args.intermediate_distilled:
        model, _ = load_model(args.intermediate_distilled, TASKS, device)
        metrics = evaluate(model, loader, TASKS, device, use_amp=use_amp)
        for task in TASKS:
            task_rows.append({
                "method": "EncoderAvg+IntermediateKD+OutputKD",
                "task": task,
                "seed": args.seed,
                **metrics,
            })
        add_method_metrics(
            "EncoderAvg+IntermediateKD+OutputKD", metrics
        )
        print("EncoderAvg+IntermediateKD+OutputKD", metrics)
        del model
        gc.collect()
        release_accelerator_cache(device)

    stl_metrics = next(
        row for row in method_rows if row["method"] == "STL"
    )
    for row in method_rows:
        row["delta_m_pct"] = delta_m_pct(row, stl_metrics)

    present_methods = [row["method"] for row in method_rows]
    method_rows.sort(
        key=lambda row: METHOD_ORDER.index(row["method"])
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    columns = ["method", "task", "seed", *PRIMARY_METRICS, "delta_m_pct"]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(method_rows)

    task_csv_path = args.task_csv or os.path.join(
        os.path.dirname(os.path.abspath(args.csv)),
        "benchmark_task_results.csv",
    )
    task_columns = sorted({key for row in task_rows for key in row})
    with open(task_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=task_columns)
        writer.writeheader()
        writer.writerows(task_rows)

    shutdown_loader(loader)
    allocated, driver = device_memory_gb(device)
    print(
        f"[resource] final device memory: allocated={allocated:.2f} GB; "
        f"driver={driver:.2f} GB"
    )
    print(f"Evaluated methods: {', '.join(present_methods)}")
    print(
        "delta_m_pct = mean relative improvement over STL across all six "
        "metrics; positive values are better."
    )
    print(f"Saved benchmark CSV: {args.csv}")
    print(f"Saved per-task diagnostic CSV: {task_csv_path}")

if __name__ == "__main__":
    main()
