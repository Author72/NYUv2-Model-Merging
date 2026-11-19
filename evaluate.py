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
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    p.add_argument("--cpu_threads", type=int, default=6)
    p.add_argument(
        "--gpu_memory_fraction", type=float, default=0.25,
        help="PyTorch CUDA allocator cap; ignored on CPU and MPS."
    )
    p.add_argument(
        "--image_normalization", choices=["imagenet", "none"],
        default="imagenet",
        help="Must match the normalization used during training."
    )
    p.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use FP16 autocast on MPS."
    )
    p.add_argument("--csv", default="./runs/benchmark_results.csv")
    args = p.parse_args()

    device = configure_device(
        args.device, args.cpu_threads, args.gpu_memory_fraction
    )
    use_amp = args.amp and device.type == "mps"
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
        }
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        **loader_options
    )

    rows = []

    # STL: one independently trained expert per task.
    for task in TASKS:
        path = os.path.join(args.run_dir, "stl_" + task, "best.pth")
        model, _ = load_model(path, [task], device)
        metrics = evaluate(model, loader, [task], device, use_amp=use_amp)
        rows.append({"method": "STL", "task": task, **metrics})
        print("STL", task, metrics)
        del model
        gc.collect()
        release_accelerator_cache(device)

    # MTL: one shared encoder and three task heads.
    mtl_path = os.path.join(args.run_dir, "mtl", "best.pth")
    model, _ = load_model(mtl_path, TASKS, device)
    metrics = evaluate(model, loader, TASKS, device, use_amp=use_amp)
    for task in TASKS:
        rows.append({"method": "MTL", "task": task, **metrics})
    print("MTL", metrics)
    del model
    gc.collect()
    release_accelerator_cache(device)

    # Encoder weight averaging + task-specific heads.
    model, _ = load_model(args.merged, TASKS, device)
    metrics = evaluate(model, loader, TASKS, device, use_amp=use_amp)
    for task in TASKS:
        rows.append({"method": "WeightAveraging", "task": task, **metrics})
    print("WeightAveraging", metrics)

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    columns = sorted({k for row in rows for k in row.keys()})
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    shutdown_loader(loader)
    allocated, driver = device_memory_gb(device)
    print(
        f"[resource] final device memory: allocated={allocated:.2f} GB; "
        f"driver={driver:.2f} GB"
    )
    print(f"Saved benchmark CSV: {args.csv}")

if __name__ == "__main__":
    main()
