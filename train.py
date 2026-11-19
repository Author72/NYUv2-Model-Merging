import argparse
import os
import random
import time
import numpy as np
from runtime import (
    autocast_context,
    configure_device,
    device_memory_gb,
    release_accelerator_cache,
    seed_device,
    shutdown_loader,
    synchronize_device,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nyuv2_dataset import NYUv2
from models import NYUModel, ResNet50Encoder, TASK_CHANNELS

TASKS = ["segmentation", "depth", "normal"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_device(seed)

def get_init_encoder(out_dir, pretrained):
    """
    Creates one common encoder initialization and reuses it for every STL/MTL run.
    This also makes --no-pretrained runs start from exactly the same random weights.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "init_encoder.pth")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    encoder = ResNet50Encoder(pretrained=pretrained)
    state = encoder.state_dict()
    torch.save(state, path)
    return state

def task_loss(task, pred, target):
    if task == "segmentation":
        return F.cross_entropy(pred, target.long(), ignore_index=-1)

    if task == "depth":
        # target shape Bx1xHxW; zero depth denotes invalid pixels.
        mask = (target.sum(dim=1, keepdim=True) != 0).float()
        denom = mask.sum().clamp_min(1.0)
        return (torch.abs(pred - target) * mask).sum() / denom

    if task == "normal":
        # NYUv2 ground-truth normals are unit vectors; zero vectors are invalid.
        pred = F.normalize(pred, p=2, dim=1, eps=1e-6)
        mask = (target.square().sum(dim=1, keepdim=True) > 0).float()
        denom = mask.sum().clamp_min(1.0)
        cosine = (pred * target).sum(dim=1, keepdim=True)
        return ((1.0 - cosine) * mask).sum() / denom

    raise ValueError(task)

@torch.inference_mode()
def evaluate(model, loader, tasks, device, use_amp=False):
    model.eval()
    # MPS and CUDA queue work asynchronously. Wait before reading validation
    # outputs when this function is called immediately after optimizer steps.
    synchronize_device(device)
    loss_sums = {t: 0.0 for t in tasks}
    n_batches = 0

    # Segmentation confusion matrix.
    # Keep histogram-style metric accumulation on CPU. This avoids relying on
    # sparse MPS operator coverage and keeps metric results identical per device.
    conf = torch.zeros(13, 13, dtype=torch.float64)
    depth_abs_sum = 0.0
    depth_rel_sum = 0.0
    depth_count = 0
    # Streaming angle statistics: avoid retaining every valid pixel angle in RAM.
    normal_hist = torch.zeros(18000, dtype=torch.int64)  # 0.01 degree bins
    normal_angle_sum = 0.0
    normal_angle_count = 0
    normal_hits = {11.25: 0, 22.5: 0, 30.0: 0}

    for image, targets in loader:
        image = image.to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            preds = model(image)
        synchronize_device(device)

        # Evaluate losses and metrics on CPU. Advanced indexing, reductions,
        # and normalization have produced inconsistent values on some MPS
        # versions. The network forward pass still runs on the accelerator.
        preds = {k: v.detach().float().cpu() for k, v in preds.items()}
        targets = {k: v.detach().cpu() for k, v in targets.items()}

        for task in tasks:
            loss_sums[task] += task_loss(task, preds[task], targets[task]).item()
        n_batches += 1

        if "segmentation" in tasks:
            gt = targets["segmentation"].long()
            pr = preds["segmentation"].argmax(dim=1)
            valid = (gt >= 0) & (gt < 13)
            inds = 13 * gt[valid] + pr[valid]
            conf += torch.bincount(inds, minlength=169).reshape(13, 13).double()

        if "depth" in tasks:
            gt = targets["depth"]
            pr = preds["depth"]
            mask = (gt > 0) & torch.isfinite(gt)
            if mask.any():
                diff = torch.abs(pr[mask] - gt[mask])
                depth_abs_sum += diff.sum().item()
                depth_rel_sum += (diff / gt[mask].clamp_min(1e-8)).sum().item()
                depth_count += int(mask.sum().item())

        if "normal" in tasks:
            gt = targets["normal"]
            pr = F.normalize(preds["normal"], p=2, dim=1, eps=1e-6)
            mask = gt.square().sum(dim=1) > 0
            if mask.any():
                dot = (pr * gt).sum(dim=1).clamp(-1.0, 1.0)
                angles = torch.rad2deg(torch.acos(dot[mask])).float()
                normal_angle_sum += angles.sum().item()
                normal_angle_count += int(angles.numel())
                for threshold in normal_hits:
                    normal_hits[threshold] += int((angles < threshold).sum().item())
                bins = torch.clamp((angles * 100).long(), min=0, max=17999)
                normal_hist += torch.bincount(bins, minlength=18000)

    result = {"loss_" + t: loss_sums[t] / max(n_batches, 1) for t in tasks}

    if "segmentation" in tasks:
        diag = conf.diag()
        denom = conf.sum(1) + conf.sum(0) - diag
        iou = diag / denom.clamp_min(1.0)
        result["seg_mIoU"] = iou.mean().item()
        result["seg_pixAcc"] = (diag.sum() / conf.sum().clamp_min(1.0)).item()

    if "depth" in tasks:
        result["depth_abs_err"] = depth_abs_sum / max(depth_count, 1)
        result["depth_rel_err"] = depth_rel_sum / max(depth_count, 1)

    if "normal" in tasks:
        if normal_angle_count:
            result["normal_mean"] = normal_angle_sum / normal_angle_count
            median_bin = int(torch.searchsorted(
                normal_hist.cumsum(0),
                torch.tensor((normal_angle_count + 1) // 2, dtype=torch.int64)
            ).item())
            result["normal_median"] = (median_bin + 0.5) / 100.0
            for threshold, hits in normal_hits.items():
                result[f"normal_acc_{threshold}"] = hits / normal_angle_count

    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["stl", "mtl"], required=True)
    p.add_argument("--task", choices=TASKS, default=None, help="Required for STL.")
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="./runs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate. The default is scaled linearly from 1e-4 at batch 2."
    )
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device", choices=["cpu", "mps", "cuda", "auto"], default="auto",
        help="auto prefers Apple MPS, then CUDA, then CPU."
    )
    p.add_argument("--cpu_threads", type=int, default=6)
    p.add_argument(
        "--gpu_memory_fraction", type=float, default=0.25,
        help="PyTorch CUDA allocator cap; ignored on CPU and MPS."
    )
    p.add_argument(
        "--image_normalization", choices=["imagenet", "none"],
        default="imagenet",
        help="Normalize RGB input for the ImageNet-pretrained encoder."
    )
    p.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use FP16 autocast on MPS."
    )
    p.add_argument(
        "--val_every", type=int, default=2,
        help="Validate every N epochs; the first and final epochs always validate."
    )
    p.add_argument(
        "--save_last", action=argparse.BooleanOptionalAction, default=False,
        help="Also save last.pth after validated epochs."
    )
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    if args.mode == "stl" and args.task is None:
        p.error("--task is required when --mode stl")

    set_seed(args.seed)
    device = configure_device(
        args.device, args.cpu_threads, args.gpu_memory_fraction
    )
    if args.val_every < 1:
        p.error("--val_every must be at least 1")
    use_amp = args.amp and device.type == "mps"
    print(
        f"[resource] device={device}; workers={args.workers}; "
        f"batch_size={args.batch_size}; amp={use_amp}; val_every={args.val_every}"
    )
    os.makedirs(args.out_dir, exist_ok=True)

    train_set = NYUv2(
        args.data_root, "train",
        image_normalization=args.image_normalization
    )
    val_set = NYUv2(
        args.data_root, "val",
        image_normalization=args.image_normalization
    )
    loader_options = {}
    if args.workers > 0:
        loader_options = {
            "prefetch_factor": 4,
        }
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=True, **loader_options
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        **loader_options
    )

    # Build/reuse a single common initialization for every run.
    init_state = get_init_encoder(args.out_dir, args.pretrained)
    tasks = TASKS if args.mode == "mtl" else [args.task]
    model = NYUModel(tasks, pretrained_encoder=False)
    model.encoder.load_state_dict(init_state, strict=True)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(args.epochs // 3, 1), gamma=0.5
    )

    run_name = "mtl" if args.mode == "mtl" else "stl_" + args.task
    save_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(save_dir, "best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        samples_seen = 0
        pbar = tqdm(train_loader, desc=f"{run_name} epoch {epoch}/{args.epochs}")
        for image, targets in pbar:
            image = image.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            with autocast_context(device, use_amp):
                preds = model(image)
            # Keep masked task losses and reductions in FP32.
            losses = [
                task_loss(t, preds[t].float(), targets[t])
                for t in tasks
            ]
            loss = torch.stack(losses).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_value = loss.detach().item()
            running_loss += loss_value * image.size(0)
            samples_seen += image.size(0)
            pbar.set_postfix(train_loss=f"{loss_value:.4f}")

        synchronize_device(device)
        scheduler.step()
        train_avg = running_loss / max(samples_seen, 1)
        epoch_seconds = time.perf_counter() - epoch_start
        should_validate = (
            epoch == 1
            or epoch == args.epochs
            or epoch % args.val_every == 0
        )
        if not should_validate:
            print(
                f"[{run_name}] epoch={epoch} train_loss={train_avg:.5f} "
                f"train_samples/s={samples_seen / epoch_seconds:.2f}"
            )
            continue

        val_metrics = evaluate(
            model, val_loader, tasks, device, use_amp=use_amp
        )
        val_objective = sum(val_metrics["loss_" + t] for t in tasks) / len(tasks)

        print(
            f"[{run_name}] epoch={epoch} train_loss={train_avg:.5f} "
            f"train_samples/s={samples_seen / epoch_seconds:.2f} "
            f"val_objective={val_objective:.5f} metrics={val_metrics}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "tasks": tasks,
            "epoch": epoch,
            "seed": args.seed,
            "val_metrics": val_metrics,
            "config": vars(args),
        }
        if args.save_last:
            torch.save(checkpoint, os.path.join(save_dir, "last.pth"))
        if val_objective < best_val:
            best_val = val_objective
            torch.save(checkpoint, best_path)

    shutdown_loader(train_loader)
    shutdown_loader(val_loader)
    release_accelerator_cache(device)
    allocated, driver = device_memory_gb(device)
    print(
        f"[resource] final device memory: allocated={allocated:.2f} GB; "
        f"driver={driver:.2f} GB"
    )
    print(f"Finished. Best checkpoint: {best_path}")

if __name__ == "__main__":
    main()
