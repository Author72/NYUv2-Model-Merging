import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import NYUModel
from nyuv2_dataset import NYUv2
from runtime import (
    autocast_context,
    configure_device,
    device_memory_gb,
    release_accelerator_cache,
    seed_device,
    shutdown_loader,
    synchronize_device,
)
from train import TASKS, evaluate, task_loss


class CachedDistillDataset(Dataset):
    def __init__(self, images, targets, teacher_outputs, teacher_intermediates):
        self.images = images
        self.targets = targets
        self.teacher_outputs = teacher_outputs
        self.teacher_intermediates = teacher_intermediates

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return (
            self.images[index],
            {task: values[index] for task, values in self.targets.items()},
            {
                task: values[index]
                for task, values in self.teacher_outputs.items()
            },
            {
                task: {
                    layer: values[index]
                    for layer, values in layers.items()
                }
                for task, layers in self.teacher_intermediates.items()
            },
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_device(seed)


def load_checkpoint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_teacher(path, task, device):
    checkpoint = load_checkpoint(path)
    if checkpoint.get("tasks") != [task]:
        raise ValueError(f"{path} is not a {task} STL checkpoint")
    model = NYUModel([task], pretrained_encoder=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def load_merged_student(path, device):
    checkpoint = load_checkpoint(path)
    if checkpoint.get("tasks") != TASKS:
        raise ValueError(f"{path} does not contain all three tasks")
    model = NYUModel(TASKS, pretrained_encoder=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), checkpoint


def check_normalization(checkpoint, expected, path):
    actual = checkpoint.get("config", {}).get("image_normalization", "none")
    if actual != expected:
        raise ValueError(
            f"{path} was trained with image_normalization={actual}, "
            f"but distillation uses {expected}"
        )


def cache_calibration_data(
    data_root, calibration_indices, teacher_paths, normalization,
    device, batch_size, use_amp, cache_intermediates=False
):
    dataset = NYUv2(
        data_root, "train",
        image_normalization=normalization,
        indices=calibration_indices,
    )
    images = []
    targets = {task: [] for task in TASKS}
    for index in range(len(dataset)):
        image, target = dataset[index]
        images.append(image)
        for task in TASKS:
            targets[task].append(target[task])

    teacher_outputs = {}
    teacher_intermediates = {}
    for task in TASKS:
        print(f"[teacher] caching {task} outputs on {len(images)} images")
        teacher, checkpoint = load_teacher(teacher_paths[task], task, device)
        check_normalization(checkpoint, normalization, teacher_paths[task])
        task_outputs = []
        task_intermediates = {"layer3": [], "layer4": []}
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                image_batch = torch.stack(
                    images[start:start + batch_size]
                ).to(device, non_blocking=True)
                with autocast_context(device, use_amp):
                    features, intermediates = teacher.encoder(
                        image_batch, return_intermediate=True
                    )
                    prediction = teacher.heads[task](
                        features, image_batch.shape[-2:]
                    )
                synchronize_device(device)
                task_outputs.extend(
                    tensor.detach().float().cpu()
                    for tensor in prediction
                )
                if cache_intermediates:
                    for layer in ("layer3", "layer4"):
                        task_intermediates[layer].extend(
                            tensor.detach().float().cpu()
                            for tensor in intermediates[layer]
                        )
        teacher_outputs[task] = task_outputs
        if cache_intermediates:
            teacher_intermediates[task] = task_intermediates
        del teacher
        release_accelerator_cache(device)
    return CachedDistillDataset(
        images, targets, teacher_outputs, teacher_intermediates
    )


def recalibrate_bn(model, dataset, device, batch_size, passes, use_amp):
    if passes <= 0:
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.encoder.train()
    with torch.no_grad():
        for pass_index in range(passes):
            for images, _, _, _ in tqdm(
                loader, desc=f"BN calibration {pass_index + 1}/{passes}"
            ):
                images = images.to(device, non_blocking=True)
                with autocast_context(device, use_amp):
                    model.encoder(images)
                synchronize_device(device)
    shutdown_loader(loader)
    model.encoder.eval()


def set_student_training_mode(model, train_scope, encoder_layers):
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    if train_scope == "heads":
        model.encoder.eval()
    else:
        model.encoder.train()
        trainable_names = set(encoder_layers)
        for name, module in model.encoder.named_children():
            module.train() if name in trainable_names else module.eval()
        for name, parameter in model.encoder.named_parameters():
            if name.split(".", 1)[0] in trainable_names:
                parameter.requires_grad_(True)
    for head in model.heads.values():
        head.train()


def intermediate_loss(student_intermediates, teacher_intermediates):
    totals = []
    for task in TASKS:
        for layer in ("layer3", "layer4"):
            student = F.normalize(
                student_intermediates[layer].float(),
                p=2,
                dim=1,
                eps=1e-6,
            )
            teacher = F.normalize(
                teacher_intermediates[task][layer].float(),
                p=2,
                dim=1,
                eps=1e-6,
            )
            totals.append(
                (1.0 - (student * teacher).sum(dim=1)).mean()
            )
    return torch.stack(totals).mean()


def distillation_losses(
    predictions, teacher_outputs, targets, temperature,
    seg_sup_weight, seg_kd_weight,
    depth_teacher_weight, depth_sup_weight,
    normal_teacher_weight, normal_sup_weight,
):
    student_logits = predictions["segmentation"].float()
    teacher_logits = teacher_outputs["segmentation"].float()
    seg_gt = targets["segmentation"].long()
    seg_valid = seg_gt >= 0
    seg_sup = F.cross_entropy(
        student_logits, seg_gt, ignore_index=-1
    )
    student_log_probs = F.log_softmax(
        student_logits / temperature, dim=1
    )
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    kd_map = F.kl_div(
        student_log_probs, teacher_probs, reduction="none"
    ).sum(dim=1)
    seg_kd = kd_map[seg_valid].mean() * temperature**2
    seg_loss = seg_sup_weight * seg_sup + seg_kd_weight * seg_kd

    student_depth = predictions["depth"].float()
    teacher_depth = teacher_outputs["depth"].float()
    depth_valid = torch.isfinite(teacher_depth)
    depth_teacher = F.smooth_l1_loss(
        student_depth[depth_valid], teacher_depth[depth_valid]
    )
    depth_gt = targets["depth"].float()
    depth_gt_valid = (depth_gt > 0) & depth_gt.isfinite()
    depth_sup = (
        torch.abs(student_depth[depth_gt_valid] - depth_gt[depth_gt_valid])
        .mean()
        if depth_gt_valid.any()
        else student_depth.sum() * 0.0
    )
    depth_loss = (
        depth_teacher_weight * depth_teacher
        + depth_sup_weight * depth_sup
    )

    student_normal = F.normalize(
        predictions["normal"].float(), p=2, dim=1, eps=1e-6
    )
    teacher_normal = F.normalize(
        teacher_outputs["normal"].float(), p=2, dim=1, eps=1e-6
    )
    normal_teacher_map = 1.0 - (
        student_normal * teacher_normal
    ).sum(dim=1)
    normal_teacher = normal_teacher_map.mean()
    normal_gt = targets["normal"].float()
    normal_gt_valid = normal_gt.square().sum(dim=1) > 0
    normal_sup = (
        torch.abs(
            student_normal.permute(0, 2, 3, 1)[normal_gt_valid]
            - normal_gt.permute(0, 2, 3, 1)[normal_gt_valid]
        ).mean()
        if normal_gt_valid.any()
        else student_normal.sum() * 0.0
    )
    normal_loss = (
        normal_teacher_weight * normal_teacher
        + normal_sup_weight * normal_sup
    )
    return {
        "segmentation": seg_loss,
        "depth": depth_loss,
        "normal": normal_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--merged", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seg_sup_weight", type=float, default=1.0)
    parser.add_argument("--seg_kd_weight", type=float, default=1.0)
    parser.add_argument("--depth_teacher_weight", type=float, default=1.0)
    parser.add_argument("--depth_sup_weight", type=float, default=0.2)
    parser.add_argument("--normal_teacher_weight", type=float, default=1.0)
    parser.add_argument("--normal_sup_weight", type=float, default=0.2)
    parser.add_argument(
        "--train_scope", choices=["heads", "encoder_heads"],
        default="heads"
    )
    parser.add_argument(
        "--encoder_layers", default="layer3,layer4",
        help="Comma-separated encoder children to unfreeze."
    )
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--intermediate_weight", type=float, default=0.2)
    parser.add_argument(
        "--recalibrate_bn", action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument("--bn_calibration_passes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device", choices=["cpu", "mps", "cuda", "auto"], default="auto"
    )
    parser.add_argument("--cpu_threads", type=int, default=6)
    parser.add_argument("--gpu_memory_fraction", type=float, default=0.25)
    parser.add_argument(
        "--image_normalization", choices=["imagenet", "none"],
        default="imagenet"
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument(
        "--method_name", default=None,
        help="Optional method label stored in the output checkpoint."
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.val_every < 1:
        parser.error("--epochs and --val_every must be positive")
    set_seed(args.seed)
    device = configure_device(
        args.device, args.cpu_threads, args.gpu_memory_fraction
    )
    use_amp = args.amp and device.type == "mps"
    if args.method_name:
        method_name = args.method_name
    elif args.train_scope == "encoder_heads":
        method_name = "encoder_average_intermediate_output_distillation"
    elif max(
        args.seg_sup_weight,
        args.depth_sup_weight,
        args.normal_sup_weight,
    ) == 0:
        method_name = "encoder_weight_average_output_distillation"
    else:
        method_name = "encoder_weight_average_head_distillation"

    with open(args.split_manifest, encoding="utf-8") as handle:
        split = json.load(handle)
    train_pool = set(split["train_pool"])
    calibration_indices = split["calibration"]
    if train_pool & set(calibration_indices):
        raise ValueError("Calibration indices overlap with the teacher train pool")

    merged_path = args.merged or os.path.join(
        args.run_dir, "merged_average.pth"
    )
    out_path = args.out or os.path.join(
        args.run_dir, "merged_distilled.pth"
    )
    teacher_paths = {
        task: os.path.join(args.run_dir, "stl_" + task, "best.pth")
        for task in TASKS
    }
    student, _ = load_merged_student(merged_path, device)
    encoder_layers = [
        layer.strip() for layer in args.encoder_layers.split(",")
        if layer.strip()
    ]
    valid_layers = {
        name for name, _ in student.encoder.named_children()
    }
    invalid_layers = set(encoder_layers) - valid_layers
    if invalid_layers:
        parser.error(f"Unknown encoder layers: {sorted(invalid_layers)}")
    if args.train_scope == "encoder_heads" and not encoder_layers:
        parser.error("--encoder_layers cannot be empty for encoder_heads")
    set_student_training_mode(
        student, args.train_scope, encoder_layers
    )

    dataset = cache_calibration_data(
        args.data_root, calibration_indices, teacher_paths,
        args.image_normalization, device, args.batch_size, use_amp,
        cache_intermediates=args.train_scope == "encoder_heads",
    )
    if args.recalibrate_bn:
        recalibrate_bn(
            student, dataset, device, args.batch_size,
            args.bn_calibration_passes, use_amp
        )
    set_student_training_mode(
        student, args.train_scope, encoder_layers
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_dataset = NYUv2(
        args.data_root, "val",
        image_normalization=args.image_normalization,
        indices=split["val"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )
    parameter_groups = [{
        "params": list(student.heads.parameters()),
        "lr": args.lr,
    }]
    if args.train_scope == "encoder_heads":
        encoder_parameters = [
            parameter
            for parameter in student.encoder.parameters()
            if parameter.requires_grad
        ]
        parameter_groups.append({
            "params": encoder_parameters,
            "lr": args.encoder_lr,
        })
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    best_objective = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        set_student_training_mode(
            student, args.train_scope, encoder_layers
        )
        running = {task: 0.0 for task in TASKS}
        batches = 0
        start_time = time.perf_counter()
        progress = tqdm(
            loader,
            desc=f"head distillation {epoch}/{args.epochs}",
        )
        for images, targets, teacher_outputs, teacher_intermediates in progress:
            images = images.to(device, non_blocking=True)
            with autocast_context(device, use_amp):
                if args.train_scope == "encoder_heads":
                    predictions, student_intermediates = (
                        student.forward_with_features(images)
                    )
                else:
                    predictions = student(images)
                    student_intermediates = None
            synchronize_device(device)
            targets = {
                task: value.to(device, non_blocking=True)
                for task, value in targets.items()
            }
            teacher_outputs = {
                task: value.to(device, non_blocking=True)
                for task, value in teacher_outputs.items()
            }
            teacher_intermediates = {
                task: {
                    layer: value.to(device, non_blocking=True)
                    for layer, value in layers.items()
                }
                for task, layers in teacher_intermediates.items()
            }
            losses = distillation_losses(
                predictions,
                teacher_outputs,
                targets,
                args.temperature,
                args.seg_sup_weight,
                args.seg_kd_weight,
                args.depth_teacher_weight,
                args.depth_sup_weight,
                args.normal_teacher_weight,
                args.normal_sup_weight,
            )
            feature_loss = None
            if args.train_scope == "encoder_heads":
                feature_loss = intermediate_loss(
                    student_intermediates, teacher_intermediates
                )
                losses["intermediate"] = feature_loss
            loss = (
                sum(losses[task] for task in TASKS) / len(TASKS)
                + args.intermediate_weight * (
                    feature_loss
                    if feature_loss is not None
                    else torch.zeros((), device=device)
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in student.parameters()
                    if parameter.requires_grad
                ],
                max_norm=1.0,
            )
            optimizer.step()
            for task in TASKS:
                running[task] += losses[task].detach().item()
            batches += 1
            progress.set_postfix(
                loss=f"{loss.detach().item():.4f}"
            )
        scheduler.step()
        synchronize_device(device)
        seconds = time.perf_counter() - start_time
        averages = {
            task: running[task] / max(batches, 1) for task in TASKS
        }
        print(
            f"[distill] epoch={epoch} losses={averages} "
            f"samples/s={len(dataset) / seconds:.2f}"
        )

        if epoch % args.val_every != 0 and epoch != args.epochs:
            continue
        metrics = evaluate(
            student, val_loader, TASKS, device, use_amp=use_amp
        )
        objective = sum(
            metrics["loss_" + task] for task in TASKS
        ) / len(TASKS)
        print(f"[distill] validation={metrics}")
        if objective < best_objective:
            best_objective = objective
            checkpoint = {
                "model": student.state_dict(),
                "tasks": TASKS,
                "method": method_name,
                "epoch": epoch,
                "val_metrics": metrics,
                "config": vars(args),
                "source_checkpoints": teacher_paths,
                "merged_checkpoint": merged_path,
                "split_manifest": args.split_manifest,
                "calibration_size": len(calibration_indices),
                "encoder_layers": encoder_layers,
                "intermediate_weight": args.intermediate_weight,
            }
            torch.save(checkpoint, out_path)

    shutdown_loader(loader)
    shutdown_loader(val_loader)
    release_accelerator_cache(device)
    allocated, driver = device_memory_gb(device)
    print(
        f"[resource] final device memory: allocated={allocated:.2f} GB; "
        f"driver={driver:.2f} GB"
    )
    print(f"Saved distilled model to: {out_path}")


if __name__ == "__main__":
    main()
