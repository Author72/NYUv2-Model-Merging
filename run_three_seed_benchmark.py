import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from aggregate_results import read_csv
from benchmark_metrics import METHOD_ORDER


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_seeds(value):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("Seeds must be unique")
    return seeds


def tail_text(path, limit=80):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-limit:])


def run_stage(
    name, script, script_args, logs_dir, marker_dir, env, resume,
    required_paths=(), reuse_existing=False,
):
    logs_dir.mkdir(parents=True, exist_ok=True)
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{name}.json"
    if resume and marker.is_file():
        print(f"[suite] skip completed stage: {name}", flush=True)
        return
    if (
        resume
        and reuse_existing
        and required_paths
        and all(path.is_file() and path.stat().st_size > 0 for path in required_paths)
    ):
        marker.write_text(
            json.dumps({
                "stage": name,
                "reused_existing": True,
                "required_paths": [str(path) for path in required_paths],
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[suite] reuse existing checkpoint for stage: {name}",
            flush=True,
        )
        return

    log_path = logs_dir / f"{name}.log"
    command = [sys.executable, str(PROJECT_ROOT / script), *script_args]
    print(f"[suite] start {name}", flush=True)
    print(f"[suite] log {log_path}", flush=True)
    start = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{' '.join(command)}\n"
        )
        log.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        print(tail_text(log_path), flush=True)
        raise RuntimeError(
            f"Stage {name} failed with exit code {result.returncode}"
        )
    elapsed = time.time() - start
    marker.write_text(
        json.dumps({
            "stage": name,
            "command": command,
            "elapsed_seconds": elapsed,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[suite] finish {name} in {elapsed / 60.0:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seeds", required=True, type=parse_seeds)
    parser.add_argument(
        "--split_manifest",
        default=str(PROJECT_ROOT / "splits/nyuv2_seed0_10pct.json"),
    )
    parser.add_argument("--init_encoder", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--distill_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--distill_lr", type=float, default=1e-3)
    parser.add_argument("--encoder_distill_lr", type=float, default=1e-4)
    parser.add_argument("--intermediate_weight", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device", choices=["cpu", "mps", "cuda", "auto"], default="cuda"
    )
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument(
        "--gpu_memory_fraction", type=float, default=0.90,
        help="Fraction of the selected NVIDIA GPU's memory allowed by PyTorch."
    )
    parser.add_argument(
        "--image_normalization",
        choices=["imagenet", "none"],
        default="imagenet",
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--reuse_existing_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Treat an existing stage output as complete even if its completion "
            "marker is absent."
        ),
    )
    parser.add_argument(
        "--dry_run", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.distill_epochs < 1:
        parser.error("Epoch budgets must be positive")
    if args.val_every < 1:
        parser.error("--val_every must be positive")

    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    split_manifest = Path(args.split_manifest).expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"Data root does not exist: {data_root}")
    if not split_manifest.is_file():
        parser.error(f"Split manifest does not exist: {split_manifest}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "NYUv2",
        "data_root": str(data_root),
        "split_manifest": str(split_manifest),
        "seeds": args.seeds,
        "seed_selection": "randomly sampled from 1..9999 with secrets.SystemRandom",
        "epochs": args.epochs,
        "distill_epochs": args.distill_epochs,
        "methods": list(METHOD_ORDER),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "distill_learning_rate": args.distill_lr,
        "encoder_distill_learning_rate": args.encoder_distill_lr,
        "device": args.device,
        "amp": args.amp,
        "amp_dtype": "float16" if args.amp else None,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "reuse_existing_checkpoints": args.reuse_existing_checkpoints,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "distillation_protocol": (
            "No calibration labels and no BN recalibration; output KD for "
            "EncoderAvg+OutputKD, and layer3/layer4 feature KD plus output "
            "KD for EncoderAvg+IntermediateKD+OutputKD."
        ),
    }
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "PYTORCH_CUDA_ALLOC_CONF": env.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
        "OMP_NUM_THREADS": str(args.cpu_threads),
        "MKL_NUM_THREADS": str(args.cpu_threads),
        "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
        "NUMEXPR_NUM_THREADS": str(args.cpu_threads),
    })
    amp_flag = "--amp" if args.amp else "--no-amp"

    common_train = [
        "--data_root", str(data_root),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--workers", str(args.workers),
        "--device", args.device,
        "--cpu_threads", str(args.cpu_threads),
        "--gpu_memory_fraction", str(args.gpu_memory_fraction),
        "--image_normalization", args.image_normalization,
        "--val_every", str(args.val_every),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--split_manifest", str(split_manifest),
        amp_flag,
    ]

    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        logs_dir = seed_dir / "logs"
        marker_dir = seed_dir / ".completed"
        logs_dir.mkdir(parents=True, exist_ok=True)
        marker_dir.mkdir(parents=True, exist_ok=True)
        seed_dir.mkdir(parents=True, exist_ok=True)

        if args.init_encoder:
            source = Path(args.init_encoder).expanduser().resolve()
            if not source.is_file():
                parser.error(f"Initial encoder does not exist: {source}")
            destination = seed_dir / "init_encoder.pth"
            if not destination.exists():
                shutil.copy2(source, destination)

        seed_common = [
            *common_train,
            "--out_dir", str(seed_dir),
            "--seed", str(seed),
        ]
        stage_outputs = {
            "01_stl_segmentation": [
                seed_dir / "stl_segmentation" / "best.pth",
            ],
            "02_stl_depth": [
                seed_dir / "stl_depth" / "best.pth",
            ],
            "03_stl_normal": [
                seed_dir / "stl_normal" / "best.pth",
            ],
            "04_mtl": [
                seed_dir / "mtl" / "best.pth",
            ],
            "05_encoder_average": [
                seed_dir / "merged_average.pth",
            ],
            "06_output_distillation": [
                seed_dir / "merged_output_distilled.pth",
            ],
            "07_intermediate_output_distillation": [
                seed_dir / "merged_intermediate_distilled.pth",
            ],
            "08_evaluate_five_methods": [
                seed_dir / "benchmark_results.csv",
            ],
        }
        stages = [
            (
                "01_stl_segmentation",
                "train.py",
                ["--mode", "stl", "--task", "segmentation", *seed_common],
            ),
            (
                "02_stl_depth",
                "train.py",
                ["--mode", "stl", "--task", "depth", *seed_common],
            ),
            (
                "03_stl_normal",
                "train.py",
                ["--mode", "stl", "--task", "normal", *seed_common],
            ),
            (
                "04_mtl",
                "train.py",
                ["--mode", "mtl", *seed_common],
            ),
            (
                "05_encoder_average",
                "merge_average.py",
                [
                    "--run_dir", str(seed_dir),
                    "--out", str(seed_dir / "merged_average.pth"),
                ],
            ),
            (
                "06_output_distillation",
                "distill_heads.py",
                [
                    "--data_root", str(data_root),
                    "--run_dir", str(seed_dir),
                    "--split_manifest", str(split_manifest),
                    "--merged", str(seed_dir / "merged_average.pth"),
                    "--out", str(seed_dir / "merged_output_distilled.pth"),
                    "--epochs", str(args.distill_epochs),
                    "--batch_size", str(args.batch_size),
                    "--lr", str(args.distill_lr),
                    "--weight_decay", str(args.weight_decay),
                    "--train_scope", "heads",
                    "--no-recalibrate_bn",
                    "--seg_sup_weight", "0",
                    "--depth_sup_weight", "0",
                    "--normal_sup_weight", "0",
                    "--method_name",
                    "encoder_average_output_distillation",
                    "--workers", str(args.workers),
                    "--device", args.device,
                    "--cpu_threads", str(args.cpu_threads),
                    "--gpu_memory_fraction", str(args.gpu_memory_fraction),
                    "--image_normalization", args.image_normalization,
                    "--seed", str(seed),
                    "--val_every", str(args.val_every),
                    amp_flag,
                ],
            ),
            (
                "07_intermediate_output_distillation",
                "distill_heads.py",
                [
                    "--data_root", str(data_root),
                    "--run_dir", str(seed_dir),
                    "--split_manifest", str(split_manifest),
                    "--merged", str(seed_dir / "merged_average.pth"),
                    "--out",
                    str(seed_dir / "merged_intermediate_distilled.pth"),
                    "--epochs", str(args.distill_epochs),
                    "--batch_size", str(args.batch_size),
                    "--lr", str(args.distill_lr),
                    "--weight_decay", str(args.weight_decay),
                    "--train_scope", "encoder_heads",
                    "--encoder_layers", "layer3,layer4",
                    "--encoder_lr", str(args.encoder_distill_lr),
                    "--intermediate_weight", str(args.intermediate_weight),
                    "--no-recalibrate_bn",
                    "--seg_sup_weight", "0",
                    "--depth_sup_weight", "0",
                    "--normal_sup_weight", "0",
                    "--method_name",
                    "encoder_average_intermediate_output_distillation",
                    "--workers", str(args.workers),
                    "--device", args.device,
                    "--cpu_threads", str(args.cpu_threads),
                    "--gpu_memory_fraction", str(args.gpu_memory_fraction),
                    "--image_normalization", args.image_normalization,
                    "--seed", str(seed),
                    "--val_every", str(args.val_every),
                    amp_flag,
                ],
            ),
            (
                "08_evaluate_five_methods",
                "evaluate.py",
                [
                    "--data_root", str(data_root),
                    "--run_dir", str(seed_dir),
                    "--merged", str(seed_dir / "merged_average.pth"),
                    "--output_distilled",
                    str(seed_dir / "merged_output_distilled.pth"),
                    "--intermediate_distilled",
                    str(seed_dir / "merged_intermediate_distilled.pth"),
                    "--batch_size", str(args.eval_batch_size),
                    "--workers", str(args.workers),
                    "--device", args.device,
                    "--cpu_threads", str(args.cpu_threads),
                    "--gpu_memory_fraction", str(args.gpu_memory_fraction),
                    "--image_normalization", args.image_normalization,
                    "--amp" if args.amp else "--no-amp",
                    "--seed", str(seed),
                    "--csv", str(seed_dir / "benchmark_results.csv"),
                    "--task_csv",
                    str(seed_dir / "benchmark_task_results.csv"),
                ],
            ),
        ]

        for stage_name, script, stage_args in stages:
            if args.dry_run:
                print(
                    f"[suite] dry run {seed}/{stage_name}: "
                    f"{script} {' '.join(stage_args)}"
                )
                continue
            run_stage(
                stage_name,
                script,
                stage_args,
                logs_dir,
                marker_dir,
                env,
                args.resume,
                required_paths=stage_outputs.get(stage_name, ()),
                reuse_existing=args.reuse_existing_checkpoints,
            )

    if args.dry_run:
        return

    for seed in args.seeds:
        result_path = output_root / f"seed_{seed}" / "benchmark_results.csv"
        result = read_csv(result_path)
        missing = set(METHOD_ORDER) - set(result)
        if missing:
            raise RuntimeError(
                f"Seed {seed} result is missing methods: {sorted(missing)}"
            )

    aggregate_args = [
        "--run_root", str(output_root),
        "--seeds", ",".join(str(seed) for seed in args.seeds),
        "--epochs", str(args.epochs),
    ]
    run_stage(
        "09_aggregate_results",
        "aggregate_results.py",
        aggregate_args,
        output_root / "logs",
        output_root / ".completed",
        env,
        args.resume,
    )
    print(
        f"[suite] complete: {output_root / 'three_seed_report.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
