import argparse
import json
import os
import random
from pathlib import Path


def npy_indices(path):
    indices = sorted(
        int(Path(name).stem)
        for name in os.listdir(path)
        if name.endswith(".npy")
    )
    if indices != list(range(len(indices))):
        raise RuntimeError(f"{path} does not contain contiguous 0.npy..N-1.npy")
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.fraction < 0.5:
        parser.error("--fraction must be between 0 and 0.5")
    out = Path(args.out)
    if out.exists() and not args.force:
        raise FileExistsError(
            f"{out} already exists; pass --force to overwrite it"
        )

    root = Path(args.data_root)
    train_indices = npy_indices(root / "train" / "image")
    val_indices = npy_indices(root / "val" / "image")
    shuffled = list(train_indices)
    random.Random(args.seed).shuffle(shuffled)
    calibration_count = max(1, round(len(train_indices) * args.fraction))
    calibration = sorted(shuffled[:calibration_count])
    calibration_set = set(calibration)
    train_pool = [index for index in train_indices if index not in calibration_set]

    payload = {
        "seed": args.seed,
        "fraction": args.fraction,
        "train_pool": train_pool,
        "calibration": calibration,
        "val": val_indices,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(
        f"Saved split to {out}: train_pool={len(train_pool)} "
        f"calibration={len(calibration)} val={len(val_indices)}"
    )


if __name__ == "__main__":
    main()
