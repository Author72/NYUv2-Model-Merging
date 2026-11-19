import argparse
import os
import torch
from models import NYUModel, TASK_CHANNELS

TASKS = ["segmentation", "depth", "normal"]

def load_checkpoint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="./runs",
                   help="Directory containing stl_segmentation/best.pth etc.")
    p.add_argument("--out", default="./runs/merged_average.pth")
    args = p.parse_args()

    ckpts = {}
    for task in TASKS:
        path = os.path.join(args.run_dir, "stl_" + task, "best.pth")
        ckpts[task] = load_checkpoint(path)

    # Checkpoint keys are encoder.* and heads.<task>.*
    enc_states = [ckpts[t]["model"] for t in TASKS]
    merged_state = {}

    reference_keys = [k for k in enc_states[0] if k.startswith("encoder.")]
    for key in reference_keys:
        tensors = [state[key] for state in enc_states]
        for i, tensor in enumerate(tensors[1:], start=1):
            if tensor.shape != tensors[0].shape:
                raise RuntimeError(f"Encoder tensor shape mismatch for {key}")
        if tensors[0].is_floating_point():
            merged_state[key] = torch.stack(
                [x.float() for x in tensors], dim=0
            ).mean(dim=0).to(tensors[0].dtype)
        else:
            # Integer buffers such as num_batches_tracked cannot be averaged.
            merged_state[key] = torch.stack(tensors, dim=0).max(dim=0).values

    # Keep each task expert's own task-specific head.
    for task in TASKS:
        state = ckpts[task]["model"]
        prefix = f"heads.{task}."
        head_keys = [k for k in state if k.startswith(prefix)]
        if not head_keys:
            raise RuntimeError(f"No task head {prefix} found in {task} checkpoint")
        for key in head_keys:
            merged_state[key] = state[key]

    model = NYUModel(TASKS, pretrained_encoder=False)
    missing, unexpected = model.load_state_dict(merged_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when assembling merged model: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing keys when assembling merged model: {missing}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "tasks": TASKS,
        "method": "encoder_weight_average",
        "source_checkpoints": {
            task: os.path.join(args.run_dir, "stl_" + task, "best.pth")
            for task in TASKS
        },
    }, args.out)
    print(f"Saved merged model to: {args.out}")
    print("Encoder floating tensors were averaged; integer buffers used elementwise max.")
    print("Task heads were copied from their corresponding STL experts.")

if __name__ == "__main__":
    main()
