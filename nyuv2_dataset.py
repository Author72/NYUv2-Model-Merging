import os
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_NET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGE_NET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

class NYUv2(Dataset):
    """
    LibMTL/MTAN preprocessed NYUv2 format:
      root/
        train/{image,label,depth,normal}/0.npy ...
        val/{image,label,depth,normal}/0.npy ...
    image: HxWx3; label: HxW; depth: HxWx1; normal: HxWx3.
    """
    def __init__(self, root, split="train", image_normalization="imagenet"):
        self.root = os.path.expanduser(root)
        self.split = split
        if image_normalization not in ("imagenet", "none"):
            raise ValueError(
                "image_normalization must be either 'imagenet' or 'none'"
            )
        self.image_normalization = image_normalization
        self.base = os.path.join(self.root, split)
        for sub in ("image", "label", "depth", "normal"):
            p = os.path.join(self.base, sub)
            if not os.path.isdir(p):
                raise FileNotFoundError(
                    f"Missing directory: {p}\n"
                    "Expected the preprocessed LibMTL/MTAN NYUv2 directory layout."
                )
        self.files = sorted(
            [f for f in os.listdir(os.path.join(self.base, "image")) if f.endswith(".npy")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        if not self.files:
            raise RuntimeError(f"No .npy images found in {os.path.join(self.base, 'image')}")

        # Require contiguous numeric indices because the official preprocessing uses 0.npy...N.npy.
        indices = [int(os.path.splitext(f)[0]) for f in self.files]
        if indices != list(range(len(indices))):
            raise RuntimeError(
                f"{split}/image filenames must be contiguous 0.npy ... N-1.npy; "
                f"found first/last indices {indices[:3]} ... {indices[-3:]}"
            )

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _to_chw(arr, expected_channels=None):
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC/ HW array, got shape {arr.shape}")
        if expected_channels is not None and arr.shape[-1] != expected_channels:
            # Accept CHW arrays too, in case the local copy was saved channel-first.
            if arr.shape[0] == expected_channels:
                return torch.from_numpy(np.ascontiguousarray(arr)).float()
            raise ValueError(f"Expected {expected_channels} channels, got shape {arr.shape}")
        return torch.from_numpy(np.ascontiguousarray(np.moveaxis(arr, -1, 0))).float()

    def __getitem__(self, idx):
        stem = os.path.splitext(self.files[idx])[0]
        def load(folder):
            return np.load(os.path.join(self.base, folder, stem + ".npy"), allow_pickle=False)

        image_np = load("image")
        image = self._to_chw(image_np, 3)
        # Handle raw uint8 images as well as preprocessed float images.
        if image.max() > 1.5:
            image = image / 255.0
        if self.image_normalization == "imagenet":
            image = (image - IMAGE_NET_MEAN) / IMAGE_NET_STD

        label_np = np.asarray(load("label"))
        if label_np.ndim == 3 and label_np.shape[-1] == 1:
            label_np = label_np[..., 0]
        label = torch.from_numpy(np.ascontiguousarray(label_np)).long()

        depth = self._to_chw(load("depth"), 1)
        normal = self._to_chw(load("normal"), 3)

        # Dataset labels use zero vectors / zero depth for invalid pixels and -1 for
        # ignored semantic pixels, as in the LibMTL NYUv2 implementation.
        return image, {
            "segmentation": label,
            "depth": depth,
            "normal": normal,
        }
