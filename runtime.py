import contextlib
import os
import platform

# Keep the optional MPS fallback for cross-platform debugging. CUDA is the
# default production target.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def is_mps_available():
    """Return whether PyTorch can use Metal in this process."""
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_built() and backend.is_available())
    except RuntimeError:
        return False


def get_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_arg == "mps" and not is_mps_available():
        raise RuntimeError(
            "MPS was requested but is not available to this PyTorch process. "
            "Use a native arm64 Python/PyTorch build on macOS 14 or newer, "
            "or run with --device cpu."
        )
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Check the NVIDIA driver, "
            "CUDA_VISIBLE_DEVICES, and that the installed PyTorch wheel is a "
            "CUDA build."
        )
    return torch.device(device_arg)


def configure_device(device_arg, cpu_threads, gpu_memory_fraction):
    """Configure process-level settings and return the selected device."""
    torch.set_num_threads(max(1, cpu_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # The inter-op thread count can only be set before parallel work starts.
        pass

    device = get_device(device_arg)
    if device.type == "cuda":
        if not (0.05 <= gpu_memory_fraction <= 1.0):
            raise ValueError("--gpu_memory_fraction must be between 0.05 and 1.0")
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
        properties = torch.cuda.get_device_properties(device)
        print(
            f"[resource] NVIDIA CUDA device={properties.name}; "
            f"capability={properties.major}.{properties.minor}; "
            f"memory={properties.total_memory / 1024**3:.1f} GB; "
            f"allocator_limit={gpu_memory_fraction:.0%}; "
            f"torch_threads={cpu_threads}"
        )
    elif device.type == "mps":
        try:
            torch.backends.mps.allow_fp16_reduced_precision_reduction = True
        except AttributeError:
            pass
        print(
            f"[resource] Apple Metal (MPS) on {platform.machine()}; "
            f"torch_threads={cpu_threads}; unsupported-op fallback=CPU"
        )
    else:
        print(f"[resource] CPU-only; torch_threads={cpu_threads}")
    return device


def seed_device(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if is_mps_available():
        torch.mps.manual_seed(seed)


def synchronize_device(device):
    """Wait for queued accelerator work before switching execution modes."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def autocast_context(device, enabled):
    """Use FP16 autocast on CUDA/MPS while keeping scalar losses in FP32."""
    if enabled and device.type in ("cuda", "mps"):
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


class _NoOpGradScaler:
    """Interface-compatible no-op scaler for CPU/MPS or disabled AMP."""

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        return optimizer.step()

    def update(self):
        return None


def make_grad_scaler(device, enabled):
    """Create a CUDA FP16 gradient scaler when required."""
    if not (enabled and device.type == "cuda"):
        return _NoOpGradScaler()
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def device_memory_gb(device):
    if device.type == "mps":
        return (
            torch.mps.current_allocated_memory() / 1024**3,
            torch.mps.driver_allocated_memory() / 1024**3,
        )
    if device.type == "cuda":
        return (
            torch.cuda.memory_allocated(device) / 1024**3,
            torch.cuda.memory_reserved(device) / 1024**3,
        )
    return (0.0, 0.0)


def shutdown_loader(loader):
    """Stop persistent DataLoader workers before moving to the next process."""
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()
    loader._iterator = None


def release_accelerator_cache(device):
    synchronize_device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            pass
