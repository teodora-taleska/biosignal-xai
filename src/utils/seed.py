import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Seed all RNGs for reproducibility.

    Covers Python, NumPy, PyTorch (CPU + GPU). Also disables cuDNN's
    auto-tuner so the same algorithm is chosen on every run.

    Limitations
    -----------
    - num_workers > 0: DataLoader workers have their own RNG state and are
      NOT covered. On Windows num_workers must be 0 anyway (see config.yaml).
    - AMP / float16: rounding in mixed-precision ops can still vary slightly
      across runs due to non-associative float arithmetic at the hardware level.
    - Some CUDA kernels use atomic ops that are inherently non-deterministic
      even with deterministic=True; in practice the variance is negligible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
