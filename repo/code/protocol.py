"""Shared helpers for patient-aware splits and reproducible runs."""

import random

import numpy as np


SEED_PROTOCOL = "seeded_v2"


def patient_id_from_key(key):
    """Normalize CrossPan sequence keys to a patient identifier.

    Examples:
      T1_train_EMC134_T1 -> EMC134
      OOP_test_EMC134_oop -> EMC134
      T1_train_NU_T1_08 -> NU_08
      OOP_test_NU_oop_08 -> NU_08
    """
    parts = str(key).split("_")
    if len(parts) < 3:
        raise ValueError(f"无法从病例 key 提取患者 ID: {key!r}")
    payload = [p for p in parts[2:] if p.lower() not in {"t1", "oop"}]
    if not payload:
        raise ValueError(f"病例 key 不含患者 ID: {key!r}")
    return "_".join(payload)


def select_nested_keys(pool, n_label, seed):
    """Return a deterministic, nested label subset for a fixed seed."""
    n_label = int(n_label)
    if n_label < 0 or n_label > len(pool):
        raise ValueError(f"标注病例数 {n_label} 超出 pool 大小 {len(pool)}")
    order = np.random.RandomState(int(seed)).permutation(len(pool))
    return [pool[i] for i in order[:n_label]]


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch CPU/CUDA RNGs for one model run."""
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

