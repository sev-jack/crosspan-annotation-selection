"""Small, auditable MedSAM fine-tuning runner for the CrossPan T1W -> OOP pilot.

This adapts the official MedSAM ViT-B checkpoint. It reads the project's existing
256x256 cached volumes and patient-disjoint split. Run from the ExperimentDG folder.
The evaluator uses GT-derived per-slice boxes, so reported metrics are oracle-box
prompted segmentation metrics, not fully automatic segmentation metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

HERE = Path(__file__).resolve().parents[1]


def patient_id_from_key(key: str) -> str:
    """Extract the patient ID from keys such as T1_train_EMC009_T1."""
    parts = str(key).split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot extract patient ID from {key!r}")
    payload = [part for part in parts[2:] if part.lower() not in {"t1", "oop"}]
    if not payload:
        raise ValueError(f"No patient ID in key {key!r}")
    return "_".join(payload)


def select_nested_keys(pool, n_label: int, seed: int):
    """Choose a deterministic nested subset, independent of model RNG."""
    order = np.random.RandomState(int(seed)).permutation(len(pool))
    return [pool[i] for i in order[:int(n_label)]]


def seed_everything(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@lru_cache(maxsize=4)
def load_case(cache_dir: str, key: str):
    """Load one project's cached volume. Arrays are [Z,H,W]."""
    path = Path(cache_dir) / f"{key}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache for {key}: {path}")
    with np.load(path) as data:
        image = np.asarray(data["img"], dtype=np.uint8)
        label = (np.asarray(data["lab"]) > 0).astype(np.uint8)
    if image.ndim != 3 or label.ndim != 3 or image.shape != label.shape:
        raise ValueError(f"Expected matching [Z,H,W] arrays for {key}; got {image.shape}, {label.shape}")
    return image, label


def load_sam(args):
    repo = Path(args.medsam_root).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"MedSAM repository not found: {repo}")
    sys.path.insert(0, str(repo))
    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:
        raise RuntimeError("Install the official MedSAM repository first (pip install -e .)") from exc

    model = sam_model_registry["vit_b"](checkpoint=str(Path(args.checkpoint).expanduser()))
    return model


def read_splits(path: Path, eval_key: str):
    splits = json.loads(path.read_text(encoding="utf-8"))
    if "source" not in splits or "targets" not in splits or "OOP" not in splits["targets"]:
        raise ValueError(f"Unexpected split file: {path}")
    oop = splits["targets"]["OOP"]
    if eval_key not in oop:
        raise KeyError(f"OOP split has no {eval_key!r}; available: {sorted(oop)}")
    source_train = list(splits["source"]["train"])
    pool = list(oop["pool"])
    eval_keys = list(oop[eval_key])
    if not source_train or not pool or not eval_keys:
        raise ValueError("Source train, OOP pool, and OOP evaluation split must all be non-empty")

    source_ids = {patient_id_from_key(k) for k in source_train}
    pool_ids = {patient_id_from_key(k) for k in pool}
    eval_ids = {patient_id_from_key(k) for k in eval_keys}
    if source_ids & eval_ids:
        raise ValueError(f"Patient overlap: source train vs OOP evaluation: {sorted(source_ids & eval_ids)[:5]}")
    if pool_ids & eval_ids:
        raise ValueError(f"Patient overlap: OOP adaptation pool vs evaluation: {sorted(pool_ids & eval_ids)[:5]}")
    if source_ids & pool_ids:
        raise ValueError(f"Patient overlap: source train vs OOP adaptation pool: {sorted(source_ids & pool_ids)[:5]}")
    return splits, source_train, pool, eval_keys


def image_tensor(image_u8: np.ndarray, device: torch.device):
    """Convert a cached grayscale slice to MedSAM's 3x1024x1024 input."""
    x = torch.from_numpy(np.ascontiguousarray(image_u8)).to(device=device, dtype=torch.float32)
    x = x[None, None] / 255.0
    x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False, antialias=True)
    return x.repeat(1, 3, 1, 1)


def box_from_mask(mask: np.ndarray, jitter_px: int, rng: np.random.Generator):
    """Return xyxy in 1024-input coordinates; empty slices receive a full-frame box."""
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return np.asarray([0.0, 0.0, 1024.0, 1024.0], dtype=np.float32)
    # Cached images are square (256x256); map pixel-edge coordinates to 1024 space.
    scale_x, scale_y = 1024.0 / w, 1024.0 / h
    x0, x1 = float(xs.min()) * scale_x, float(xs.max() + 1) * scale_x
    y0, y1 = float(ys.min()) * scale_y, float(ys.max() + 1) * scale_y
    if jitter_px:
        x0 -= int(rng.integers(0, jitter_px + 1))
        y0 -= int(rng.integers(0, jitter_px + 1))
        x1 += int(rng.integers(0, jitter_px + 1))
        y1 += int(rng.integers(0, jitter_px + 1))
    return np.asarray([
        np.clip(x0, 0, 1024), np.clip(y0, 0, 1024),
        np.clip(x1, 0, 1024), np.clip(y1, 0, 1024),
    ], dtype=np.float32)


class SliceDataset(Dataset):
    def __init__(self, keys, cache_dir: Path, jitter_px: int, seed: int):
        self.cache_dir = str(cache_dir)
        self.jitter_px = int(jitter_px)
        self.rng = np.random.default_rng(seed)
        self.samples = []
        for key in keys:
            image, _ = load_case(self.cache_dir, key)
            self.samples.extend((key, z) for z in range(image.shape[0]))
        if not self.samples:
            raise ValueError("No slices found for training")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        key, z = self.samples[index]
        image, label = load_case(self.cache_dir, key)
        mask = label[z]
        box = box_from_mask(mask, self.jitter_px, self.rng)
        target = torch.from_numpy(np.ascontiguousarray(mask[None])).float()
        return torch.from_numpy(image[z].copy()), target, torch.from_numpy(box), key


def freeze_prompt_encoder(model):
    for parameter in model.prompt_encoder.parameters():
        parameter.requires_grad_(False)


def medsam_logits(model, images, boxes):
    image_embeddings = model.image_encoder(images)
    with torch.no_grad():
        box_t = boxes.to(device=images.device, dtype=torch.float32)[:, None, :]
        sparse, dense = model.prompt_encoder(points=None, boxes=box_t, masks=None)
        image_pe = model.prompt_encoder.get_dense_pe()
    logits, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return logits


def segmentation_loss(logits, target):
    target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    dice = 1.0 - ((2.0 * (prob * target).sum(dims) + 1.0) /
                  (prob.sum(dims) + target.sum(dims) + 1.0)).mean()
    return bce + dice


def extract_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def train(model, dataset, args, device, init_path: Path | None):
    if init_path is not None:
        state = extract_state(init_path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}")
        print(f"Loaded initialization checkpoint: {init_path}", flush=True)

    freeze_prompt_encoder(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_samples = args.steps_per_epoch * args.batch_size

    for epoch in range(1, args.epochs + 1):
        sampler = RandomSampler(
            dataset, replacement=True, num_samples=total_samples,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            num_workers=0, pin_memory=(device.type == "cuda"))
        model.train()
        losses = []
        started = time.time()
        for image_u8, target, box, _ in loader:
            images = torch.cat([image_tensor(x, device) for x in image_u8.numpy()], dim=0)
            target = target.to(device, non_blocking=True)
            boxes = box.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda" and args.amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = medsam_logits(model, images, boxes)
                    loss = segmentation_loss(logits.float(), target)
            else:
                logits = medsam_logits(model, images, boxes)
                loss = segmentation_loss(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch}/{args.epochs} loss={np.mean(losses):.5f} "
              f"steps={len(losses)} seconds={time.time() - started:.1f}", flush=True)


@torch.no_grad()
def evaluate(model, eval_keys, cache_dir: Path, device, jitter_px: int, seed: int, save_dir: Path):
    rng = np.random.default_rng(seed)
    model.eval()
    rows = []
    pred_dir = save_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for key in eval_keys:
        image, label = load_case(str(cache_dir), key)
        prediction = np.zeros_like(label, dtype=np.uint8)
        for z in range(image.shape[0]):
            box = box_from_mask(label[z], jitter_px, rng)
            images = image_tensor(image[z], device)
            boxes = torch.from_numpy(box[None]).to(device)
            logits = medsam_logits(model, images, boxes)
            logits = F.interpolate(logits.float(), size=label[z].shape, mode="bilinear", align_corners=False)
            prediction[z] = (torch.sigmoid(logits[0, 0]).cpu().numpy() >= 0.5).astype(np.uint8)
        intersection = int(np.logical_and(prediction, label).sum())
        denom = int(prediction.sum()) + int(label.sum())
        dice = 1.0 if denom == 0 else 2.0 * intersection / denom
        rows.append({"case": key, "dice_3d": float(dice), "gt_voxels": int(label.sum()),
                     "pred_voxels": int(prediction.sum())})
        np.savez_compressed(pred_dir / f"{key}.npz", prediction=prediction)
        print(f"eval {key}: dice_3d={dice:.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("zero", "source", "adapt", "target-only", "eval"), required=True)
    ap.add_argument("--work", type=Path, default=None,
                    help="ExperimentDG root containing cache/ and splits_oop_pd.json; auto-detected from parent folders when omitted")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--medsam-root", required=True, help="Path to the official bowang-lab/MedSAM checkout")
    ap.add_argument("--checkpoint", required=True, help="Official medsam_vit_b.pth checkpoint")
    ap.add_argument("--init-checkpoint", default=None, help="Source checkpoint for adapt/eval modes")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-target", type=int, default=10, help="Number of OOP pool cases for adapt/target-only")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="Training/evaluation RNG seed")
    ap.add_argument("--eval-key", default="eval_patient_disjoint")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-epoch", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--box-jitter", type=int, default=20, help="Training box expansion in 1024-input pixels")
    ap.add_argument("--eval-box-jitter", type=int, default=0, help="Evaluation oracle-box expansion; 0 means exact")
    ap.add_argument("--limit-eval-cases", type=int, default=0,
                    help="Optional smoke-test limit; 0 evaluates the complete fixed split")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    if args.work is None:
        candidates = [HERE, *HERE.parents[:4]]
        args.work = next((p for p in candidates
                          if (p / "cache").is_dir() and (p / "splits_oop_pd.json").is_file()), HERE)
    args.work = args.work.expanduser().resolve()
    cache_dir = (args.cache or args.work / "cache").expanduser().resolve()
    split_path = (args.splits or args.work / "splits_oop_pd.json").expanduser().resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}; set --work or --cache to the uploaded ExperimentDG data")
    if not split_path.is_file():
        raise FileNotFoundError(f"Patient-disjoint split file not found: {split_path}; set --work or --splits explicitly")
    _, source_train, target_pool, eval_keys = read_splits(split_path, args.eval_key)
    if args.limit_eval_cases > 0:
        eval_keys = eval_keys[:args.limit_eval_cases]
    if args.n_target < 1 or args.n_target > len(target_pool):
        raise ValueError(f"--n-target must be in [1,{len(target_pool)}]")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; source_train={len(source_train)}; "
          f"OOP_pool={len(target_pool)}; OOP_eval={len(eval_keys)}", flush=True)

    out_dir = args.work / "runs" / "medsam" / args.run_id
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Run directory already exists and is non-empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_sam(args).to(device)
    train_keys = []
    init_path = None
    if args.mode == "source":
        train_keys = source_train
    elif args.mode == "adapt":
        if not args.init_checkpoint:
            raise ValueError("--mode adapt requires --init-checkpoint pointing to the source-only model")
        train_keys = select_nested_keys(target_pool, args.n_target, args.subset_seed)
        init_path = Path(args.init_checkpoint).expanduser().resolve()
    elif args.mode == "target-only":
        train_keys = select_nested_keys(target_pool, args.n_target, args.subset_seed)
    elif args.mode == "eval":
        if not args.init_checkpoint:
            raise ValueError("--mode eval requires --init-checkpoint")
        init_path = Path(args.init_checkpoint).expanduser().resolve()

    if args.mode in {"source", "adapt", "target-only"}:
        dataset = SliceDataset(train_keys, cache_dir, args.box_jitter, args.seed)
        print(f"training_cases={len(train_keys)}; training_slices={len(dataset)}; "
              f"steps={args.epochs * args.steps_per_epoch}", flush=True)
        train(model, dataset, args, device, init_path)
        torch.save(model.state_dict(), out_dir / "model.pth")

    if args.mode == "zero":
        freeze_prompt_encoder(model)
    if args.mode in {"eval", "zero"}:
        # eval mode loads a previously adapted checkpoint; zero mode keeps official weights.
        if args.mode == "eval":
            model.load_state_dict(extract_state(init_path), strict=True)

    per_case = evaluate(model, eval_keys, cache_dir, device,
                        args.eval_box_jitter, args.seed, out_dir)
    metadata = {
        "run_id": args.run_id,
        "mode": args.mode,
        "train_case_ids": train_keys,
        "n_target": len(train_keys) if args.mode in {"adapt", "target-only"} else None,
        "subset_seed": args.subset_seed,
        "train_seed": args.seed,
        "eval_key": args.eval_key,
        "eval_case_ids": eval_keys,
        "prompt_protocol": "per-slice GT-derived box; full-frame box on empty slices; oracle localization",
        "box_jitter_train_pixels_at_1024": args.box_jitter,
        "box_jitter_eval_pixels_at_1024": args.eval_box_jitter,
        "training_steps": args.epochs * args.steps_per_epoch if args.mode in {"source", "adapt", "target-only"} else 0,
        "official_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "initial_checkpoint": str(init_path) if init_path else None,
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "device": str(device),
        "mean_patient_dice_3d": float(np.mean([row["dice_3d"] for row in per_case])),
        "per_case": per_case,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"mean_patient_dice_3d={metadata['mean_patient_dice_3d']:.4f}", flush=True)
    print(f"outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
