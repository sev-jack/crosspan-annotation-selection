"""填表的基本单元：给定 (源域, 目标域, 预算, 种子)，跑一次，报一个目标域 Dice。

协议（对应你文档里的五条纪律）:
  - 固定 epoch 预算，不做早停、不用目标域调参
  - 源模型每个种子只训一次，所有预算档位共用（省算力，也更公平）
  - 预算 = 目标域 pool 里可以标注的病例比例；0% 就是零样本
  - 评估一律用 3D 体 Dice，在中标目标域的 eval 集上

单跑:
    python train_eval.py --source T1W --target OOP --budget 0.1 --seed 0

结果同时打印并返回，run_grid.py 负责把整张表跑完。
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from monai.losses import DiceCELoss
from monai.networks.nets import UNet

torch.backends.cudnn.benchmark = False

from config import (
    ALL_SLICES,
    N_SLICES,
    RUN_TAG,
    CACHE,
    RUNS,
    SPLITS,
    BATCH,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    MODEL_CHANNELS,
    MODEL_STRIDES,
    NUM_WORKERS,
    USE_AMP,
    AMP_DTYPE,
    USE_TTA,
    TTA_SCALE,
    SOURCE_EPOCHS,
    SOURCE_LR,
    SIZE,
    TARGET_SPACING,
    BODY_CROP,
)
from protocol import (SEED_PROTOCOL, patient_id_from_key, seed_everything,
                      select_nested_keys)

_CACHE_INDEX = None


# ---------------------------------------------------------------- 数据

def load_case(key):
    d = np.load(CACHE / f"{key}.npz")
    return d["img"], d["lab"]      # uint8 [Z,256,256]


def norm(x_u8):
    """uint8 -> float32。预处理已按每例 0.5/99.5 分位裁剪，这里只做固定仿射。"""
    return (x_u8.astype(np.float32) / 255.0 - 0.5) / 0.25


def volume_to_25d(img):
    """把 3D 体数据变成 2.5D 通道堆叠：每个切片带上前后各一张邻居。

    为什么必须这么做：2D 逐切片模型看不到胰腺的立体形状，
    而"胰腺在哪、往哪走"主要靠相邻切片的关系判断，这是胰腺分割的
    主要线索。实测纯 2D 的域内验证 Dice 卡在 0.42 上不去。
    返回 (Z, 3, H, W)，uint8。
    """
    r = N_SLICES // 2
    pad = np.pad(img, ((r, r), (0, 0), (0, 0)), mode="edge")
    z = img.shape[0]
    return np.stack([pad[i:i + z] for i in range(N_SLICES)], axis=1)


def _augment(x, y):
    """x 是 (3,256,256) 的 0-1 float32，y 是 (256,256) 的 0/1 标签。

    只有翻转和旋转会让模型过拟合到训练病例（实测训练 0.79 / 验证 0.41），
    所以补上强度扰动、Gamma 和高斯噪声——这三样对 MRI 跨中心的泛化最有效。
    """
    # 几何：翻转与 90 度旋转
    if np.random.rand() < 0.5:
        x, y = x[:, ::-1, :], y[:, ::-1]
    if np.random.rand() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    k = np.random.randint(4)
    if k:
        x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(0, 1))

    # 强度平移与缩放：模拟不同扫描仪的对比度差异
    if np.random.rand() < 0.8:
        x = x * np.random.uniform(0.80, 1.20) + np.random.uniform(-0.10, 0.10)
    # Gamma：模拟对比度曲线的非线性差异
    if np.random.rand() < 0.4:
        x = np.clip(x, 0.0, 1.0) ** np.random.uniform(0.7, 1.4)
    # 噪声：模拟不同信噪比
    if np.random.rand() < 0.3:
        x = x + np.random.normal(0.0, 0.02, size=x.shape).astype(np.float32)

    x = np.clip(x, 0.0, 1.0)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def build_slices(keys, cap_per_case=None, n_neg=8, seed=0):
    """收集训练用切片。

    关键：每例除了含前景的切片，还必须掺入若干"空切片"。
    胰腺只出现在约 1/3 的轴位切片上；只喂前景切片会让模型学会
    "每张都有胰腺"，评估整卷时到处乱报，体 Dice 被假阳性压垮。
    空切片从整卷随机抽，而不是只取邻域，让模型见到全卷的空背景。
    """
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    for k in keys:
        img, lab = load_case(k)
        vol = volume_to_25d(img)
        flat = lab.reshape(lab.shape[0], -1).max(1)
        pos = np.where(flat > 0)[0]
        if len(pos) == 0:
            print(f"  !! {k} 没有任何前景切片，跳过")
            continue
        if ALL_SLICES:
            # 全层训练：模型必须见过整卷的空背景，才不会在远处乱报
            zs = np.arange(lab.shape[0])
        else:
            neg = np.where(flat == 0)[0]
            if len(neg) > n_neg:
                neg = rng.choice(neg, size=n_neg, replace=False)
            zs = np.sort(np.concatenate([pos, neg]))
        if cap_per_case and len(zs) > cap_per_case:
            sel = np.linspace(0, len(zs) - 1, cap_per_case).round().astype(int)
            zs = zs[sel]
        xs.append(vol[zs])
        ys.append(lab[zs])
    if not xs:
        raise SystemExit("没有可用切片")
    return np.concatenate(xs), np.concatenate(ys)


class SliceDS(Dataset):
    def __init__(self, X, Y, aug=False):
        self.X, self.Y, self.aug = X, Y, aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].astype(np.float32) / 255.0     # (3,H,W)
        y = self.Y[i]
        if self.aug:
            x, y = _augment(x, y)
        xt = torch.from_numpy((x - 0.5) / 0.25)
        yt = torch.from_numpy(y.astype(np.int64))
        return xt, yt


# ---------------------------------------------------------------- 模型与指标

def model_cfg():
    """架构指纹：结构改了旧权重就作废，避免 shape mismatch 报错。"""
    return dict(in_channels=N_SLICES, channels=list(MODEL_CHANNELS), strides=list(MODEL_STRIDES),
                amp=USE_AMP, amp_dtype=AMP_DTYPE, batch=BATCH, aug="affine_v2",
                seed_protocol=SEED_PROTOCOL, all_slices=ALL_SLICES,
                source_epochs=SOURCE_EPOCHS, source_lr=SOURCE_LR,
                cache=str(CACHE.resolve()), size=SIZE,
                target_spacing=TARGET_SPACING, body_crop=BODY_CROP,
                use_tta=USE_TTA, tta_scale=TTA_SCALE)


def split_fingerprint(train_keys, val_keys):
    payload = json.dumps(
        {"train": sorted(train_keys), "val": sorted(val_keys)},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def amp_ctx(dev):
    """混合精度上下文。CPU 上自动退化为 no-op。"""
    if not USE_AMP or dev != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if AMP_DTYPE == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_model():
    return UNet(
        spatial_dims=2,
        in_channels=N_SLICES,
        out_channels=2,
        channels=MODEL_CHANNELS,
        strides=MODEL_STRIDES,
        num_res_units=2,
    )


def dice3d(pred, gt):
    p = pred.reshape(-1) > 0
    g = gt.reshape(-1) > 0
    denom = int(p.sum()) + int(g.sum())
    if denom == 0:
        return 1.0
    return 2.0 * int(np.logical_and(p, g).sum()) / denom


def _spacing_zyx_for(key):
    global _CACHE_INDEX
    index_path = CACHE / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"缺少缓存索引 {index_path}，先运行 prepare.py / prepare_ct.py")
    if _CACHE_INDEX is None:
        _CACHE_INDEX = json.loads(index_path.read_text(encoding="utf-8"))
    info = _CACHE_INDEX.get(key, {})
    spacing = info.get("spacing_zyx")
    if not spacing or len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError(
            f"病例 {key} 缺少有效 spacing_zyx；先运行 prepare.py 和 prepare_ct.py 更新物理间距元数据"
        )
    return tuple(float(x) for x in spacing)


def surface_distances_mm(pred, gt, spacing_zyx):
    """Return HD95 and symmetric ASSD in mm; undefined for a one-sided empty mask."""
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if not pred.any() and not gt.any():
        return {"hd95_mm": 0.0, "assd_mm": 0.0}
    if not pred.any() or not gt.any():
        return {"hd95_mm": None, "assd_mm": None}

    structure = generate_binary_structure(3, 1)
    surf_pred = pred & ~binary_erosion(pred, structure=structure, border_value=0)
    surf_gt = gt & ~binary_erosion(gt, structure=structure, border_value=0)
    dt_to_gt = distance_transform_edt(~surf_gt, sampling=spacing_zyx)
    dt_to_pred = distance_transform_edt(~surf_pred, sampling=spacing_zyx)
    pred_to_gt = dt_to_gt[surf_pred]
    gt_to_pred = dt_to_pred[surf_gt]
    all_distances = np.concatenate([pred_to_gt, gt_to_pred])
    return {
        "hd95_mm": float(np.percentile(all_distances, 95)),
        "assd_mm": float((pred_to_gt.mean() + gt_to_pred.mean()) / 2.0),
    }


@torch.no_grad()
def forward_tta(model, xt, dev):
    """测试时增强：原图 + 三种翻转 + 一个缩放，预测取平均。

    做法是把输入变换后跑模型，再把输出**反变换回去**，最后平均。
    评估成本 ×5，但不用重训。
    """
    logits = []
    with amp_ctx(dev):
        logits.append(model(xt))

        xf = torch.flip(xt, [3])
        logits.append(torch.flip(model(xf), [3]))

        xv = torch.flip(xt, [2])
        logits.append(torch.flip(model(xv), [2]))

        xb = torch.flip(xt, [2, 3])
        logits.append(torch.flip(model(xb), [2, 3]))

        if TTA_SCALE:
            h, w = xt.shape[-2:]
            # 缩小后必须仍是 16 的整数倍（模型 4 次下采样），否则跳跃连接尺寸对不上
            sh = max(16, (int(h * 0.75) // 16) * 16)
            sw = max(16, (int(w * 0.75) // 16) * 16)
            xs = torch.nn.functional.interpolate(
                xt, size=(sh, sw), mode="bilinear", align_corners=False)
            ls = model(xs)
            logits.append(torch.nn.functional.interpolate(
                ls, size=(h, w), mode="bilinear", align_corners=False))
    return torch.stack(logits, 0).mean(0)


@torch.no_grad()
def forward_plain(model, xt, dev):
    with amp_ctx(dev):
        return model(xt)


@torch.no_grad()
def eval_volumes(model, keys, dev, batch=16, tta=None, forward=None,
                 include_surface_metrics=False):
    """评估整卷 3D Dice。forward 可传入自定义前向（用于多模型集成）。"""
    model.eval()
    use_tta = USE_TTA if tta is None else tta
    fwd = forward or (forward_tta if use_tta else forward_plain)
    per_case = {}
    per_case_metrics = {}
    for k in keys:
        img, lab = load_case(k)
        vol = volume_to_25d(img)
        Z = img.shape[0]
        pred = np.zeros_like(lab)
        for z0 in range(0, Z, batch):
            z1 = min(Z, z0 + batch)
            xt = torch.from_numpy(norm(vol[z0:z1])).to(dev)
            logits = fwd(model, xt, dev)
            pred[z0:z1] = logits.argmax(1).cpu().numpy().astype(np.uint8)
        per_case[k] = dice3d(pred, lab)
        if include_surface_metrics:
            per_case_metrics[k] = surface_distances_mm(pred, lab, _spacing_zyx_for(k))
    mean_dice = float(np.mean(list(per_case.values())))
    if include_surface_metrics:
        return mean_dice, per_case, per_case_metrics
    return mean_dice, per_case


# ---------------------------------------------------------------- 训练

def _augment_gpu(x, y):
    """整批在 GPU 上做增广。x:(B,3,H,W) y:(B,H,W)，都已是目标 device。

    放在 GPU 上是为了消掉 CPU 取数瓶颈：实测 num_workers=0 + numpy 增广时，
    GPU 大部分时间在等数据（单 epoch 39 秒，其中一多半是等待）。
    """
    dev = x.device
    b = x.shape[0]

    fh = torch.rand(b, device=dev) < 0.5
    fw = torch.rand(b, device=dev) < 0.5
    x = torch.where(fh[:, None, None, None], torch.flip(x, [2]), x)
    y = torch.where(fh[:, None, None], torch.flip(y, [1]), y)
    x = torch.where(fw[:, None, None, None], torch.flip(x, [3]), x)
    y = torch.where(fw[:, None, None], torch.flip(y, [2]), y)

    k = int(torch.randint(0, 4, (1,), device=dev).item())
    if k:
        x = torch.rot90(x, k, [2, 3]).contiguous()
        y = torch.rot90(y, k, [1, 2]).contiguous()

    # 随机仿射：缩放 + 小角度旋转 + 平移。
    # 这是之前最关键的缺失。胰腺在不同病人身上的大小和姿态差异很大，
    # 没有这一项，模型只能背训练病例（实测训练 0.95 / 验证 0.39）。
    b2 = x.shape[0]
    ang = (torch.rand(b2, device=dev) * 2 - 1) * (25.0 * math.pi / 180.0)
    sc = torch.empty(b2, device=dev).uniform_(0.75, 1.35)
    tx = (torch.rand(b2, device=dev) * 2 - 1) * 0.12
    ty = (torch.rand(b2, device=dev) * 2 - 1) * 0.12
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(b2, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, mode="bilinear",
                                        padding_mode="border", align_corners=False)
    y = torch.nn.functional.grid_sample(
        y.unsqueeze(1).float(), grid, mode="nearest",
        padding_mode="zeros", align_corners=False).squeeze(1).long()

    # 低分辨率模拟：模拟不同层厚／矩阵，让模型不依赖锐利细节
    if torch.rand(1, device=dev).item() < 0.25:
        h, w = x.shape[-2:]
        small = torch.nn.functional.interpolate(x, size=(h // 2, w // 2), mode="bilinear",
                                                align_corners=False)
        x = torch.nn.functional.interpolate(small, size=(h, w), mode="bilinear",
                                            align_corners=False)

    s = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.80, 1.20)
    o = torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.10, 0.10)
    x = x * s + o

    g = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.7, 1.4)
    use_g = (torch.rand(b, 1, 1, 1, device=dev) < 0.4)
    x = torch.where(use_g, x.clamp(0, 1).pow(g), x)

    use_n = (torch.rand(b, 1, 1, 1, device=dev) < 0.3).to(x.dtype)
    x = x + use_n * 0.02 * torch.randn_like(x)
    return x.clamp_(0, 1), y


def train(keys, epochs, lr, dev, init_ckpt=None, tag="run", augment=True,
          val_keys=None, eval_every=20, seed=0):
    seed_everything(seed)
    X, Y = build_slices(keys, seed=seed)
    print(f"  {tag}: {len(keys)} 例 -> {len(X)} 张含前景切片，前景像素 {Y.mean() * 100:.2f}%",
          flush=True)

    # 整份训练数据一次性搬进显存（uint8，约 1 GB / 5000 片）。
    # 这是"用显存换时间"的正解：GPU 不再等 CPU 取数。
    N = len(X)
    # 数据留 CPU 内存，按批上卡。全层训练样本量翻三倍，整份塞显存会爆。
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    print(f"  数据就绪: X {tuple(Xt.shape)}  Y {tuple(Yt.shape)} (常驻内存，按批上卡)", flush=True)

    model = build_model()
    if init_ckpt:
        model.load_state_dict(torch.load(init_ckpt, map_location="cpu"))
        print(f"  从 {init_ckpt} 初始化")
    model = model.to(dev)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(2, epochs), eta_min=lr * 0.02)

    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        t_ep_start = time.time()
        tot, nb = 0.0, 0
        perm = torch.randperm(N)          # 索引要与数据同在 CPU，按批再上卡
        for b0 in range(0, max(1, N - 1), BATCH):
            bi = perm[b0:b0 + BATCH]
            if len(bi) < 2:
                continue
            # 顺序不能错：先在 [0,1] 空间做增广，最后才归一化。
            # 反过来的话 _augment_gpu 里的 clamp(0,1) 会把归一化后的负数全砍掉，
            # 训练输入和评估输入分布对不上，模型直接废掉（曾把 val Dice 打到 0.0004）。
            xi = Xt[bi].to(dev, non_blocking=True).float().div_(255.0)
            yi = Yt[bi].to(dev, non_blocking=True).long()
            if augment:
                xi, yi = _augment_gpu(xi, yi)
            xi = xi.sub_(0.5).div_(0.25)
            opt.zero_grad(set_to_none=True)
            with amp_ctx(dev):
                loss = loss_fn(model(xi), yi.unsqueeze(1))
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        # 每 10 个 epoch 打一次，并强制 flush——否则管道会攒着不吐，看着像卡死
        if ep % max(1, epochs // 30) == 0 or ep == epochs:
            el = time.time() - t_ep_start
            print(f"    epoch {ep:3d}/{epochs}  loss={tot / max(nb, 1):.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  {el:.0f}s/epoch", flush=True)
        sched.step()

        # 最佳验证权重回滚：300 epoch 里模型会在后期从"会了"滑向"背了"，
        # 保留验证最好的那一版，而不是最后一版。只在源域用（目标域不能用来选模型）。
        if val_keys and (ep % eval_every == 0 or ep == epochs):
            d, _ = eval_volumes(model, val_keys, dev)
            if d > best:
                best = d
                best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
            print(f"    [val] epoch {ep:3d}  Dice={d:.4f}  (best {best:.4f})", flush=True)
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  已回滚到最佳 val 权重: Dice={best:.4f}", flush=True)
    return model


def get_source_model(source, seed, splits, dev):
    RUNS.mkdir(parents=True, exist_ok=True)
    ckpt = RUNS / f"source_{source}_s{seed}{RUN_TAG}.pt"
    meta = RUNS / f"source_{source}_s{seed}{RUN_TAG}.json"
    tr = splits["source"]["train"]
    va = splits["source"]["val"]

    if ckpt.exists():
        info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        expected_split = split_fingerprint(tr, va)
        if (info.get("model_cfg") == model_cfg()
                and info.get("seed_protocol") == SEED_PROTOCOL
                and info.get("source_split_fingerprint") == expected_split):
            print(f"源模型已存在: {ckpt}")
            return ckpt, info.get("source_val_dice")
        raise RuntimeError(
            f"已有源模型与当前代码/划分不匹配，已保留原文件且不会覆盖：{ckpt}\n"
            "请设置新的 EXP_RUNS 与 EXP_TAG，为新协议建立独立输出目录。"
        )

    print(f"训练源模型 {source} seed={seed} (train={len(tr)} val={len(va)})")
    t0 = time.time()
    model = train(tr, SOURCE_EPOCHS, SOURCE_LR, dev, tag="source", val_keys=va, seed=seed)
    src_dice, _ = eval_volumes(model, va, dev)
    torch.save(model.state_dict(), ckpt)
    meta.write_text(
        json.dumps(dict(source=source, seed=seed, epochs=SOURCE_EPOCHS,
                        model_cfg=model_cfg(),
                        seed_protocol=SEED_PROTOCOL,
                        source_split_fingerprint=split_fingerprint(tr, va),
                        source_val_dice=round(float(src_dice), 6)),
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  源域 val 体 Dice = {src_dice:.4f}   用时 {time.time() - t0:.0f}s  -> {ckpt}")
    return ckpt, src_dice


# ---------------------------------------------------------------- 单次运行

def run_one(source, target, budget, seed, splits=None, dev=None, verbose=True,
            eval_key="eval", n_label_override=None, checkpoint_path=None,
            include_surface_metrics=False):
    splits = splits or json.loads(SPLITS.read_text(encoding="utf-8"))
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")

    target_split = splits["targets"][target]
    pool = target_split["pool"]
    if eval_key not in target_split:
        raise KeyError(f"目标域 {target} 没有评估划分 {eval_key!r}")
    ev = target_split[eval_key]
    if not ev:
        raise ValueError(f"目标域 {target} 的评估划分 {eval_key!r} 为空")
    pool_patients = {patient_id_from_key(k) for k in pool}
    eval_patients = {patient_id_from_key(k) for k in ev}
    overlap = pool_patients & eval_patients
    if overlap:
        raise ValueError(
            f"目标域 {target} 的 pool/eval 存在患者重叠，拒绝运行：{sorted(overlap)[:8]}"
        )
    if eval_key == "eval_patient_disjoint":
        source_patients = {
            patient_id_from_key(k)
            for k in splits["source"]["train"] + splits["source"]["val"]
        }
        leaked = eval_patients & source_patients
        if leaked:
            raise ValueError(
                f"patient-disjoint eval 与 source train/val 重叠：{sorted(leaked)[:8]}"
            )
        # 标注池也必须是患者独立的：否则"跨序列标注收益"里会混进"模型已经见过
        # 该病人另一序列"的成分，patient-disjoint 的口径就不成立。
        pool_leaked = pool_patients & source_patients
        # 例外：显式设置 EXP_ALLOW_POOL_OVERLAP=1 时放行，仅用于"含重叠池 vs
        # 患者独立池"的敏感性对照实验，并在日志里留痕。
        _allow_overlap = os.environ.get("EXP_ALLOW_POOL_OVERLAP", "").strip() not in ("", "0", "false", "False")
        if pool_leaked and _allow_overlap:
            print(f"  [警告] 已放行患者重叠标注池（EXP_ALLOW_POOL_OVERLAP=1）："
                  f"{sorted(pool_leaked)[:8]}（共 {len(pool_leaked)} 位）"
                  "—— 结果只能作为敏感性对照，不能作为 patient-disjoint 结论。")
        elif pool_leaked:
            raise ValueError(
                "标注池与 source train/val 存在患者重叠，patient-disjoint 口径不成立："
                f"{sorted(pool_leaked)[:8]}（共 {len(pool_leaked)} 位）\n"
                "请改用患者独立标注池，例如 EXP_SPLITS=splits_oop_pd.json；"
                "若确实要跑含重叠池的对照，请显式设置 EXP_ALLOW_POOL_OVERLAP=1。"
            )
    if n_label_override is None:
        n_label = int(round(budget * len(pool)))
        if budget > 0:
            n_label = max(1, n_label)  # 非零百分比预算至少给 1 例
        n_label = min(n_label, len(pool))
    else:
        n_label = int(n_label_override)
        if not 0 <= n_label <= len(pool):
            raise ValueError(f"标注病例数 {n_label} 超出 {target} pool 大小 {len(pool)}")
        budget = n_label / len(pool)

    if verbose:
        print(f"\n=== {source} -> {target} | eval={eval_key} | budget={budget:.2%} "
              f"({n_label}/{len(pool)} 例) | seed={seed} ===")

    ckpt, src_dice = get_source_model(source, seed, splits, dev)

    label_keys = []
    if n_label == 0:
        model = build_model()
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(dev)
    else:
        label_keys = select_nested_keys(pool, n_label, seed)
        if verbose:
            shown = label_keys[:12]
            suffix = " …" if len(label_keys) > len(shown) else ""
            print(f"  预算内病例: {shown}{suffix}")
        model = train(label_keys, FINETUNE_EPOCHS, FINETUNE_LR, dev,
                      init_ckpt=ckpt, tag=f"finetune {budget:.2%}", seed=seed)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    t0 = time.time()
    if include_surface_metrics:
        dice, per_case, per_case_metrics = eval_volumes(
            model, ev, dev, include_surface_metrics=True
        )
    else:
        dice, per_case = eval_volumes(model, ev, dev)
        per_case_metrics = None
    if verbose:
        print(f"  {target} eval ({len(ev)} 例) 体 Dice = {dice:.4f}   评估用时 {time.time() - t0:.0f}s")

    return dict(
        source=source,
        target=target,
        eval_key=eval_key,
        budget=budget,
        n_label=n_label,
        pool_size=len(pool),
        seed=seed,
        dice=round(dice, 6),
        source_val_dice=None if src_dice is None else round(float(src_dice), 6),
        n_eval=len(ev),
        label_keys=label_keys,
        seed_protocol=SEED_PROTOCOL,
        per_case={k: round(v, 6) for k, v in per_case.items()},
        per_case_metrics=per_case_metrics,
    )


def main():
    import config

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="T1W")
    ap.add_argument("--target", default="OOP")
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    config.start_log(config.RUNS / "train.log")
    print(json.dumps(run_one(args.source, args.target, args.budget, args.seed), indent=2))


if __name__ == "__main__":
    main()
