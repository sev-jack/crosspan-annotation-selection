"""上游任务第 1 阶段：给每一例目标域病例算「无标签可算」的域移严重度分数。

设计（对应 07 号方案第 3 节）：
  D1 分布型分数（不需要标签）：s_knn5 / s_mmd_sim，基于源模型瓶颈特征
  D2 预测型分数（不需要标签）：s_ens（3 种子+TTA 分歧）、s_entropy、s_energy
  D3 性能型真值（需要标签，只作验证）：逐例零样本 Dice

覆盖范围是「标注池 + 评估集」的全部病例，因为池里的病例也有金标准，
只是模型没在上面训练过——这正是可以当「未标注病例」用的原因。

不需要训练，只需要源模型前向。

    python code/upstream_scores.py                  # 全部三个域
    python code/upstream_scores.py --domains OOP    # 只跑一个域
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import config
import train_eval as TE
from protocol import patient_id_from_key

FEATURE_MODULE = "model.1.submodule.1.submodule.1.submodule.1.submodule"
OUT = config.WORK / "outputs" / "upstream"
SEEDS = [0, 1, 2]
BATCH = 16
KNN_K = 5


def load_split_files():
    """OOP 用患者独立池，CT/NYU 用主划分；评估集与主分析一致。"""
    main = json.loads((config.WORK / "splits.json").read_text(encoding="utf-8"))
    pd_file = config.WORK / "splits_oop_pd.json"
    pd = json.loads(pd_file.read_text(encoding="utf-8")) if pd_file.exists() else main
    return {
        "OOP": dict(pool=pd["targets"]["OOP"]["pool"],
                    eval=pd["targets"]["OOP"]["eval_patient_disjoint"]),
        "CT": dict(pool=main["targets"]["CT"]["pool"],
                   eval=main["targets"]["CT"]["eval"]),
        "NYU": dict(pool=main["targets"]["NYU"]["pool"],
                    eval=main["targets"]["NYU"]["eval"]),
        "source": main["source"]["train"] + main["source"]["val"],
    }


@torch.no_grad()
def embed(model, keys, dev, batch=12):
    """512 维瓶颈特征，每例取 5 张等距切片做平均（与 report_feature_space 同口径）。"""
    module = dict(model.named_modules())[FEATURE_MODULE]
    buf = []

    def capture(_m, _i, output):
        buf.append(output.detach().mean(dim=(-2, -1)).float().cpu().numpy())

    handle = module.register_forward_hook(capture)
    result = {}
    try:
        for key in keys:
            image, _ = TE.load_case(key)
            volume = TE.volume_to_25d(image)
            zs = np.unique(np.linspace(0, len(image) - 1, 5).round().astype(int))
            for i in range(0, len(zs), batch):
                xt = torch.from_numpy(TE.norm(volume[zs[i:i + batch]])).to(dev)
                with TE.amp_ctx(dev):
                    model(xt)
            result[key] = np.concatenate(buf, axis=0).mean(axis=0)
            buf.clear()
    finally:
        handle.remove()
    return result


def kernel(a, b, gamma):
    d2 = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)
    return np.exp(-gamma * d2)


def mmd2(x, y, gamma):
    return float(kernel(x, x, gamma).mean() + kernel(y, y, gamma).mean()
                 - 2 * kernel(x, y, gamma).mean())


def median_gamma(pool):
    d = np.sum(pool * pool, axis=1)[:, None] + np.sum(pool * pool, axis=1)[None, :] - 2 * pool @ pool.T
    nz = d[np.triu_indices(len(pool), 1)]
    nz = nz[nz > 0]
    return 1.0 / float(np.median(nz)) if nz.size else 1.0


def rankdata(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # 处理并列
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    for idx, c in enumerate(counts):
        if c > 1:
            ranks[inv == idx] = ranks[inv == idx].mean()
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def score_case(key, models, dev):
    image, lab = TE.load_case(key)
    volume = TE.volume_to_25d(image)
    z_total = image.shape[0]

    sum_fg = None
    sumsq_fg = None
    ent_sum = ent_n = 0.0
    en_sum = en_n = 0.0
    dices = []

    for seed in SEEDS:
        model = models[seed]
        pred = np.zeros_like(lab)
        for z0 in range(0, z_total, BATCH):
            z1 = min(z_total, z0 + BATCH)
            xt = torch.from_numpy(TE.norm(volume[z0:z1])).to(dev)
            logits = TE.forward_tta(model, xt, dev)
            prob = torch.softmax(logits, dim=1)
            fg = prob[:, 1].float().cpu().numpy()
            if sum_fg is None:
                sum_fg = np.zeros((z_total, fg.shape[1], fg.shape[2]), dtype=np.float64)
                sumsq_fg = np.zeros_like(sum_fg)
            sum_fg[z0:z1] += fg
            sumsq_fg[z0:z1] += fg * fg
            if seed == 0:
                ent = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1)
                energy = -torch.logsumexp(logits, dim=1)
                mask = torch.from_numpy(fg > 0.5).to(ent.device)
                if bool(mask.any()):
                    ent_sum += float(ent[mask].sum().item())
                    ent_n += float(mask.sum().item())
                    en_sum += float(energy[mask].sum().item())
                    en_n += float(mask.sum().item())
            pred[z0:z1] = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        dices.append(TE.dice3d(pred, lab))

    n = len(SEEDS)
    mean_fg = sum_fg / n
    var_fg = np.maximum(sumsq_fg / n - mean_fg * mean_fg, 0.0)
    # 只在"模型认为可能是前景"的区域统计分歧。若在全体素上取平均，
    # 分数会被器官体积占比污染（背景几乎无分歧），从而变成"器官大小"的代理。
    roi = mean_fg > 0.05
    feature = embed(models[SEEDS[0]], [key], dev)[key]

    return dict(
        n_slices=int(z_total),
        fg_fraction=float(roi.mean()),
        dice_s0=round(dices[0], 6), dice_s1=round(dices[1], 6), dice_s2=round(dices[2], 6),
        dice_mean=round(float(np.mean(dices)), 6),
        dice_sd=round(float(np.std(dices, ddof=1)), 6),
        s_ens=float(var_fg[roi].mean()) if bool(roi.any()) else 0.0,
        s_entropy=float(ent_sum / ent_n) if ent_n else float("nan"),
        s_energy=float(en_sum / en_n) if en_n else float("nan"),
    ), feature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="OOP,CT,NYU")
    ap.add_argument("--max-cases", type=int, default=0, help="调试用，每域最多多少例")
    args = ap.parse_args()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    OUT.mkdir(parents=True, exist_ok=True)
    splits = load_split_files()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}", flush=True)

    models = {}
    for seed in SEEDS:
        ckpt = config.WORK / "runs_seeded_v2" / f"source_T1W_s{seed}_seeded_v2.pt"
        model = TE.build_model()
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        models[seed] = model.to(dev).eval()
        print(f"载入源模型 seed {seed} <- {ckpt.name}", flush=True)

    print(f"\n提取源域特征库（{len(splits['source'])} 例）…", flush=True)
    t0 = time.time()
    source_feat = embed(models[0], splits["source"], dev)
    source_keys = list(source_feat)
    print(f"  完成，用时 {time.time() - t0:.0f}s", flush=True)

    rows = []
    features = {}
    for domain in domains:
        cases = [(k, "pool") for k in splits[domain]["pool"] if k not in splits[domain]["eval"]]
        cases += [(k, "eval") for k in splits[domain]["eval"]]
        if args.max_cases:
            cases = cases[:args.max_cases]
        print(f"\n=== {domain}: {len(cases)} 例 ===", flush=True)
        t0 = time.time()
        for i, (key, split) in enumerate(cases, 1):
            stats, feature = score_case(key, models, dev)
            features[key] = feature
            row = dict(case_key=key, domain=domain, split=split,
                       patient_id=patient_id_from_key(key), **stats)
            rows.append(row)
            if i % 20 == 0 or i == len(cases):
                el = time.time() - t0
                print(f"  [{i}/{len(cases)}] 用时 {el:.0f}s  "
                      f"零样本 Dice {stats['dice_mean']:.4f}  ens {stats['s_ens']:.4f}", flush=True)
        print(f"  {domain} 完成，用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)

    # ---- 特征标准化（只用源域统计量，保证分数是"无标签可算"的）----
    src_matrix = np.asarray([source_feat[k] for k in source_keys], dtype=float)
    mu = src_matrix.mean(axis=0)
    sd = src_matrix.std(axis=0)
    sd[sd < 1e-8] = 1.0
    src_z = (src_matrix - mu) / sd
    gamma = median_gamma(src_z)
    tgt_z = {k: (v - mu) / sd for k, v in features.items()}

    for row in rows:
        z = tgt_z[row["case_key"]]
        d = np.linalg.norm(src_z - z, axis=1)
        kth = np.sort(d)[:KNN_K]
        row["s_knn5"] = float(kth.mean())
        row["s_mmd_sim"] = float(1.0 - kernel(z[None, :], src_z, gamma).mean())
        row["s_knn5_rank"] = 0.0

    # ---- 域级 MMD² 复核：对得上已有 0.078/0.162/0.209 才说明特征管线一致 ----
    check = []
    for domain in domains:
        ev = [r["case_key"] for r in rows if r["domain"] == domain and r["split"] == "eval"]
        if not ev:
            continue
        z = np.asarray([tgt_z[k] for k in ev], dtype=float)
        check.append(dict(domain=domain, n=len(ev), mmd2=mmd2(src_z, z, gamma)))

    fields = ["case_key", "domain", "split", "patient_id", "n_slices", "fg_fraction",
              "dice_s0", "dice_s1", "dice_s2", "dice_mean", "dice_sd",
              "s_ens", "s_entropy", "s_energy", "s_knn5", "s_mmd_sim"]
    with (OUT / "severity_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})

    with (OUT / "domain_mmd_check.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "n", "mmd2"])
        w.writeheader()
        w.writerows(check)

    # ---- 相关性（H1）----
    score_names = ["s_ens", "s_entropy", "s_energy", "s_knn5", "s_mmd_sim"]
    lines = ["# 严重度分数与真值的相关性（H1）", "",
             "真值 = 逐例零样本 Dice（3 种子均值，越低越严重）。分数越高应越严重，",
             "所以预期与 Dice 呈负相关。", "",
             "| 分数 | 全体 ρ | OOP | CT | NYU |", "|---|---:|---:|---:|---:|"]
    print("\n=== Spearman 相关（分数 vs 零样本 Dice）===", flush=True)
    for s in score_names:
        all_r = spearman([r[s] for r in rows], [r["dice_mean"] for r in rows])
        parts = []
        for domain in domains:
            sub = [r for r in rows if r["domain"] == domain]
            parts.append(spearman([r[s] for r in sub], [r["dice_mean"] for r in sub]))
        print(f"  {s:10s} 全体 {all_r:+.3f} | " + " ".join(f"{d} {p:+.3f}" for d, p in zip(domains, parts)), flush=True)
        lines.append(f"| {s} | {all_r:+.3f} | " + " | ".join(f"{p:+.3f}" for p in parts) + " |")
    (OUT / "severity_correlation.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print("\n域级 MMD 复核: " + ", ".join(f"{c['domain']} {c['mmd2']:.4f}" for c in check), flush=True)
    print(f"\n结果写入 {OUT}", flush=True)


if __name__ == "__main__":
    main()
