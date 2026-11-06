"""域对网格（E4）：合并两个源配置（MCF-only 与 MCF+IU），共 11 个域对。
每个 (源配置, 目标域) 是一个独立的观测单位：MMD2 相对该源配置的模型计算。
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
import config, train_eval as TE

FEATURE_MODULE = "model.1.submodule.1.submodule.1.submodule.1.submodule"
BASE = config.WORK; OUT = BASE / "outputs" / "upstream"
CONFIGS = [("MCF", "splits_pairs_mcf.json", "runs_pairs_mcf", "results_pairs_mcf.csv", "_pairs_mcf"),
           ("MCF+IU", "splits_pairs_mcfiu.json", "runs_pairs_mcfiu", "results_pairs_mcfiu.csv", "_pairs_mcfiu")]
SEEDS = (0, 1, 2)

@torch.no_grad()
def embed(model, keys, dev, batch=12):
    m = dict(model.named_modules())[FEATURE_MODULE]; buf = []
    def cap(_m, _i, o): buf.append(o.detach().mean(dim=(-2, -1)).float().cpu().numpy())
    h = m.register_forward_hook(cap); res = {}
    try:
        for k in keys:
            img, _ = TE.load_case(k); vol = TE.volume_to_25d(img)
            zs = np.unique(np.linspace(0, len(img) - 1, 5).round().astype(int))
            for i in range(0, len(zs), batch):
                xt = torch.from_numpy(TE.norm(vol[zs[i:i + batch]])).to(dev)
                with TE.amp_ctx(dev): model(xt)
            res[k] = np.concatenate(buf, 0).mean(0); buf.clear()
    finally: h.remove()
    return res

def kf(a, b, g):
    d2 = np.maximum((a*a).sum(1)[:, None] + (b*b).sum(1)[None, :] - 2*a@b.T, 0)
    return np.exp(-g*d2)
def mmd2(x, y, g): return float(kf(x, x, g).mean() + kf(y, y, g).mean() - 2*kf(x, y, g).mean())

def shift_of(t):
    return "跨序列" if t == "OOP" else "跨模态" if t == "CT" else "跨中心"

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for cfg, spf, rundir, resf, tag in CONFIGS:
        sp = json.loads((BASE / spf).read_text(encoding="utf-8"))
        res = list(csv.DictReader(open(BASE / resf, encoding="utf-8")))
        def dice_of(t, n):
            v = [float(r["dice"]) for r in res if r["target"] == t and int(r["n_label"]) == n]
            return (float(np.mean(v)), float(np.std(v, ddof=1))) if len(v) > 1 else (float(np.mean(v)), 0.0)
        mm = {t: [] for t in sp["targets"]}
        sv = []
        for seed in SEEDS:
            model = TE.build_model()
            model.load_state_dict(torch.load(BASE / rundir / f"source_T1W_s{seed}{tag}.pt", map_location="cpu"))
            model.to(dev).eval()
            sv.append(float(json.loads((BASE / rundir / f"source_T1W_s{seed}{tag}.json").read_text(encoding="utf-8"))["source_val_dice"]))
            sk = sp["source"]["train"] + sp["source"]["val"]
            src = embed(model, sk, dev); S = np.asarray([src[k] for k in sk], float)
            mu, sd = S.mean(0), S.std(0); sd[sd < 1e-8] = 1.0; Sz = (S - mu) / sd
            d2 = np.sum(Sz*Sz, 1)[:, None] + np.sum(Sz*Sz, 1)[None, :] - 2*Sz@Sz.T
            nz = d2[np.triu_indices(len(Sz), 1)]; nz = nz[nz > 0]; g = 1.0 / float(np.median(nz))
            for t, v in sp["targets"].items():
                fe = embed(model, v["eval"], dev)
                E = np.asarray([(fe[k]-mu)/sd for k in v["eval"]], float)
                mm[t].append(mmd2(Sz, E, g))
            del model
            if dev == "cuda": torch.cuda.empty_cache()
        for t in sp["targets"]:
            z0, z0s = dice_of(t, 0); z10, z10s = dice_of(t, 10)
            rows.append(dict(source_cfg=cfg, target=t, shift=shift_of(t),
                             n_eval=len(sp["targets"][t]["eval"]),
                             source_val_dice=round(float(np.mean(sv)), 4),
                             mmd2_eval=round(float(np.mean(mm[t])), 4),
                             mmd2_sd=round(float(np.std(mm[t], ddof=1)), 4) if len(mm[t]) > 1 else 0.0,
                             zero_shot=round(z0, 4), zero_shot_sd=round(z0s, 4),
                             k10=round(z10, 4), k10_sd=round(z10s, 4), gain=round(z10 - z0, 4)))
    with (OUT / "pair_grid_all.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"{'源配置':8}{'目标':9}{'类型':8}{'源val':>8}{'MMD2':>9}{'零样本':>9}{'SD':>7}{'k10':>9}{'增益':>9}")
    for r in rows:
        print(f"{r['source_cfg']:8}{r['target']:9}{r['shift']:8}{r['source_val_dice']:>8.3f}{r['mmd2_eval']:>9.4f}"
              f"{r['zero_shot']:>9.4f}{r['zero_shot_sd']:>7.3f}{r['k10']:>9.4f}{r['gain']:>+9.4f}")
    def rk(a):
        a = np.asarray(a, float); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a)+1); return r
    def sp2(x, y):
        rx, ry = rk(x)-np.mean(rk(x)), rk(y)-np.mean(rk(y))
        return float((rx*ry).sum()/np.sqrt((rx*rx).sum()*(ry*ry).sum()))
    m = [r["mmd2_eval"] for r in rows]; z = [r["zero_shot"] for r in rows]; g = [r["gain"] for r in rows]
    print(f"\nn = {len(rows)} 个域对（2 个源配置 × 3 种子）")
    print(f"  MMD2 与零样本    {sp2(m,z):+.3f}")
    print(f"  MMD2 与增益      {sp2(m,g):+.3f}")
    print(f"  零样本 与增益     {sp2(z,g):+.3f}")
    sub = [r for r in rows if r["shift"] != "跨中心"]
    print(f"\n去掉跨中心后 n = {len(sub)}：MMD2 与零样本 {sp2([r['mmd2_eval'] for r in sub],[r['zero_shot'] for r in sub]):+.3f}")
    print(f"\n写入 {OUT/'pair_grid_all.csv'}")

if __name__ == "__main__": main()
