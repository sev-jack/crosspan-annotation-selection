"""实验配置：路径、域定义、预算档位、训练超参。

改域、改预算档位、改 epoch 都只动这个文件，其它脚本不用碰。
"""

import os
import sys
from pathlib import Path


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip() not in ("0", "", "false", "False")

# ---------- 路径 ----------
ROOT = Path(r"<DATA_ROOT>\CrossPan")
# CT 数据集（MSD Task07 胰腺）。指标签 1 = 胰腺，2 = 肿瘤（肿瘤算背景）。
CT_ROOT = Path(os.environ.get("EXP_CT_ROOT", r"<DATA_ROOT>\MSD_Task07\Task07_Pancreas"))
CT_HU_WINDOW = (-100.0, 240.0)   # 腹部软组织窗，CT 的标准做法
CT_EVAL_FRAC = 0.22              # 282 例里留 22% 作评估集（约 62 例）
# 工作目录。本机是 Windows 路径；换到别的机器（比如实验室服务器）时
# 用环境变量 EXP_WORK 指向那边的项目目录即可，不用改代码。
WORK = Path(os.environ.get("EXP_WORK", r"<WORKSPACE>"))
# 缓存目录、输入尺寸、2.5D 层数、权重后缀都可用环境变量切换，
# 这样一条脚本就能把多个消融配置串起来跑，互不覆盖。
def _work_path_env(name, default):
    p = Path(os.environ.get(name, default))
    return p if p.is_absolute() else WORK / p


CACHE = _work_path_env("EXP_CACHE", "cache")
RUNS = _work_path_env("EXP_RUNS", "runs")
RESULTS = _work_path_env("EXP_RESULTS", "results.csv")
SPLITS = _work_path_env("EXP_SPLITS", "splits.json")
LOG = _work_path_env("EXP_LOG", "run.log")

# ---------- 预处理 ----------
SIZE = _env_int("EXP_SIZE", 256)      # 每张轴位切片统一缩放到 SIZE x SIZE
N_SLICES = _env_int("EXP_N_SLICES", 3)  # 2.5D 通道数（相邻切片数，须为奇数）
RUN_TAG = os.environ.get("EXP_TAG", "")  # 追加到权重文件名，避免不同实验互相覆盖
# 平面内重采样间距。必须随 SIZE 反比缩放，才能保持物理视场不变：
#   SIZE=256 -> 1.5 mm（视场 384 mm）
#   SIZE=384 -> 1.0 mm（同样 384 mm 视场，但每毫米更多像素 = 真·分辨率提升）
# 如果不缩放间距，SIZE=384 只会把 260 px 的图填黑边到 384，视场变成 576mm，
# 器官反而更小（实测前景占比会从 0.40% 掉到 0.12%）。
TARGET_SPACING = float(os.environ.get("EXP_SPACING", 384.0 / SIZE))
FG_LABEL = 1               # 前景标签值（先跑 check_data.py 确认）

# ---------- 域定义 ----------
# 源域：T1/train 全部 + T1/test 里**非 NYU** 的部分。
# 把 NYU 整中心留出来，才能构成真正的"跨中心"目标域
# （之前的 T1-C 中心和源域重合，名不副实）。
SOURCES = {
    "T1W": dict(specs=[("T1", "train"), ("T1", "test")], exclude_centers=["NYU"]),
}

# 目标域：
#   pool = 可以花标注预算的病例池
#   eval = 评估用的病例（绝不参与训练）
#   eval_frac = 当 pool 与 eval 同源时，按比例切分（None 表示用独立划分）
#   centers = 只保留这些中心的病例（整中心留出用）
TARGETS = {
    "OOP":  dict(pool=[("OOP", "train")], eval=[("OOP", "val"), ("OOP", "test")], eval_frac=None),
    "NYU":  dict(pool=[("T1", "test")], eval=None, eval_frac=0.5, centers=["NYU"]),
}

# 需要预处理的所有 (序列, 划分)
ALL_SPECS = [
    ("T1", "train"),
    ("T1", "test"),
    ("OOP", "train"),
    ("OOP", "val"),
    ("OOP", "test"),
]

# ---------- 预算档位 ----------
BUDGETS = [0.00, 0.01, 0.05, 0.10, 0.20, 1.00]
SEEDS = [0, 1, 2]
SPLIT_SEED = 0             # 固定，保证所有人跑出的划分一致
SEED_PROTOCOL = "seeded_v2" # Python / NumPy / PyTorch CPU+CUDA；cuDNN deterministic

# ---------- 训练 ----------
SOURCE_EPOCHS = 300
FINETUNE_EPOCHS = 60
BATCH = 32
SOURCE_LR = 1e-3
FINETUNE_LR = 1e-4
SRC_VAL_FRAC = 0.15
NUM_WORKERS = 0            # Windows 上设 0 最稳

# 混合精度。4060 是 Ada 架构，bf16 张量核心能白捡 1.5~2 倍速度。
# 这是"用显存换时间"的正确用法，不是把显存花在加容量上。
USE_AMP = True
AMP_DTYPE = "bfloat16"

# 骨干容量。16,32,64,128 太小（实测训练集上也欠拟合），改用标准 4 层 UNet。
MODEL_CHANNELS = (32, 64, 128, 256, 512)
MODEL_STRIDES = (2, 2, 2, 2)

# 消融开关：True = 全部轴位切片；False = 前景切片 + 每例 8 张随机空切片
ALL_SLICES = _env_bool("EXP_ALL_SLICES", True)

# 消融开关：True = 按身体轮廓裁掉空气再统一缩放（各中心尺度归一）；
#           False = 固定 1.5mm 视场后居中裁/填到 SIZE。
# 第 13 条实测：身体裁剪让源域 val Dice 从 0.5894 掉到 0.5693（−0.020），是负收益，
# 所以回到 False。想复现就把这里改成 True，并把 cache 目录换成 cache_v2_bodycrop。
BODY_CROP = _env_bool("EXP_BODY_CROP", False)

# 测试时增强：翻转 + 缩放多视图平均。只影响评估，不影响训练。
USE_TTA = _env_bool("EXP_TTA", True)
# 是否把缩放视图也算进 TTA。实测负收益（36 例验证集：仅翻转 0.5953，加缩放 0.5684），
# 所以默认关闭，只保留翻转。想复现那个对比就把这里改成 True。
TTA_SCALE = _env_bool("EXP_TTA_SCALE", False)


def run_id(source, target, budget, seed):
    pct = f"{int(round(budget * 100)):03d}pct"
    return f"{source}2{target}_{pct}_s{seed}"


# ---------- 日志 ----------
# 关键：屏幕和文件用不同的编码，否则总有一边是乱码。
#   屏幕：跟随系统编码（中文 Windows 是 GBK），终端里看得懂。
#   日志文件：固定 UTF-8，任何编辑器打开都不乱码。
# 所以不要再把输出用 `| Tee-Object` 导进文件——那会让两边共用一种编码。

class _Tee:
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, s):
        try:
            self._stream.write(s)
        except Exception:
            pass
        self._fh.write(s)
        return len(s)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        self._fh.flush()

    def isatty(self):
        return False


def start_log(path):
    """把 stdout/stderr 同时写到 UTF-8 日志文件。返回文件句柄。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    print(f"[日志同时写入] {path}", flush=True)
    return fh
