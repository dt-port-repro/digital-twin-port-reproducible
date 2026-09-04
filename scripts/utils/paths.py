"""路径工具 — 数据目录定位。

所有路径相对于复现包根目录（scripts/ 的父目录）。
"""
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # utils/ → scripts/ → 复现包根目录


def raw_2024_dir() -> Path:
    """返回原始数据目录: data/raw/2024/"""
    p = _REPO_ROOT / "data" / "raw" / "2024"
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir() -> Path:
    """返回处理数据目录: data/processed/"""
    p = _REPO_ROOT / "data" / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p
