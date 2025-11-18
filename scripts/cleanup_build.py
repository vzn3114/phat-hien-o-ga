"""
Utility script to remove old build artifacts (build/, dist/, __pycache__/).
Run this before tạo bản build mới để giữ dự án gọn gàng.
"""

from pathlib import Path
import shutil

CLEAN_TARGETS = [
    "build",
    "dist",
    "__pycache__",
]

ROOT = Path(__file__).resolve().parents[1]


def remove_path(path: Path):
    if not path.exists():
        return
    if path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def main():
    for target in CLEAN_TARGETS:
        full_path = ROOT / target
        if full_path.exists():
            print(f"🧹 Removing {full_path}")
            remove_path(full_path)
        else:
            print(f"✅ Skip {full_path} (not found)")


if __name__ == "__main__":
    main()

