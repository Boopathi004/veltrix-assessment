from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def prepare_data() -> list[Path]:
    sources = (
        (ROOT / "corpus", DATA_DIR / "corpus"),
        (ROOT / "manifest.json", DATA_DIR / "manifest.json"),
        (ROOT / "veltrix.db", DATA_DIR / "veltrix.db"),
    )
    copied: list[Path] = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for source, destination in sources:
        if not source.exists():
            raise FileNotFoundError(f"Required assignment data is missing: {source}")
        if destination.exists():
            continue
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        copied.append(destination)
    return copied


if __name__ == "__main__":
    for path in prepare_data():
        print(f"Prepared {path.relative_to(ROOT)}")
    print("Existing data was left unchanged.")
