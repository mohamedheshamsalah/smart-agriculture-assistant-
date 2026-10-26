"""Download the two CSV datasets; optionally download original soil images.

Run from the project directory:
    python src/download_data.py
    python src/download_data.py --images  # optional ~238 MB of image files
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from pathlib import Path
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"User-Agent": "SmartAgricultureStudentProject/1.0"}


def download_tabular():
    crop_dir = ROOT / "data" / "crop"
    fertility_dir = ROOT / "data" / "fertility"
    crop_dir.mkdir(parents=True, exist_ok=True)
    fertility_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading crop dataset from Kaggle...")
    response = requests.get(
        "https://www.kaggle.com/api/v1/datasets/download/atharvaingle/crop-recommendation-dataset",
        headers=HEADERS, timeout=120,
    )
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        (crop_dir / "Crop_recommendation.csv").write_bytes(archive.read(csv_name))
    print(f"Saved {crop_dir / 'Crop_recommendation.csv'}")

    print("Downloading Soil Sight dataset version 2 from Mendeley Data...")
    response = requests.get(
        "https://data.mendeley.com/public-api/zip/h73wn2nw56/download/2",
        headers=HEADERS, timeout=180,
    )
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        candidates = [name for name in archive.namelist()
                      if name.lower().endswith(".csv") and "version_2" in name.lower()]
        if not candidates:
            raise RuntimeError("The Mendeley archive did not contain the expected Version 2 CSV.")
        (fertility_dir / "soil_sight.csv").write_bytes(archive.read(candidates[0]))
    print(f"Saved {fertility_dir / 'soil_sight.csv'}")


def download_images():
    destination = ROOT / "data" / "soil_images"
    destination.mkdir(parents=True, exist_ok=True)
    api = "https://api.github.com/repos/Phantom-fs/Soil-Classification-Dataset/git/trees/main?recursive=1"
    response = requests.get(api, headers=HEADERS, timeout=60)
    response.raise_for_status()
    entries = response.json()["tree"]
    prefix = "Orignal-Dataset/"
    images = [item for item in entries if item["type"] == "blob"
              and item["path"].startswith(prefix)
              and Path(item["path"]).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not images:
        raise RuntimeError("Original image folder was not found in the GitHub repository.")
    print(f"Downloading {len(images):,} original images. This is about 238 MB.")
    base = "https://raw.githubusercontent.com/Phantom-fs/Soil-Classification-Dataset/main/"
    def fetch_one(item):
        relative = item["path"][len(prefix):]
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == item.get("size"):
            return "already downloaded"
        image_response = requests.get(base + requests.utils.quote(item["path"], safe="/"),
                                      headers=HEADERS, timeout=60)
        image_response.raise_for_status()
        target.write_bytes(image_response.content)
        return "downloaded"

    completed = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(fetch_one, item) for item in images]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == len(images):
                print(f"Processed {completed:,}/{len(images):,}")
    print(f"Images saved to {destination}")
    print("Keep the source repository citation and CC BY-NC 4.0 non-commercial restriction.")


def main():
    parser = argparse.ArgumentParser(description="Download official project datasets.")
    parser.add_argument("--images", action="store_true",
                        help="Also download original soil images (~238 MB).")
    args = parser.parse_args()
    download_tabular()
    if args.images:
        download_images()


if __name__ == "__main__":
    main()
