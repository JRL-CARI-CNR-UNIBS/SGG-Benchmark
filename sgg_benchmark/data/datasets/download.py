"""
Dataset downloader for SGG-Benchmark.

Downloads a dataset from Hugging Face Hub and reconstructs the local
COCO-format directory structure expected by the SGG-Benchmark codebase.

Supported datasets
------------------
    PSG       → maelic/PSG-coco-format
                Output : datasets/PSG/coco_format/{train,val,test}/

    VG150     → maelic/VG150-coco-format
                Output : datasets/VG150/VG150_coco_format/{train,val,test}/

    IndoorVG  → maelic/IndoorVG-coco-format
                Output : datasets/IndoorVG/IndoorVG_coco_format/{train,val,test}/

The reconstructed directory layout per split is::

    {output_dir}/{split}/_annotations.coco.json
    {output_dir}/{split}/<image files>   (only when save_images=True)
"""
from __future__ import annotations

import json
from pathlib import Path

# Project root: sgg_benchmark/data/datasets/download.py → 4 levels up
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ---------------------------------------------------------------------------
# Per-dataset configuration
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[str, dict] = {
    "PSG": {
        "hub_repo":    "maelic/PSG-coco-format",
        "default_dir": _PROJECT_ROOT / "datasets/PSG/coco_format",
        "splits":      ("train", "val", "test"),
    },
    "VG150": {
        "hub_repo":    "maelic/VG150-coco-format",
        "default_dir": _PROJECT_ROOT / "datasets/VG150/VG150_coco_format",
        "splits":      ("train", "val", "test"),
    },
    "IndoorVG": {
        "hub_repo":    "maelic/IndoorVG-coco-format",
        "default_dir": _PROJECT_ROOT / "datasets/IndoorVG/IndoorVG_coco_format",
        "splits":      ("train", "val", "test"),
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_metadata(hub_repo: str) -> tuple[list, list]:
    """Load category + predicate lists from the repo's categories.json file."""
    from huggingface_hub import hf_hub_download
    try:
        cats_file = hf_hub_download(hub_repo, "categories.json", repo_type="dataset")
        with open(cats_file) as f:
            meta = json.load(f)
        return meta.get("categories", []), meta.get("rel_categories", [])
    except Exception:
        return [], []


def _build_coco_json(split_ds, categories: list, rel_categories: list) -> dict:
    """
    Reconstruct a COCO-format annotation dict from one HF Dataset split.

    Each row is expected to have the fields produced by push_to_hub.py:
        image, image_id, width, height, file_name, objects, relations
    """
    images = []
    annotations = []
    rel_annotations = []

    for row in split_ds:
        img_id = row["image_id"]
        images.append({
            "id":        img_id,
            "file_name": row["file_name"],
            "width":     row["width"],
            "height":    row["height"],
        })

        for obj in row["objects"]:
            annotations.append({
                "id":           obj["id"],
                "image_id":     img_id,
                "category_id":  obj["category_id"],
                "bbox":         obj["bbox"],
                "area":         obj["area"],
                "iscrowd":      obj["iscrowd"],
                "segmentation": obj.get("segmentation", []),
            })

        for rel in row["relations"]:
            rel_annotations.append({
                "id":           rel["id"],
                "image_id":     img_id,
                "subject_id":   rel["subject_id"],
                "object_id":    rel["object_id"],
                "predicate_id": rel["predicate_id"],
            })

    return {
        "images":          images,
        "annotations":     annotations,
        "rel_annotations": rel_annotations,
        "categories":      categories,
        "rel_categories":  rel_categories,
    }


def _save_images(split_ds, split_dir: Path) -> None:
    """Save PIL images from the HF dataset to *split_dir*."""
    split_dir.mkdir(parents=True, exist_ok=True)
    total = len(split_ds)
    for i, row in enumerate(split_ds, 1):
        dst = split_dir / row["file_name"]
        if not dst.exists():
            row["image"].save(dst)
        if i % 500 == 0 or i == total:
            print(f"    saved {i}/{total} images …", end="\r")
    print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_dataset(
    dataset_name: str,
    output_dir: Path | None = None,
    save_images: bool = False,
) -> None:
    """Download *dataset_name* from HuggingFace Hub into *output_dir*.

    Parameters
    ----------
    dataset_name : str
        One of the keys in :data:`DATASET_CONFIGS` (``"PSG"``, ``"VG150"``,
        ``"IndoorVG"``).
    output_dir : Path | None
        Where to write the reconstructed dataset.  Defaults to the standard
        SGG-Benchmark path for each dataset.
    save_images : bool
        When ``True``, images are also downloaded and saved alongside the
        annotation JSON files.  This can require tens of GB of disk space.
    """
    from datasets import load_dataset  # heavy import — deferred on purpose

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_CONFIGS)}"
        )

    cfg = DATASET_CONFIGS[dataset_name]
    hub_repo:  str  = cfg["hub_repo"]
    splits:   tuple = cfg["splits"]
    local_dir: Path = Path(output_dir) if output_dir is not None else cfg["default_dir"]

    print(f"\n{'='*70}")
    print(f"  Dataset  : {dataset_name}")
    print(f"  Hub repo : {hub_repo}")
    print(f"  Output   : {local_dir}")
    print(f"{'='*70}\n")

    print(f"Downloading {hub_repo} from Hugging Face Hub …")
    dataset_dict = load_dataset(hub_repo)

    categories, rel_categories = _extract_metadata(hub_repo)
    print(f"  categories     : {len(categories)}")
    print(f"  rel_categories : {len(rel_categories)}\n")

    for split in splits:
        if split not in dataset_dict:
            print(f"  [SKIP] split '{split}' not found in the Hub dataset.")
            continue

        split_dir = local_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing split '{split}' ({len(dataset_dict[split])} rows) …")

        coco_json = _build_coco_json(dataset_dict[split], categories, rel_categories)
        ann_file = split_dir / "_annotations.coco.json"
        with open(ann_file, "w") as f:
            json.dump(coco_json, f)
        print(f"  Wrote {ann_file}")
        print(f"    images={len(coco_json['images'])}, "
              f"annotations={len(coco_json['annotations'])}, "
              f"relations={len(coco_json['rel_annotations'])}")

        if save_images:
            print(f"  Saving images to {split_dir} …")
            _save_images(dataset_dict[split], split_dir)

    print("\nAnnotation JSON files written successfully.")

    if not save_images:
        print("\n⚠  Images were NOT downloaded.  You need to supply them separately.")
        if "image_note" in cfg:
            print(cfg["image_note"])
        else:
            print(f"To download them automatically from Hugging Face, run:\n"
                  f"  python tools/download_from_hub.py --dataset {dataset_name} --save-images")
