#!/usr/bin/env python3
"""
tools/check_dataset_balance.py

General-purpose dataset balance checker and comparison tool for Scene Graph Generation datasets.

Usage:
  1. Analyze a single dataset:
     python tools/check_dataset_balance.py datasets/NV3/coco_format

  2. Compare two datasets side-by-side (Before vs After):
     python tools/check_dataset_balance.py datasets/NV3/coco_format datasets/NV3_rebalanced/coco_format
"""

import json
import os
import sys

# Try importing numpy, fallback to standard python math if unavailable
try:
    import numpy as np

    def compute_gini(counts):
        counts = np.sort(np.array(counts, dtype=np.float64))
        n = len(counts)
        if n == 0 or np.sum(counts) == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * counts) / (n * np.sum(counts))) - (n + 1) / n
except ImportError:
    def compute_gini(counts):
        counts = sorted(counts)
        n = len(counts)
        total = sum(counts)
        if n == 0 or total == 0:
            return 0.0
        return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(counts)) / (n * total)

from collections import Counter

def load_split_data(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    annotations = data.get('annotations', [])
    rel_annotations = data.get('rel_annotations', [])

    obj_cats = {c['id']: c['name'] for c in data.get('categories', [])}
    rel_cats = {c['id']: c['name'] for c in data.get('rel_categories', [])}

    obj_counts = Counter(ann['category_id'] for ann in annotations)
    rel_counts = Counter(ann['predicate_id'] for ann in rel_annotations)

    return {
        'num_images': len(images),
        'obj_cats': obj_cats,
        'rel_cats': rel_cats,
        'obj_counts': obj_counts,
        'rel_counts': rel_counts,
        'total_objs': sum(obj_counts.values()),
        'total_rels': sum(rel_counts.values())
    }

def print_single_split(split_name, info):
    print("=" * 80)
    print(f" SPLIT: {split_name.upper()}")
    print(f" Images: {info['num_images']} | Objects: {info['total_objs']} | Relationships: {info['total_rels']}")
    print("=" * 80)

    # Objects
    print("\n--- OBJECT CLASS DISTRIBUTION ---")
    print(f"{'ID':<4} {'Class Name':<25} {'Count':<10} {'Percentage':<12}")
    print("-" * 55)
    obj_freqs = []
    for cat_id, name in sorted(info['obj_cats'].items()):
        cnt = info['obj_counts'][cat_id]
        pct = (cnt / info['total_objs'] * 100) if info['total_objs'] > 0 else 0
        obj_freqs.append(cnt)
        print(f"{cat_id:<4} {name:<25} {cnt:<10} {pct:.2f}%")

    max_o = max(obj_freqs) if obj_freqs else 1
    min_o = min(obj_freqs) if obj_freqs else 1
    print(f"\nObject Imbalance Ratio (Max/Min): {max_o / max(min_o, 1):.2f}x")
    print(f"Object Gini Coefficient: {compute_gini(obj_freqs):.3f}")

    # Predicates
    print("\n--- PREDICATE CLASS DISTRIBUTION ---")
    print(f"{'ID':<4} {'Predicate Name':<25} {'Count':<10} {'Percentage':<12}")
    print("-" * 55)
    rel_freqs = []
    for rel_id, name in sorted(info['rel_cats'].items()):
        cnt = info['rel_counts'][rel_id]
        pct = (cnt / info['total_rels'] * 100) if info['total_rels'] > 0 else 0
        rel_freqs.append(cnt)
        print(f"{rel_id:<4} {name:<25} {cnt:<10} {pct:.2f}%")

    max_r = max(rel_freqs) if rel_freqs else 1
    min_r = min(rel_freqs) if rel_freqs else 1
    print(f"\nPredicate Imbalance Ratio (Max/Min): {max_r / max(min_r, 1):.2f}x")
    print(f"Predicate Gini Coefficient: {compute_gini(rel_freqs):.3f}\n")

def compare_splits(split_name, info1, info2, path1_label="BEFORE", path2_label="AFTER"):
    print("=" * 95)
    print(f" COMPARISON [{split_name.upper()}]: {path1_label} vs {path2_label}")
    print(f" Images        : {info1['num_images']}  ->  {info2['num_images']} (Shift: +{info2['num_images'] - info1['num_images']})")
    print(f" Objects       : {info1['total_objs']}  ->  {info2['total_objs']} (Shift: +{info2['total_objs'] - info1['total_objs']})")
    print(f" Relationships : {info1['total_rels']}  ->  {info2['total_rels']} (Shift: +{info2['total_rels'] - info1['total_rels']})")
    print("=" * 95)

    # Predicate comparison table
    print("\n--- PREDICATE CLASS COMPARISON ---")
    header = f"{'ID':<4} {'Predicate Name':<25} {path1_label + ' Count':<14} {path1_label + ' %':<10} {path2_label + ' Count':<14} {path2_label + ' %':<10} {'Change':<10}"
    print(header)
    print("-" * len(header))

    rel_cats = dict(info1['rel_cats'])
    rel_cats.update(info2['rel_cats'])

    rel_freqs1, rel_freqs2 = [], []

    for rel_id, name in sorted(rel_cats.items()):
        c1 = info1['rel_counts'][rel_id]
        p1 = (c1 / info1['total_rels'] * 100) if info1['total_rels'] > 0 else 0
        c2 = info2['rel_counts'][rel_id]
        p2 = (c2 / info2['total_rels'] * 100) if info2['total_rels'] > 0 else 0

        rel_freqs1.append(c1)
        rel_freqs2.append(c2)

        ratio_str = f"{c2 / c1:.2f}x" if c1 > 0 else "N/A"
        print(f"{rel_id:<4} {name:<25} {c1:<14} {p1:<9.2f}% {c2:<14} {p2:<9.2f}% {ratio_str:<10}")

    ir1 = max(rel_freqs1) / max(min(rel_freqs1), 1) if rel_freqs1 else 1
    ir2 = max(rel_freqs2) / max(min(rel_freqs2), 1) if rel_freqs2 else 1
    gini1 = compute_gini(rel_freqs1)
    gini2 = compute_gini(rel_freqs2)

    print("\n--- METRICS COMPARISON ---")
    print(f"Predicate Imbalance Ratio (Max/Min) : {ir1:.2f}x  ->  {ir2:.2f}x")
    print(f"Predicate Gini Coefficient (0=equal): {gini1:.3f}  ->  {gini2:.3f}")
    print()

def resolve_path(arg):
    if os.path.isfile(arg):
        return arg
    elif os.path.isdir(arg):
        # Check train _annotations.coco.json
        candidate = os.path.join(arg, 'train', '_annotations.coco.json')
        if os.path.exists(candidate):
            return candidate
        candidate2 = os.path.join(arg, '_annotations.coco.json')
        if os.path.exists(candidate2):
            return candidate2
    return arg

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single dataset : python tools/check_dataset_balance.py <dataset_dir_or_json>")
        print("  Compare 2 datasets: python tools/check_dataset_balance.py <before_dataset> <after_dataset>")
        sys.exit(1)

    arg1 = sys.argv[1]
    arg2 = sys.argv[2] if len(sys.argv) > 2 else None

    splits = ['train', 'val', 'test']

    if arg2 is None:
        # Single dataset analysis
        for split in splits:
            json_path = os.path.join(arg1, split, '_annotations.coco.json') if os.path.isdir(arg1) else arg1
            info = load_split_data(json_path)
            if info:
                print_single_split(split, info)
    else:
        # Side-by-side comparison
        print(f"\nComparing Datasets:")
        print(f"  BEFORE (Dataset 1): {arg1}")
        print(f"  AFTER  (Dataset 2): {arg2}\n")

        for split in splits:
            path1 = os.path.join(arg1, split, '_annotations.coco.json') if os.path.isdir(arg1) else arg1
            path2 = os.path.join(arg2, split, '_annotations.coco.json') if os.path.isdir(arg2) else arg2

            info1 = load_split_data(path1)
            info2 = load_split_data(path2)

            if info1 and info2:
                compare_splits(split, info1, info2, path1_label="BEFORE", path2_label="AFTER")

if __name__ == '__main__':
    main()
