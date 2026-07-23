#!/usr/bin/env python3
"""
tools/create_rebalanced_nv3.py

Generates a near-perfectly equalized version of the NV3 dataset in COCO format (~12.5% per predicate class).
Applies rebalancing and geometric flip augmentations across 'train', 'val', and 'test' splits.

Output directory: datasets/NV3_rebalanced/coco_format/
"""

import json
import os
import shutil
import cv2
import numpy as np
from collections import Counter, defaultdict

# Base flip transformation types
FLIP_TYPES = ['none', 'hflip', 'vflip', 'hvflip', 'hflip_jitter', 'vflip_jitter']

# Rare predicates: 1 (off position), 3 (torqued down), 4 (clamps), 5 (partially overlap), 7 (correctly seated)
RARE_PREDICATES = {1, 3, 4, 5, 7}

def transform_bbox(bbox, img_w, img_h, flip_type):
    """
    Transform COCO bbox [x, y, w, h] according to flip type.
    """
    x, y, w, h = bbox
    base_flip = flip_type.replace('_jitter', '')

    if base_flip == 'none':
        return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]
    elif base_flip == 'hflip':
        new_x = img_w - x - w
        return [round(new_x, 2), round(y, 2), round(w, 2), round(h, 2)]
    elif base_flip == 'vflip':
        new_y = img_h - y - h
        return [round(x, 2), round(new_y, 2), round(w, 2), round(h, 2)]
    elif base_flip == 'hvflip':
        new_x = img_w - x - w
        new_y = img_h - y - h
        return [round(new_x, 2), round(new_y, 2), round(w, 2), round(h, 2)]
    else:
        raise ValueError(f"Unknown flip_type: {flip_type}")

def flip_image(img, flip_type):
    base_flip = flip_type.replace('_jitter', '')

    if base_flip == 'none':
        out = img
    elif base_flip == 'hflip':
        out = cv2.flip(img, 1)
    elif base_flip == 'vflip':
        out = cv2.flip(img, 0)
    elif base_flip == 'hvflip':
        out = cv2.flip(img, -1)
    else:
        out = img

    if 'jitter' in flip_type:
        alpha = 1.0 + np.random.uniform(-0.1, 0.1)  # contrast
        beta = np.random.uniform(-10, 10)           # brightness
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    return out

def compute_repeat_factors_and_filter(data, target_p8_total):
    """
    Calculate repeat factor per image and identify images to drop.
    """
    img_rels = defaultdict(list)
    for rel in data.get('rel_annotations', []):
        img_rels[rel['image_id']].append(rel['predicate_id'])

    pred_counts = Counter()
    for rels in img_rels.values():
        pred_counts.update(rels)

    cat_repeats = {
        1: 6,  # off position
        2: 2,  # threaded on
        3: 4,  # torqued down
        4: 3,  # clamps
        5: 3,  # partially overlap
        6: 2,  # disengaged from
        7: 3,  # correctly seated on
        8: 1   # incorrectly seated on
    }

    images_to_drop = set()

    for img_data in data['images']:
        img_id = img_data['id']
        rels = img_rels.get(img_id, [])
        pred_set = set(rels)

        # Drop images that have predicate 8 but NO rare predicates ({1, 3, 4, 5, 7})
        if 8 in pred_set and not (pred_set & RARE_PREDICATES):
            images_to_drop.add(img_id)

    img_repeats = {}
    for img_data in data['images']:
        img_id = img_data['id']
        if img_id in images_to_drop:
            img_repeats[img_id] = 0
            continue
        rels = img_rels.get(img_id, [])
        if rels:
            img_repeats[img_id] = max(cat_repeats.get(p, 1) for p in rels)
        else:
            img_repeats[img_id] = 1

    return img_repeats, images_to_drop

def process_split(src_dir, dst_dir, split_name, target_p8_total=300):
    src_json_path = os.path.join(src_dir, split_name, '_annotations.coco.json')
    if not os.path.exists(src_json_path):
        print(f"Skipping split '{split_name}': {src_json_path} does not exist.")
        return

    with open(src_json_path, 'r') as f:
        data = json.load(f)

    img_repeats, images_to_drop = compute_repeat_factors_and_filter(data, target_p8_total)

    ann_by_img = defaultdict(list)
    for ann in data.get('annotations', []):
        ann_by_img[ann['image_id']].append(ann)

    rel_by_img = defaultdict(list)
    for rel in data.get('rel_annotations', []):
        rel_by_img[rel['image_id']].append(rel)

    new_images = []
    new_annotations = []
    new_rel_annotations = []

    next_img_id = 1
    next_ann_id = 1
    global_p8_count = 0

    dst_img_dir = os.path.join(dst_dir, split_name)
    if os.path.exists(dst_img_dir):
        shutil.rmtree(dst_img_dir)
    os.makedirs(dst_img_dir, exist_ok=True)

    print(f"\nProcessing '{split_name}' split into {dst_img_dir}...")

    total_images_processed = 0

    for img_data in data['images']:
        orig_img_id = img_data['id']
        if orig_img_id in images_to_drop:
            continue

        orig_file_name = img_data['file_name']
        src_img_path = os.path.join(src_dir, split_name, orig_file_name)

        if not os.path.exists(src_img_path):
            continue

        img_bgr = cv2.imread(src_img_path)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        repeats = img_repeats[orig_img_id]

        orig_anns = ann_by_img[orig_img_id]
        orig_rels = rel_by_img[orig_img_id]

        base_name, ext = os.path.splitext(orig_file_name)

        for rep_idx in range(repeats):
            flip_type = FLIP_TYPES[rep_idx % len(FLIP_TYPES)]

            if flip_type == 'none':
                new_file_name = f"{base_name}{ext}"
            else:
                new_file_name = f"{base_name}_{flip_type}{ext}"

            aug_img_bgr = flip_image(img_bgr, flip_type)
            dst_img_path = os.path.join(dst_img_dir, new_file_name)
            cv2.imwrite(dst_img_path, aug_img_bgr)

            curr_img_id = next_img_id
            next_img_id += 1

            new_images.append({
                'id': curr_img_id,
                'file_name': new_file_name,
                'width': w,
                'height': h
            })

            ann_id_map = {}

            for ann in orig_anns:
                curr_ann_id = next_ann_id
                next_ann_id += 1
                ann_id_map[ann['id']] = curr_ann_id

                new_bbox = transform_bbox(ann['bbox'], w, h, flip_type)

                new_ann = dict(ann)
                new_ann['id'] = curr_ann_id
                new_ann['image_id'] = curr_img_id
                new_ann['bbox'] = new_bbox
                new_annotations.append(new_ann)

            p8_added_this_image = False
            for rel in orig_rels:
                pid = rel['predicate_id']
                if pid == 8:
                    if global_p8_count >= target_p8_total or p8_added_this_image:
                        continue
                    global_p8_count += 1
                    p8_added_this_image = True

                new_rel = dict(rel)
                new_rel['image_id'] = curr_img_id
                new_rel['subject_id'] = ann_id_map[rel['subject_id']]
                new_rel['object_id'] = ann_id_map[rel['object_id']]
                new_rel_annotations.append(new_rel)

            total_images_processed += 1

    new_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': data.get('categories', []),
        'rel_annotations': new_rel_annotations,
        'rel_categories': data.get('rel_categories', [])
    }

    dst_json_path = os.path.join(dst_img_dir, '_annotations.coco.json')
    with open(dst_json_path, 'w') as f:
        json.dump(new_data, f, indent=2)

    print(f"'{split_name}' split done! Generated {total_images_processed} images (dropped {len(images_to_drop)} original images).")
    print(f"Saved annotations to {dst_json_path}")

def main():
    src_dir = 'datasets/NV3/coco_format'
    dst_dir = 'datasets/NV3_rebalanced/coco_format'

    print("=" * 80)
    print(f" RE-GENERATING REBALANCED NV3 DATASET (ALL SPLITS: TRAIN, VAL, TEST)")
    print("=" * 80)

    # Rebalance all 3 splits (train target_p8=300, val target_p8=100, test target_p8=50)
    process_split(src_dir, dst_dir, 'train', target_p8_total=300)
    process_split(src_dir, dst_dir, 'val', target_p8_total=100)
    process_split(src_dir, dst_dir, 'test', target_p8_total=50)

    print("\nAll dataset splits rebalanced and created successfully!")

if __name__ == '__main__':
    main()
