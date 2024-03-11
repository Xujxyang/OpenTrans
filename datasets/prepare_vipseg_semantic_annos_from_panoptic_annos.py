#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from fvcore.common.download import download
from panopticapi.utils import rgb2id
from PIL import Image

# from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

VIPSeg_CATEGORIES = [
    { "id": 0, "name": "wall", "isthing": 0,},
    { "id": 1, "name": "ceiling", "isthing": 0,},
    { "id": 2, "name": "door", "isthing": 1,},
    { "id": 3, "name": "stair", "isthing": 0,},
    { "id": 4, "name": "ladder", "isthing": 1,},
    { "id": 5, "name": "escalator", "isthing": 0,},
    { "id": 6, "name": "Playground_slide", "isthing": 0,},
    { "id": 7, "name": "handrail_or_fence", "isthing": 0,},
    { "id": 8, "name": "window", "isthing": 1,},
    { "id": 9, "name": "rail", "isthing": 0,},
    { "id": 10, "name": "goal", "isthing": 1,},
    { "id": 11, "name": "pillar", "isthing": 0,},
    { "id": 12, "name": "pole", "isthing": 0,},
    { "id": 13, "name": "floor", "isthing": 0,},
    { "id": 14, "name": "ground", "isthing": 0,},
    { "id": 15, "name": "grass", "isthing": 0,},
    { "id": 16, "name": "sand", "isthing": 0,},
    { "id": 17, "name": "athletic_field", "isthing": 0,},
    { "id": 18, "name": "road", "isthing": 0,},
    { "id": 19, "name": "path", "isthing": 0,},
    { "id": 20, "name": "crosswalk", "isthing": 0,},
    { "id": 21, "name": "building", "isthing": 0,},
    { "id": 22, "name": "house", "isthing": 0,},
    { "id": 23, "name": "bridge", "isthing": 0,},
    { "id": 24, "name": "tower", "isthing": 0,},
    { "id": 25, "name": "windmill", "isthing": 0,},
    { "id": 26, "name": "well_or_well_lid", "isthing": 0,},
    { "id": 27, "name": "other_construction", "isthing": 0,},
    { "id": 28, "name": "sky", "isthing": 0,},
    { "id": 29, "name": "mountain", "isthing": 0,},
    { "id": 30, "name": "stone", "isthing": 0,},
    { "id": 31, "name": "wood", "isthing": 0,},
    { "id": 32, "name": "ice", "isthing": 0,},
    { "id": 33, "name": "snowfield", "isthing": 0,},
    { "id": 34, "name": "grandstand", "isthing": 0,},
    { "id": 35, "name": "sea", "isthing": 0,},
    { "id": 36, "name": "river", "isthing": 0,},
    { "id": 37, "name": "lake", "isthing": 0,},
    { "id": 38, "name": "waterfall", "isthing": 0,},
    { "id": 39, "name": "water", "isthing": 0,},
    { "id": 40, "name": "billboard_or_Bulletin_Board", "isthing": 0,},
    { "id": 41, "name": "sculpture", "isthing": 1,},
    { "id": 42, "name": "pipeline", "isthing": 0,},
    { "id": 43, "name": "flag", "isthing": 1,},
    { "id": 44, "name": "parasol_or_umbrella", "isthing": 1,},
    { "id": 45, "name": "cushion_or_carpet", "isthing": 0,},
    { "id": 46, "name": "tent", "isthing": 1,},
    { "id": 47, "name": "roadblock", "isthing": 1,},
    { "id": 48, "name": "car", "isthing": 1,},
    { "id": 49, "name": "bus", "isthing": 1,},
    { "id": 50, "name": "truck", "isthing": 1,},
    { "id": 51, "name": "bicycle", "isthing": 1,},
    { "id": 52, "name": "motorcycle", "isthing": 1,},
    { "id": 53, "name": "wheeled_machine", "isthing": 0,},
    { "id": 54, "name": "ship_or_boat", "isthing": 1,},
    { "id": 55, "name": "raft", "isthing": 1,},
    { "id": 56, "name": "airplane", "isthing": 1,},
    { "id": 57, "name": "tyre", "isthing": 0,},
    { "id": 58, "name": "traffic_light", "isthing": 0,},
    { "id": 59, "name": "lamp", "isthing": 0,},
    { "id": 60, "name": "person", "isthing": 1,},
    { "id": 61, "name": "cat", "isthing": 1,},
    { "id": 62, "name": "dog", "isthing": 1,},
    { "id": 63, "name": "horse", "isthing": 1,},
    { "id": 64, "name": "cattle", "isthing": 1,},
    { "id": 65, "name": "other_animal", "isthing": 1,},
    { "id": 66, "name": "tree", "isthing": 0,},
    { "id": 67, "name": "flower", "isthing": 0,},
    { "id": 68, "name": "other_plant", "isthing": 0,},
    { "id": 69, "name": "toy", "isthing": 0,},
    { "id": 70, "name": "ball_net", "isthing": 0,},
    { "id": 71, "name": "backboard", "isthing": 0,},
    { "id": 72, "name": "skateboard", "isthing": 1,},
    { "id": 73, "name": "bat", "isthing": 0,},
    { "id": 74, "name": "ball", "isthing": 1,},
    { "id": 75, "name": "cupboard_or_showcase_or_storage_rack", "isthing": 0,},
    { "id": 76, "name": "box", "isthing": 1,},
    { "id": 77, "name": "traveling_case_or_trolley_case", "isthing": 1,},
    { "id": 78, "name": "basket", "isthing": 1,},
    { "id": 79, "name": "bag_or_package", "isthing": 1,},
    { "id": 80, "name": "trash_can", "isthing": 0,},
    { "id": 81, "name": "cage", "isthing": 0,},
    { "id": 82, "name": "plate", "isthing": 1,},
    { "id": 83, "name": "tub_or_bowl_or_pot", "isthing": 1,},
    { "id": 84, "name": "bottle_or_cup", "isthing": 1,},
    { "id": 85, "name": "barrel", "isthing": 1,},
    { "id": 86, "name": "fishbowl", "isthing": 1,},
    { "id": 87, "name": "bed", "isthing": 1,},
    { "id": 88, "name": "pillow", "isthing": 1,},
    { "id": 89, "name": "table_or_desk", "isthing": 1,},
    { "id": 90, "name": "chair_or_seat", "isthing": 1,},
    { "id": 91, "name": "bench", "isthing": 1,},
    { "id": 92, "name": "sofa", "isthing": 1,},
    { "id": 93, "name": "shelf", "isthing": 0,},
    { "id": 94, "name": "bathtub", "isthing": 0,},
    { "id": 95, "name": "gun", "isthing": 1,},
    { "id": 96, "name": "commode", "isthing": 1,},
    { "id": 97, "name": "roaster", "isthing": 1,},
    { "id": 98, "name": "other_machine", "isthing": 0,},
    { "id": 99, "name": "refrigerator", "isthing": 1,},
    { "id": 100, "name": "washing_machine", "isthing": 1,},
    { "id": 101, "name": "Microwave_oven", "isthing": 1,},
    { "id": 102, "name": "fan", "isthing": 1,},
    { "id": 103, "name": "curtain", "isthing": 0,},
    { "id": 104, "name": "textiles", "isthing": 0,},
    { "id": 105, "name": "clothes", "isthing": 0,},
    { "id": 106, "name": "painting_or_poster", "isthing": 1,},
    { "id": 107, "name": "mirror", "isthing": 1,},
    { "id": 108, "name": "flower_pot_or_vase", "isthing": 1,},
    { "id": 109, "name": "clock", "isthing": 1,},
    { "id": 110, "name": "book", "isthing": 0,},
    { "id": 111, "name": "tool", "isthing": 0,},
    { "id": 112, "name": "blackboard", "isthing": 0,},
    { "id": 113, "name": "tissue", "isthing": 0,},
    { "id": 114, "name": "screen_or_television", "isthing": 1,},
    { "id": 115, "name": "computer", "isthing": 1,},
    { "id": 116, "name": "printer", "isthing": 1,},
    { "id": 117, "name": "Mobile_phone", "isthing": 1,},
    { "id": 118, "name": "keyboard", "isthing": 1,},
    { "id": 119, "name": "other_electronic_product", "isthing": 0,},
    { "id": 120, "name": "fruit", "isthing": 0,},
    { "id": 121, "name": "food", "isthing": 0,},
    { "id": 122, "name": "instrument", "isthing": 1,},
    { "id": 123, "name": "train", "isthing": 1,}
]

def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.
    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.
    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(categories) <= 254
    for i, k in enumerate(categories):
        id_map[k["id"]] = i
    # what is id = 0?
    # id_map[0] = 255
    print(id_map)

    with open(panoptic_json) as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for v_anno in obj["annotations"]:
            v_id = v_anno["video_id"]
            os.makedirs(os.path.join(sem_seg_root, v_id), exist_ok=True)
            for anno in v_anno["annotations"]:
                file_name = anno["file_name"]
                segments = anno["segments_info"]
                input = os.path.join(panoptic_root, v_id, file_name)
                output = os.path.join(sem_seg_root, v_id, file_name)
                yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "VIPSeg/VIPSeg_720P")
    novel_clsID = [78, 94, 112, 76, 102, 32, 122, 116, 80, 57, 34, 11]
    VIPSeg_NOVEL_CATEGORIES = [k for k in VIPSeg_CATEGORIES if k["id"] in novel_clsID]
    VIPSeg_BASE_CATEGORIES = [k for k in VIPSeg_CATEGORIES if k["id"] not in novel_clsID]
    for s in ["val_base", "val_novel"]:
        separate_coco_semantic_from_panoptic(
            os.path.join(dataset_dir, "panoptic_gt_VIPSeg_{}.json".format(s)),
            os.path.join(dataset_dir, "panomasksRGB"),
            os.path.join(dataset_dir, "panoptic_semseg_{}".format(s)),
            VIPSeg_NOVEL_CATEGORIES if 'novel' in s else VIPSeg_BASE_CATEGORIES,
        )
    separate_coco_semantic_from_panoptic(
        os.path.join(dataset_dir, "panoptic_gt_VIPSeg_val.json"),
        os.path.join(dataset_dir, "panomasksRGB"),
        os.path.join(dataset_dir, "panoptic_semseg_val"),
        VIPSeg_CATEGORIES,
    )