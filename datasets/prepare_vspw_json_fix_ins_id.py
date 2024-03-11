import argparse
import os
import os.path as osp
import shutil
from functools import partial
from glob import glob
from typing import Callable

import mmcv
import numpy as np
from PIL import Image

import json

from tqdm import tqdm
# from pycocotools.coco import COCO
# from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, COCO_CATEGORIES

# COCO_LEN = 123287
with open('data/VIPSeg/VIPSeg_720P/panoVIPSeg_categories.json','r') as f:
    CATEGORIES = json.load(f)

novel_clsID = [78, 94, 112, 76, 102, 32, 122, 116, 80, 57, 34, 11]
base_clsID = [k["id"] for k in CATEGORIES if k["id"] not in novel_clsID]


def convert_ignore_json(json_path, out_json_path):
    with open(json_path) as f:
        vspw_json = json.load(f)
    # target_json = {}

    video_annotation = []
    for video_anno in tqdm(vspw_json["annotations"]):
        # has_keep, has_remove = False, False
        new_video = {'video_id' : video_anno['video_id']}
        frame_annotations = []
        video_id_set = set()
        for ann in video_anno["annotations"]:
            new_ann = {
                'file_name' : ann['file_name'], 
                'image_id' : ann['image_id'],
                'segments_info' : ann['segments_info'],
            }
            video_id_set.update(ann['ins_id_set'])

            frame_annotations.append(new_ann)

        new_video['annotations'] = frame_annotations
        new_video['ins_id_set'] = list(video_id_set)
        video_annotation.append(new_video)

    with open(out_json_path, "w") as f:
        json.dump({"annotations" : video_annotation, 'categories' : CATEGORIES}, f)


if __name__ == "__main__":
    convert_ignore_json("data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_train.json", 
        "data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_train_insID.json")
    print("\nTrain base ignore DONE.")

    convert_ignore_json("data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json", 
        "data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val_insID.json")
    print("\nVal base DONE.")