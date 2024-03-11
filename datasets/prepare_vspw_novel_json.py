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
# from pycocotools.coco import COCO
# from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, COCO_CATEGORIES

# COCO_LEN = 123287
with open('data/VIPSeg/VIPSeg_720P/panoVIPSeg_categories.json','r') as f:
    CATEGORIES = json.load(f)

novel_clsID = [78, 94, 112, 76, 102, 32, 122, 116, 80, 57, 34, 11]
base_clsID = [k["id"] for k in CATEGORIES if k["id"] not in novel_clsID]


def convert_ignore_json(json_path, out_json_path, keepID, remain_img):
    with open(json_path) as f:
        vspw_json = json.load(f)
    # target_json = {}
    keep_count = 0
    ann_len = len(vspw_json["annotations"])
    video_annotation = []
    for i, video_anno in enumerate(vspw_json["annotations"]):
        has_keep, has_remove = False, False
        new_video = {
            'video_id' : video_anno['video_id'],
            # 'ins_id_set' : video_anno['ins_id_set'],
        }
        video_id_set = set()
        frame_annotations = []
        for ann in video_anno["annotations"]:
            new_ann = {'file_name' : ann['file_name'], 
                'image_id' : ann['image_id']}
            seg_info = []

            for seg in ann["segments_info"]:
                if seg['category_id'] in keepID:
                    seg_info.append(seg)
                    video_id_set.add(seg['instance_id'])
                    has_keep = True
                else:
                    has_remove = True
            new_ann['segments_info'] = seg_info
            if len(seg_info) > 0:
                frame_annotations.append(new_ann)

        if has_keep and (remain_img or not has_remove):
            new_video['annotations'] = frame_annotations
            new_video['ins_id_set'] = list(video_id_set)
            video_annotation.append(new_video)
            keep_count += 1
        print("[{}]/[{}], video count: {}".format(i, ann_len, keep_count), end='\r')

    with open(out_json_path, "w") as f:
        json.dump({"annotations" : video_annotation, 'categories' : CATEGORIES}, f)


if __name__ == "__main__":
    convert_ignore_json("data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_train.json", 
        "data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_train_base_ignore.json", base_clsID, True)
    print("\nTrain base ignore DONE.")
    convert_ignore_json("data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_train.json", 
        "data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_train_base_delete.json", base_clsID, False)
    print("\nTrain base delete DONE.")
    convert_ignore_json("data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json", 
        "data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val_base.json", base_clsID, True)
    print("\nVal base DONE.")
    convert_ignore_json("data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json", 
        "data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val_novel.json", novel_clsID, True)
    print("\nVal novel DONE.")