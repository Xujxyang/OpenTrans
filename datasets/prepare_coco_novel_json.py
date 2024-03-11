import argparse
import os
import os.path as osp
import shutil
from functools import partial
from glob import glob

import mmcv
import numpy as np
from PIL import Image

import json
from pycocotools.coco import COCO
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, COCO_CATEGORIES

COCO_LEN = 123287

def remap_contiguous_ids_to_original_ids(contiguous_ids, stuff_dataset_id_to_contiguous_id):
    original_ids = []
    for contiguous_id in contiguous_ids:
        for original_id, contiguous_value in stuff_dataset_id_to_contiguous_id.items():
            if contiguous_value == contiguous_id:
                original_ids.append(original_id)
                break
    return original_ids

stuff_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79, 92: 80, 93: 81, 95: 82, 100: 83, 107: 84, 109: 85, 112: 86, 118: 87, 119: 88, 122: 89, 125: 90, 128: 91, 130: 92, 133: 93, 138: 94, 141: 95, 144: 96, 145: 97, 147: 98, 148: 99, 149: 100, 151: 101, 154: 102, 155: 103, 156: 104, 159: 105, 161: 106, 166: 107, 168: 108, 171: 109, 175: 110, 176: 111, 177: 112, 178: 113, 180: 114, 181: 115, 184: 116, 185: 117, 186: 118, 187: 119, 188: 120, 189: 121, 190: 122, 191: 123, 192: 124, 193: 125, 194: 126, 195: 127, 196: 128, 197: 129, 198: 130, 199: 131, 200: 132}

unneeded_id = [36, 13, 35, 75, 111, 30, 74, 108, 49, 54]
# [92, 62, 51, 3, 46, 26, 82, 68, 49, 35, 33, 120, 13, 47, 25, 42, 118, 54, 101, 111, 17, 86, 74, 8, 28, 41, 14, 1, 75, 93, 130, 44, 40, 60, 58, 61, 45, 108, 113, 107, 30, 79, 87, 7, 88, 99, 31, 66, 5, 9, 34, 127, 16, 85, 43, 112, 4, 67, 23, 109, 89, 39, 11, 29, 53, 114, 27, 83, 6, 55, 128, 110, 57, 70, 76, 69, 48, 52, 77, 15, 73, 95, 59, 65, 12, 72, 36, 21, 10, 78]
# [122, 87, 102, 92, 45, 120, 55, 64, 52, 4, 110, 41, 34, 20, 106, 61, 70, 6, 42, 57, 1, 10, 101, 85, 69, 56, 96, 59, 11, 84, 98, 43, 127, 63, 53, 71, 128, 51, 26, 0, 17, 14, 18, 46, 12, 7, 33, 68, 86, 31, 80, 104, 130, 37, 62, 60, 109, 129, 121, 93, 114, 126, 88, 15, 79, 16, 72, 47, 24, 58, 3, 83, 40, 76, 29, 123, 5, 95, 8, 38, 36, 13, 35, 75, 111, 30, 74, 108, 49, 54]

novel_clsID = remap_contiguous_ids_to_original_ids(unneeded_id, stuff_dataset_id_to_contiguous_id)

# import pdb; pdb.set_trace()
# novel_clsID = [2, 33, 21, 25, 95, 76, 53, 57, 70, 145, 87, 46, 13, 100, 34, 41, 175, 148]
base_clsID = [k["id"] for k in COCO_CATEGORIES if k["id"] not in novel_clsID]

import pdb; pdb.set_trace()
# NOVEL_CATEGORIES = [k for k in COCO_CATEGORIES if k["id"] in novel_clsID]
# BASE_CATEGORIES = [k for k in COCO_CATEGORIES if k["id"] not in novel_clsID]

# novel_clsID2trainID = {}
# for i, cat in enumerate(NOVEL_CATEGORIES):
#         novel_clsID2trainID[cat["id"]] = i

# base_clsID2trainID = {}
# for i, cat in enumerate(BASE_CATEGORIES):
#         base_clsID2trainID[cat["id"]] = i

# novel_clsID = [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
# base_clsID = [k for k in full_clsID_to_trID.keys() if k not in novel_clsID + [255]]
# novel_clsID_to_trID = {k: i for i, k in enumerate(novel_clsID)}
# base_clsID_to_trID = {k: i for i, k in enumerate(base_clsID)}

def convert_ignore_json(json_path, out_json_path, keepID, remain_img):
    with open(json_path) as f:
        coco_json = json.load(f)
    # target_json = {}
    annotations = []
    keep_count = 0
    ann_len = len(coco_json["annotations"])
    for i, ann in enumerate(coco_json["annotations"]):
        new_ann = {'file_name' : ann['file_name'], 
            'image_id' : ann['image_id']}
        seg_info = []
        has_keep, has_remove = False, False
        for seg in ann["segments_info"]:
            if seg['category_id'] in keepID:
                seg_info.append(seg)
                has_keep = True
            else:
                has_remove = True
        if has_keep and (remain_img or not has_remove):
            new_ann['segments_info'] = seg_info
            annotations.append(new_ann)
            keep_count += 1
        print("[{}]/[{}], image count: {}".format(i, ann_len, keep_count), end='\r')

    with open(out_json_path, "w") as f:
        json.dump({"annotations" : annotations}, f)


if __name__ == "__main__":
    convert_ignore_json("/opt/data/private/xjx/data/coco/annotations/panoptic_train2017.json", 
        "/opt/data/private/xjx/data/coco/annotations/panoptic_train2017_entropy_ignore_8.json", base_clsID, True)
    print("Train base ignore DONE.")
    convert_ignore_json("/opt/data/private/xjx/data/coco/annotations/panoptic_train2017.json", 
        "/opt/data/private/xjx/data/coco/annotations/panoptic_train2017_entropy_delete_8.json", base_clsID, False)
    print("Train base delete DONE.")
    # convert_ignore_json("data/coco/annotations/panoptic_val2017.json", 
    #     "data/coco/annotations/panoptic_val2017_base.json", base_clsID, True)
    # print("Val base DONE.")
    # convert_ignore_json("data/coco/annotations/panoptic_val2017.json", 
    #     "data/coco/annotations/panoptic_val2017_novel.json", novel_clsID, True)
    # print("Val novel DONE.")