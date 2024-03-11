import os
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
import tqdm
from PIL import Image
import scipy.io

if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    
    pc_dir = os.path.join(dataset_dir, "VOCdevkit", "VOC2010")
    val_list_dir = os.path.join(pc_dir, "ImageSets", "SegmentationContext", "val.txt")
    dest_dir = os.path.join(pc_dir, "SegmentationClassContext_459")
    os.makedirs(dest_dir, exist_ok=True)

    with open(val_list_dir) as f:
        val = f.readlines()
        val = [v.strip() for v in val]

    for v in val:
        m = scipy.io.loadmat(os.path.join(pc_dir, "trainval", "{}.mat".format(v)))
        cv2.imwrite(os.path.join(dest_dir, "{}.png".format(v)), m["LabelMap"] - 1)