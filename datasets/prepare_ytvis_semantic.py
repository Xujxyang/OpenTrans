import os
import tqdm
import json
import imageio
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util

from mask2former_video.data_video.datasets.ytvis_api.ytvos import YTVOS
from evaluation import jaccard

# with open('data/train/meta_expressions.json') as f:
#     ref = json.load(f)

ytvis_api = YTVOS('./data/opt/tiger/ytvis/annotations/instances_val_sub_GT.json')
vid_ids = sorted(ytvis_api.vids.keys())
vids = ytvis_api.loadVids(vid_ids)
anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]

# vis_ref = {"videos": {}}
for v_ann in tqdm.tqdm(anns):
    vid = v_ann[0]['video_id']
    v_info = ytvis_api.vids[vid]

    v_name = v_info['file_names'][0].split('/')[0]
    h, w = v_info['height'], v_info['width']

    for o_ann in v_ann:
        iid = o_ann['id']
        ar = np.array(o_ann['areas'])
        ar[ar == None] = -1
        # import pdb; pdb.set_trace()
        idx = np.argmax(ar)
        rle = mask_util.frPyObjects(o_ann['segmentations'][idx], h, w)
        mask = mask_util.decode(rle)

        frame = v_info['file_names'][idx]

        max_iou, max_oid = -1, -1
        max_mask = None
        for oid in vos_oids:
            if oid == 0:
                continue

            o_mask = np.zeros_like(vos_ann)
            o_mask[vos_ann == oid] = 1
            iou = jaccard(mask, o_mask)
            if max_iou < iou:
                max_iou, max_oid = iou, oid
                max_mask = o_mask

