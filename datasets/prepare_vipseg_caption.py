import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip import blip_decoder
import utils
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result

import glob
import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from detectron2.data import detection_utils
from panopticapi.utils import rgb2id

import math
import imageio

def get_bbox(h, w, left, top, right, bottom, ratio=1.5):
    assert left <= right, "right less than left"
    assert top <= bottom, "bottom less than top"

    # wid_off = (right - left) * (ratio - 1.)
    # hei_off = (bottom - top) * (ratio - 1.)

    wid_off = hei_off = (right - left + bottom - top) * (ratio - 1.) / 2

    # wid_off = hei_off = math.sqrt((right - left) * (bottom - top) * (ratio - 1.))

    return (
        max(0, left - wid_off),
        max(0, top - hei_off),
        min(w, right + wid_off),
        min(h, bottom + hei_off)
    )

def load_demo_image(path, bbox, image_size, device):
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(path).convert('RGB').crop(bbox)

    # w,h = raw_image.size
    # display(raw_image.resize((w//5,h//5)))
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image, raw_image


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           prompt=config['prompt'])

    model = model.to(device)   
    
    vip_path = "../ky_open_voca/data/VIPSeg/VIPSeg_720P"
    with open(os.path.join(vip_path, "panoptic_gt_VIPSeg_train.json")) as f:
        vip = json.load(f)

    cat = vip["categories"]
    id2thing = {}
    for c in cat:
        id2thing[c["id"]] = c

    image_path = os.path.join(vip_path, "images")
    pano_path = os.path.join(vip_path, "panomasksRGB")

    debug_count, debug_info, is_debug = 0, {}, False
    # is_debug = True
    vip_ins_cap = {}
    if is_debug:
        vip["annotations"] = vip["annotations"][:10]
    for v_anno in tqdm.tqdm(vip["annotations"]):
        v_id = v_anno["video_id"]
        # ins_set = v_anno["ins_id_set"]
        ins_set = set()
        v_len = len(v_anno["annotations"])

        # ins_area = {iid : [0] * v_len for iid in ins_set}
        ins_area, iid2seg = {}, {}
        # find max area frame for every instance
        for f_i, f_anno in enumerate(v_anno["annotations"]):
            for seg in f_anno["segments_info"]:
                if seg["instance_id"] not in ins_area:
                    ins_area[seg["instance_id"]] = [0] * v_len
                ins_area[seg["instance_id"]][f_i] = seg["area"]
                iid2seg[seg["instance_id"]] = seg
                ins_set.add(seg['instance_id'])

        ins_set = list(ins_set)
        v_ins_cap = {}
        for iid in ins_set:
            # remove stuff
            if iid < 125:
                continue
            max_fid = np.argmax(ins_area[iid])
            max_anno = v_anno["annotations"][max_fid]
            fid = max_anno["image_id"]

            pan_seg_gt = detection_utils.read_image(os.path.join(pano_path, v_id, f"{fid}.png"), "RGB")
            pan_seg_gt = rgb2id(pan_seg_gt)
            ins_mask = np.zeros_like(pan_seg_gt, dtype=np.uint8)
            ins_mask[pan_seg_gt == iid2seg[iid]["id"]] = 255

            img_x, img_y = ins_mask.shape

            # remove small object 
            if ins_area[iid][max_fid] / (img_x * img_y) < 0.001:
                continue

            inds_y, inds_x = np.where(ins_mask > 128)
            left, right = np.min(inds_x), np.max(inds_x)
            top, bottom = np.min(inds_y), np.max(inds_y)

            ratio_bbox = get_bbox(img_x, img_y, left, top, right, bottom, ratio=1.3)

            crop_img, raw_image = load_demo_image(
                os.path.join(image_path, v_id, f"{fid}.jpg"),
                ratio_bbox,
                config['image_size'],
                device
            )
            caption_beam = model.generate(crop_img, sample=False, num_beams=3, max_length=20, min_length=5, num_return_sequences=2) 
            caption_nucleus = model.generate(crop_img, sample=True, top_p=0.9, max_length=20, min_length=5, num_return_sequences=2) 

            ins_caption = caption_beam + caption_nucleus

            # remove captions that identify not-person objs as person
            # "xxxxx next to people" is acceptable
            person_word = ["person", "people", "woman", "man"]
            if id2thing[iid2seg[iid]['category_id']]['name'] != "person":
                ins_caption = [i_c for i_c in ins_caption if not any([pw in i_c[:30] for pw in person_word])]

            # if debug_count >= 5:
            #     import pdb;pdb.set_trace()

            # all person ???
            if len(ins_caption) == 0:
                continue

            v_ins_cap[iid] = ins_caption

            if is_debug:
                debug_name = "0" * (5 - len(str(debug_count))) + str(debug_count)
                ins_cat = id2thing[iid2seg[iid]['category_id']]
                raw_image.save("tmp/{}_{}_{}.jpg".format(debug_name, ins_cat['name'], ins_area[iid][max_fid] / (img_x * img_y)))
                imageio.imwrite("tmp/{}_{}_{}.png".format(debug_name, ins_cat['name'], ins_cat['isthing']), ins_mask)

                debug_info[debug_count] = caption_beam + caption_nucleus
                debug_count += 1

        vip_ins_cap[v_id] = v_ins_cap

    with open("vip_ins_cap.json", "w") as f:
        json.dump(vip_ins_cap, f)

    if is_debug:
        with open("tmp/debug_info.json", "w") as f:
            json.dump(debug_info, f)

    # #### Inference ####
    # for path in tqdm.tqdm(args.input):
    #     image = load_demo_image(path, config['image_size'], device)
    #     # beam search
    #     caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5, num_return_sequences=3) 
    #     # nucleus sampling
    #     # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5, num_return_sequences=3) 
    #     import pdb;pdb.set_trace()
    #     print('caption: '+caption[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/nocaps.yaml')
    parser.add_argument('--output_dir', default='caption')        
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    args = parser.parse_args()
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)