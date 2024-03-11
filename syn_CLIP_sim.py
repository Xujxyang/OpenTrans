# pip install git+https://github.com/openai/CLIP.git
import os
import json
import torch
import clip
from PIL import Image

os.environ['DETECTRON2_DATASETS'] = '/opt/tiger/debug/code/ky_open_voca/data'
from detectron2.data import DatasetCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from mask2former.data.datasets.register_coco_panopic_caption import register_coco_panopic_caption, get_metadata

# %load code/ky_open_voca/mask2former/data/datasets/register_coco_panopic_caption.py

coco_train = DatasetCatalog.get("coco_2017_train_panoptic_caption")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
meta = get_metadata()

id2name = {}
for cat in COCO_CATEGORIES:
    id2name[meta["stuff_dataset_id_to_contiguous_id"][cat['id']]] = cat["name"]

with open("coco_name_syn.json") as f:
    coco_cls_syn = json.load(f)

def clip_sim(img, cls_list):
    image = preprocess(img).unsqueeze(0).to(device)
    prompt_list = ["a photo of " + c for c in cls_list]
    text = clip.tokenize(prompt_list).to(device)
    with torch.no_grad():
        # image_features = model.encode_image(image)
        # text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    #return logits_per_image
    return probs
# clip_sim(img.crop([168, 163, 168+310, 163+465]), coco_cls_syn[id2name[0]])

coco_anno_json = []
coco_anno_len = len(coco_train)
for i, c_anno in enumerate(coco_train):
    ori_image = Image.open(c_anno['file_name'])
    ret = {'file_name': c_anno['file_name'], 'image_id': c_anno['image_id'], 'segments_info': []}
    for s_anno in c_anno['segments_info']:
        s_ret = {'id': s_anno['id'], 'ori_category_id' : s_anno['category_id']}
        s_x, s_y, s_w, s_h = s_anno['bbox']
        syn_cls = coco_cls_syn[id2name[s_anno['category_id']]]
        prob = clip_sim(ori_image.crop([s_x, s_y, s_x + s_w, s_y + s_h]), syn_cls)
        s_ret['syn_category'] = {cls_ : float(p) for cls_, p in zip(syn_cls, prob.flatten())}
        ret['segments_info'].append(s_ret)
    coco_anno_json.append(ret)
    print("[{}]/[{}]".format(i, coco_anno_len), end='\r')
with open("coco_anno_syn.json", "w") as f:
    json.dump(coco_anno_json, f)
