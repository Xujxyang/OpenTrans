# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances

import random
import nltk

from detectron2.data import MetadataCatalog

from ...utils.clip import tokenize
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES, _get_builtin_metadata
from ...modeling.utils.misc import process_coco_cat

import cv2
from panopticapi.utils import rgb2id
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

__all__ = ["DETRPanopticCaptionDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    # if is_train:
    #     min_size = cfg.INPUT.MIN_SIZE_TRAIN
    #     max_size = cfg.INPUT.MAX_SIZE_TRAIN
    #     sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    # else:
    #     min_size = cfg.INPUT.MIN_SIZE_TEST
    #     max_size = cfg.INPUT.MAX_SIZE_TEST
    #     sample_style = "choice"
    # if sample_style == "range":
    #     assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
    #         len(min_size)
    #     )

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeScale(cfg.INPUT.SCALE_RESIZE.SCALE_MIN, cfg.INPUT.SCALE_RESIZE.SCALE_MAX,
            cfg.MODEL.RESNETS.INPUT_RESOLUTION, cfg.MODEL.RESNETS.INPUT_RESOLUTION))
    tfm_gens.append(T.FixedSizeCrop((cfg.MODEL.RESNETS.INPUT_RESOLUTION, cfg.MODEL.RESNETS.INPUT_RESOLUTION)))
    # tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def replace_values(tensor, old_value, new_value):
    mask = tensor == old_value  # 创建一个布尔掩码，将与旧值相等的元素设为True
    tensor[mask] = new_value

def label_to_onehot(label, num_classes):
    # 将label转换为one-hot形式
    onehot = torch.nn.functional.one_hot(label, num_classes)
    return onehot

# This is specifically designed for the COCO dataset.
class DeeplabPanopticCaptionKDDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        crop_gen,
        tfm_gens,
        image_format,
        text_input,
        per_region_super,
        grounding_super,
        prompt,
        dataset_list,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.crop_gen = crop_gen
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[DETRPanopticDatasetMapper] Full TransformGens used in training: {}, crop: {}".format(
                str(self.tfm_gens), str(self.crop_gen)
            )
        )

        self.img_format = image_format
        self.is_train = is_train

        self.text_input = text_input
        self.per_region_super = per_region_super
        self.grounding_super = grounding_super
        self.prompt = prompt

        self.dataset2token = {}
        
        for dataset in dataset_list:
            data_meta = MetadataCatalog.get(dataset)
            if hasattr(data_meta, 'dataset_contiguous_id2name'):
                id2name = data_meta.dataset_contiguous_id2name
            else:
                # id2name = {}
                # assert 'coco' in dataset
                # for coco_log in COCO_CATEGORIES:
                raise ValueError
                # pass

            # for cls_id in range(len(id2name)):
            cls_name = [id2name[x] for x in range(len(id2name))]
            cls_name = process_coco_cat(cls_name)       #也不一定需要过这个
            cls_name = [self.prompt.format(x) for x in cls_name]
            # import pdb;pdb.set_trace()
            all_class_token = tokenize(cls_name, context_length=77)

            # import pdb; pdb.set_trace()

            self.dataset2token[dataset] = all_class_token

        # self.id2name, self.name2id = {}, {}
        # coco_meta = _get_builtin_metadata("coco_panoptic_standard")
        # # vspw_meta = _get_builtin_metadata("coco_panoptic_standard")
        # for coco_log in COCO_CATEGORIES:
        #     _id, _name = coco_log['id'], coco_log['name']       #把metadata里的name和id对应上
        #     if _id in coco_meta["thing_dataset_id_to_contiguous_id"]:
        #         _id = coco_meta["thing_dataset_id_to_contiguous_id"][_id]
        #     else:
        #         _id = coco_meta["stuff_dataset_id_to_contiguous_id"][_id]
        #     self.id2name[_id] = _name
        #     self.name2id[_name] = _id

        # if self.per_region_super:
        #     cls_id = [i for i in range(133)]        #类别数需要改变

        #     cls_name = [self.id2name[x] for x in cls_id]
        #     cls_name = process_coco_cat(cls_name)       #也不一定需要过这个
        #     cls_name = [self.prompt.format(x) for x in cls_name]
        #     # import pdb;pdb.set_trace()
        #     self.all_class_token = tokenize(cls_name, context_length=77)

        self.kd_transform = self._transform((224, 224))


    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if cfg.INPUT.CROP.ENABLED and is_train:
            crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            crop_gen = None

        tfm_gens = build_transform_gen(cfg, is_train)

        text_input = cfg.MODEL.TEXT_ENCODER.INPUT       #这里是caption，如果没有caption的话我们是否还需要？

        ret = {
            "is_train": is_train,
            "crop_gen": crop_gen,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "text_input": text_input,
            "per_region_super": cfg.MODEL.PER_REGION.ENABLED,
            "grounding_super": cfg.MODEL.GROUNDING.ENABLED,
            "prompt": cfg.MODEL.TEXT_ENCODER.PROMPT,
            "dataset_list": cfg.DATASETS.TRAIN,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # import pdb;pdb.set_trace()

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict["pan_seg_file_name"], "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            # import pdb;pdb.set_trace()
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            pan_seg_gt = rgb2id(pan_seg_gt)

            pan_seg_gt_onehot = pan_seg_gt

            instances = Instances(image_shape)
            classes = []
            masks = []
            # 在这处理data
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                # import pdb; pdb.set_trace()
                replace_values(pan_seg_gt_onehot, segment_info["id"], class_id)
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])
            
            # class_id 需要连续
            replace_values(pan_seg_gt_onehot, 0, 133)
            replace_values(pan_seg_gt_onehot, 16777215, 133)

            class_count = 134
            onehot = torch.from_numpy(pan_seg_gt_onehot)
            dataset_dict["gt_label"] = onehot
            onehot = label_to_onehot(onehot.long(), class_count)

            dataset_dict["onehot"] = onehot

            # import pdb; pdb.set_trace()

            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        # kd crop
        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
        # import pdb;pdb.set_trace()
        segments_info = dataset_dict["segments_info"]
        pan_seg_gt = rgb2id(pan_seg_gt)
        ins_mask_list = []
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                ins_mask_list.append(pan_seg_gt == segment_info["id"])
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)


        kd_image_list = []
        for ins_mask in ins_mask_list:
            h_idx, w_idx = ins_mask.nonzero()
            h_min, h_max = h_idx.min(), h_idx.max()
            w_min, w_max = w_idx.min(), w_idx.max()
            img = copy.deepcopy(image)
            # import pdb;pdb.set_trace()
            # img = img * ins_mask[:, :, np.newaxis]
            # img = img[h_min:h_max, w_min:w_max, :]
            img = Image.fromarray(img)
            ins_pil = Image.fromarray(~ins_mask)
            img.paste((128, 128, 128), mask=ins_pil)
            img = img.crop([w_min, h_min, w_max, h_max])
            img = self.kd_transform(img)
            kd_image_list.append(img)

        # import pdb;pdb.set_trace()
        if len(kd_image_list) > 0:
            dataset_dict['kd_image'] = torch.stack(kd_image_list)
        else:
            dataset_dict['kd_image'] = None

        if self.grounding_super:
            # num_caption = len(dataset_dict["caption"])
            cap = random.choice(dataset_dict["caption"])
            cap = cap['caption']
            tokens = nltk.word_tokenize(cap)
            tags = nltk.pos_tag(tokens)
            nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
            # nouns = ' '.join(nouns)

            # random drop by 25%
            len_noun = len(nouns)
            prob = np.random.rand(len_noun)
            if np.sum(prob > 0.25) == 0:
                prob = np.ones(len_noun)
            nouns = [n for n, p in zip(nouns, prob) if p > 0.25]

            num_max_world = 5
            if len(nouns) > num_max_world:
                nouns = random.sample(nouns, num_max_world)

            nouns = process_coco_cat(nouns)
            nouns = [self.prompt.format(x) for x in nouns]
            noun_token = tokenize(nouns, context_length=77) # (num_world, 77)

            len_noun = len(noun_token)
            token = torch.ones(num_max_world, 77).long()
            token_mask = torch.zeros(num_max_world).long()
            token[:len_noun, :] = noun_token
            token_mask[:len_noun] = 1

            dataset_dict['caption'] = token
            dataset_dict['caption_mask'] = token_mask

        if self.per_region_super:
            dataset_dict['all_caption'] = self.dataset2token[dataset_dict.pop('dataset')]
        # import pdb;pdb.set_trace()

        return dataset_dict


    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])