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

from ...utils.clip import tokenize
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES, _get_builtin_metadata
from ...modeling.utils.misc import process_coco_cat

import os
import json

__all__ = ["DETRPanopticCaptionSynDatasetMapper"]


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


# This is specifically designed for the COCO dataset.
class DETRPanopticCaptionSynDatasetMapper:
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
    COCO_SYN_CLASS_PATH = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco", 'coco_name_syn_alter.json')

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
        syn_class_super,
        plain_class_super,
        syn_class_type,
        grounding_super,
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
        assert per_region_super, "Make sure set per-region supervision enable as True when use DETRPanopticCaptionSynDatasetMapper"
        assert syn_class_super, "Make sure set syn class supervision enable as True when use DETRPanopticCaptionSynDatasetMapper"

        self.img_format = image_format
        self.is_train = is_train

        self.text_input = text_input
        # self.per_region_super = per_region_super
        self.grounding_super = grounding_super

        self.plain_class_super = plain_class_super
        self.syn_class_type = syn_class_type

        self.id2name, self.name2id = {}, {}
        coco_meta = _get_builtin_metadata("coco_panoptic_standard")
        for coco_log in COCO_CATEGORIES:
            _id, _name = coco_log['id'], coco_log['name']
            if _id in coco_meta["thing_dataset_id_to_contiguous_id"]:
                _id = coco_meta["thing_dataset_id_to_contiguous_id"][_id]
            else:
                _id = coco_meta["stuff_dataset_id_to_contiguous_id"][_id]
            self.id2name[_id] = _name
            self.name2id[_name] = _id

        cls_id = [i for i in range(133)]
        cls_name = [self.id2name[x] for x in cls_id]
        self.cls_name = process_coco_cat(cls_name)
        if self.plain_class_super:
            self.plain_class_token = tokenize(self.cls_name, context_length=77)

        if self.syn_class_type in ['clusterMax', 'clusterAvg']:
            with open(self.COCO_SYN_CLASS_PATH) as f:
                syn_data = json.load(f)
            self.syn_name = []
            self.syn_slice_dic = {}
            self.syn_slice_list = []
            count = 0
            for i, name in enumerate(cls_name):
                index = []
                for s_name in syn_data[name]:
                    self.syn_name.append(s_name)
                    index.append(count)
                    count += 1
                self.syn_slice_dic[i] = np.array(index, dtype=np.int16)
                self.syn_slice_list.append(np.array(index, dtype=np.int16))
            self.syn_class_token = tokenize(self.syn_name, context_length=77)

        self.syn_process_each = (self.syn_class_type == 'thresRand')


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

        text_input = cfg.MODEL.TEXT_ENCODER.INPUT

        ret = {
            "is_train": is_train,
            "crop_gen": crop_gen,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "text_input": text_input,
            "per_region_super": cfg.MODEL.PER_REGION.ENABLED,
            "syn_class_super": cfg.MODEL.PER_REGION.SYN_CLASS.ENABLED,
            "plain_class_super": cfg.MODEL.PER_REGION.PLAIN_CLASS,
            "syn_class_type": cfg.MODEL.PER_REGION.SYN_CLASS.TYPE,
            "grounding_super": cfg.MODEL.GROUNDING.ENABLED,
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

        if self.syn_process_each:
            max_cls_idx = 133
            cls_is_change = np.zeros(max_cls_idx)
            cls_id = [i for i in range(133)]
            cls_name = self.cls_name.copy()

            cls_dic = {}

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
            syn_cls = dataset_dict["syn_class"]

            # apply the same transformation to panoptic segmentation
            # import pdb;pdb.set_trace()
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            syn_classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                syn_prob = syn_cls[segment_info["id"]]
                # randomly choose one, threshold
                if self.syn_process_each:
                    new_class_id = class_id
                    new_class = random.choice([n for n, p in syn_prob.items() if p > 0.1])
                    if cls_name[new_class_id] == new_class:
                        cls_is_change[new_class_id] = 1
                    elif cls_is_change[new_class_id] == 0:
                        cls_name[new_class_id] = new_class
                        cls_is_change[new_class_id] = 1
                    else:
                        is_find = False
                        for find_idx in range(133, max_cls_idx):
                            if cls_name[find_idx] == new_class:
                                new_class_id = find_idx
                                is_find = True
                        if not is_find:
                            cls_name.append(new_class)
                            new_class_id = max_cls_idx
                            max_cls_idx += 1

                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    if self.syn_process_each:
                        syn_classes.append(new_class_id)
                    masks.append(pan_seg_gt == segment_info["id"])

            classes = np.array(classes)
            if self.syn_process_each:
                syn_classes = np.array(syn_classes)
                dataset_dict["syn_classes"] = torch.tensor(syn_classes, dtype=torch.int64)
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

        # import pdb;pdb.set_trace()

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
            noun_token = tokenize(nouns, context_length=77) # (num_world, 77)

            len_noun = len(noun_token)
            token = torch.ones(num_max_world, 77).long()
            token_mask = torch.zeros(num_max_world).long()
            token[:len_noun, :] = noun_token
            token_mask[:len_noun] = 1

            dataset_dict['caption'] = token
            dataset_dict['caption_mask'] = token_mask

        if self.plain_class_super:
            dataset_dict['all_caption'] = self.plain_class_token
        if self.syn_process_each:
            dataset_dict['syn_class'] = tokenize(cls_name, context_length=77)
        else:
            dataset_dict['syn_class'] = self.syn_class_token
            dataset_dict['all_syn_slice'] = self.syn_slice_dic
            dataset_dict['all_syn_slice'] = self.syn_slice_list

        return dataset_dict
