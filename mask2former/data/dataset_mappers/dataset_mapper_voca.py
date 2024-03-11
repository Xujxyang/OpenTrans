# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

# from . import detection_utils as utils
# from . import transforms as T

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES, CITYSCAPES_CATEGORIES, _get_builtin_metadata, ADE20K_SEM_SEG_CATEGORIES
# from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
# import nltk

from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Instances
import torch.nn.functional as F

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


from ...utils.clip import tokenize
from ...modeling.utils.misc import process_coco_cat

from ..datasets.register_vspw import VSPW_CATEGORIES
from ..datasets.register_pascal_context import PC_CATEGORIES, PC_FULL_CATEGORIES
from ..datasets.register_ade20k_full import ADE20K_SEM_SEG_FULL_CATEGORIES
from ..datasets.register_pascal_voc_20_semantic import PASCAL_VOC_20_CATEGORIES

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapperVoca:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        per_region_super: bool = False,
        grounding_super: bool = False,
        prompt: str = '{}',
        dataset_list: list = [],
        full_name_prompt:bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.per_region_super       = per_region_super
        self.grounding_super        = grounding_super
        self.prompt                 = prompt
        self.full_name_prompt       = full_name_prompt
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        # for dataset in dataset_list:
        #     data_meta = MetadataCatalog.get(dataset)
        self.dataset2name = {}
        self.dataset2namelist = {}
        self.dataset2another_split = {}
        for dataset in [
            'coco_2017_val_panoptic_caption_base', 
            'coco_2017_val_panoptic_caption_novel',
            'coco_2017_val_panoptic_caption',
            'vipseg_val_base_image',
            'vipseg_val_novel_image',
            'vipseg_val_image',
        ]:
            data_meta = MetadataCatalog.get(dataset)
            id2name = data_meta.dataset_contiguous_id2name

            cls_name = [id2name[x] for x in range(len(id2name))]
            cls_name = process_coco_cat(cls_name)       #也不一定需要过这个
            cls_name = [self.prompt.format(x) for x in cls_name]
            # import pdb;pdb.set_trace()
            # all_class_token = tokenize(cls_name, context_length=77)
            self.dataset2name[dataset] = cls_name

            if hasattr(data_meta, 'dataset_contiguous_id2namelist'):
                id2namelist = data_meta.dataset_contiguous_id2namelist
                name_i = [(name, i) for i in range(len(id2namelist)) for name in id2namelist[i]]
                cls_index = [idx for _, idx in name_i]
                cls_name = [name for name, _ in name_i]
                cls_name = process_coco_cat(cls_name)
                cls_name = [self.prompt.format(x) for x in cls_name]

                self.dataset2namelist[dataset] = {
                    'name' : cls_name,
                    'index' : cls_index
                }
            if hasattr(data_meta, 'another_split_classname'):
                cls_name = data_meta.another_split_classname
                cls_name = process_coco_cat(cls_name)
                self.dataset2another_split[dataset] = [self.prompt.format(x) for x in cls_name]

        # COCO cls id-name dict
        self.coco_id2name, self.name2id = {}, {}
        self.coco_cat_num = 133
        coco_meta = _get_builtin_metadata("coco_panoptic_standard")
        for coco_log in COCO_CATEGORIES:
            _id, _name, is_thing = coco_log['id'], coco_log['name'], coco_log['isthing']
            if _id in coco_meta["thing_dataset_id_to_contiguous_id"]:
                _id = coco_meta["thing_dataset_id_to_contiguous_id"][_id]
            else:
                _id = coco_meta["stuff_dataset_id_to_contiguous_id"][_id]
            _name = process_coco_cat(_name)[0]
            self.coco_id2name[_id] = _name
            # self.name2id[_name] = _id

        # meta = {}
        # thing_dataset_id_to_contiguous_id = {}
        # stuff_dataset_id_to_contiguous_id = {}
        # for k in CITYSCAPES_CATEGORIES:
        #     if k["isthing"] == 1:
        #         thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        #     else:
        #         stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

        # meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
        # meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

        self.city_id2name = {}
        self.city_cat_num = 19
        for k in CITYSCAPES_CATEGORIES:
            self.city_id2name[k['trainId']] = k['name']

        self.ori_image_trans = Compose([
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.ade_id2name = {}
        for i, k in enumerate(ADE20K_SEM_SEG_CATEGORIES):
            self.ade_id2name[i] = k
        self.ade_cat_num = len(ADE20K_SEM_SEG_CATEGORIES)

        self.vspw_id2name = {}
        for k in VSPW_CATEGORIES:
            if k["id"] == 0:
                continue
            # TODO multi-label  e.g. "cupboard_or_showcase_or_storage_rack"
            name = k["name"].replace("_", " ")
            # VSPW ignore 'other' class when eval
            self.vspw_id2name[k["id"] - 1] = name
        self.vspw_cat_num = len(VSPW_CATEGORIES) - 1

        self.pc_id2name = {}
        for k in PC_CATEGORIES:
            if k["id"] == 0:
                continue
            self.pc_id2name[k["id"] - 1] = k["name"]
        self.pc_cat_num = len(PC_CATEGORIES) - 1
        
        self.pc_full_id2name = {}
        for k in PC_FULL_CATEGORIES:
            self.pc_full_id2name[k["id"] - 1] = k["name"]
        self.pc_full_cat_num = len(PC_FULL_CATEGORIES)

        self.ade_full_id2name = {}
        for k in ADE20K_SEM_SEG_FULL_CATEGORIES:
            self.ade_full_id2name[k["trainId"]] = k["name"]
        self.ade_full_cat_num = len(ADE20K_SEM_SEG_FULL_CATEGORIES)

        self.pc_20_id2name = {}
        # import pdb; pdb.set_trace()
        for k in PASCAL_VOC_20_CATEGORIES:
            self.pc_20_id2name[k["id"] - 1] = k["name"]
        self.pc_num = len(PASCAL_VOC_20_CATEGORIES)
        # import pdb;pdb.set_trace()

        # if self.per_region_super:
        #     pass

        # self.all_ins_count = 0
        # self.small_ins_count = 0
        # self.vspw_cls_frame_count = np.zeros(self.vspw_cat_num)
        # self.vspw_cls_video_count = np.zeros(self.vspw_cat_num)
        # self.vspw_cls2video = [set() for i in range(self.vspw_cat_num)]


    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "per_region_super": cfg.MODEL.PER_REGION.ENABLED,
            "grounding_super": cfg.MODEL.GROUNDING.ENABLED,
            "prompt": cfg.MODEL.TEXT_ENCODER.PROMPT,
            "dataset_list": cfg.DATASETS.TEST,
            "full_name_prompt": cfg.MODEL.MASK_FORMER.TEST.VOCA_FULL_CLASS,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        ori_image = self.ori_image_trans(image).unsqueeze(0)
        dataset_dict["ori_image"] = ori_image

        is_vspw = False
        if 'dataset' not in dataset_dict:
            # import pdb;pdb.set_trace()
            file_path = dataset_dict["sem_seg_file_name"]
            if 'ADE20K_2021_17_01' in file_path:
                cat_num, id2name = self.ade_full_cat_num, self.ade_full_id2name
            elif 'ADEChallengeData2016' in file_path:
                cat_num, id2name = self.ade_cat_num, self.ade_id2name
            elif 'SegmentationClassContext_459' in file_path:
                cat_num, id2name = self.pc_full_cat_num, self.pc_full_id2name
            elif 'SegmentationClassContext' in file_path:
                cat_num, id2name = self.pc_cat_num, self.pc_id2name
            elif 'pascal_voc' in file_path:
                cat_num, id2name = self.pc_num, self.pc_20_id2name
            elif 'cityscapes' in file_path:
                cat_num, id2name = self.city_cat_num, self.city_id2name
            elif 'VSPW' in file_path:
                is_vspw = True
                cat_num, id2name = self.vspw_cat_num, self.vspw_id2name
            elif 'coco' in file_path:
                cat_num, id2name = self.coco_cat_num, self.coco_id2name
            else:
                raise NotImplementedError

            # cls_id = [x["category_id"] for x in dataset_dict["segments_info"]]
            # cls_id = np.unique(cls_id)
            cls_id = [i for i in range(cat_num)]

            cls_name = [id2name[x] for x in cls_id]
            cls_name = [self.prompt.format(x) for x in cls_name]
            # cls_name = process_coco_cat(cls_name)
        else:
            dataset = dataset_dict.pop('dataset')
            # if some config TODO
            if False:
            # if dataset in self.dataset2namelist:
                cls_name = self.dataset2namelist[dataset]['name']
                cls_id = self.dataset2namelist[dataset]['index']
            else:
                cls_name = copy.deepcopy(self.dataset2name[dataset])
                cls_id = [i for i in range(len(cls_name))]
            cat_num = len(np.unique(cls_id))
            if self.full_name_prompt:
                for name in self.dataset2another_split[dataset]:
                    cls_name.append(name)
                    cls_id.append(cat_num)
        
        # import pdb;pdb.set_trace()

        name_token = tokenize(cls_name, context_length=77) # (num_world, 77)
        dataset_dict['cls_name'] = cls_name
        # if self.per_region_super:
        dataset_dict["all_caption"] = torch.tensor(name_token)
        # else:
        dataset_dict["caption"] = torch.tensor(name_token)
        dataset_dict["caption_mask"] = torch.ones(len(name_token)).long()
        dataset_dict["name2id"] = torch.tensor(cls_id)
        dataset_dict["num_class"] = cat_num

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
            if is_vspw:
                # import pdb;pdb.set_trace()
                sem_seg_gt[sem_seg_gt == 0] = 255
                sem_seg_gt = sem_seg_gt - 1
                sem_seg_gt[sem_seg_gt == 254] = 255
        else:
            sem_seg_gt = None

        ##### debug, save per-class mask
        # import pdb;pdb.set_trace()
        # import os
        # import imageio
        # # debug use, save class mask
        # # if not hasattr(self, 'vsps_cls_count'):
        # #     self.vsps_cls_count = np.zeros(cat_num)
        # self.debug_ = False
        # white_list = [37, 97, 14]
        # for gt_idx in unique(sem_seg_gt):
        #     if gt_idx == 255:
        #         continue
        #     if gt_idx >= cat_num:
        #         print(file_path, gt_idx)
        #         continue
        #     self.vspw_cls_frame_count[gt_idx] += 1

        #     # if self.debug_:
        #     #     import pdb;pdb.set_trace()
        #     # get video name
        #     video_name = os.path.basename(file_path).split("_")[0]

        #     folder_name = "vspw_vis"
        #     if video_name not in self.vspw_cls2video[gt_idx]:
        #     # if video_name not in self.vspw_cls2video[gt_idx] and gt_idx in white_list:
        #         self.vspw_cls2video[gt_idx].add(video_name)
        #         self.vspw_cls_video_count[gt_idx] += 1

        #     # if self.vspw_cls_frame_count[gt_idx] <= 25 and gt_idx in white_list:
        #     # if self.vspw_cls_frame_count[gt_idx] <= 5:
        #         if self.vspw_cls_video_count[gt_idx] <= 5:
        #             # if not save_image:
        #             imageio.imwrite("./tmp/{}/cat_{}_{}.jpg".format(folder_name, id2name[gt_idx], os.path.basename(file_path)), image)
        #             mask_ = np.zeros_like(sem_seg_gt)
        #             mask_[sem_seg_gt == gt_idx] = 255
        #             imageio.imwrite("./tmp/{}/cat_{}_{}.png".format(folder_name, id2name[gt_idx], os.path.basename(file_path)), mask_)

        # # np.save("vspw_cls_frame_count.npy", self.vspw_cls_frame_count)
        # # np.save("vspw_cls_video_count.npy", self.vspw_cls_video_count)
        # if (self.vspw_cls_frame_count >= 5).sum() == len(self.vspw_cls_frame_count):
        # # if (self.vspw_cls_frame_count >= 5).sum() == len(white_list):
        #     mission is DONE
        #---------------------------

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        #### De-comment code below to contain 'gt instance' in forward, debug use
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            # import pdb;pdb.set_trace()
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])

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

        # ins_mask = instances._fields['gt_masks'].float()
        # ins_mask = F.interpolate(ins_mask.unsqueeze(0), size=(16, 16), mode='nearest').squeeze(0)
        # ins_sum = ins_mask.sum(dim=(-1, -2))
        # self.all_ins_count += len(ins_sum)
        # self.small_ins_count += (ins_mask.sum(dim=(-1, -2)) == 0).sum().item()

        # print("{}/{}, {}".format(self.small_ins_count, self.all_ins_count, self.small_ins_count / self.all_ins_count))
        
        # import pdb;pdb.set_trace()

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
