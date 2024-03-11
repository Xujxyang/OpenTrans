# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
# from numpy.lib.utils import get_include

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

# from torch.nn.modules import faster_transformer, loss
from torch.nn.modules import loss

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

from .clip_models import CLIPTextEncoder
from .modeling.utils.misc import mask_avg_pool, get_sup_propotype
from .modeling.utils.loss import GroundingHead

import torch.distributed as dist

# from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, COCO_CATEGORIES
# from pytorch_memlab import LineProfiler, profile, profile_every
# from torch.autograd import gradcheck

from itertools import accumulate

import numpy as np
from .kd_clip import CLIPResNetWithAttention

@META_ARCH_REGISTRY.register()
class OpenSegMaskFormerTextKDSyn(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        vision_clip: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        text_encoder: nn.Module,
        grounding_head: nn.Module,
        grounding_super: bool,
        region_supervision: bool,
        learned_temp: bool,
        region_dropout: float,
        plain_class_super: bool,
        syn_class_type: str,
        vision_feature: str,
        embed_dim: int,
        no_object_weight: float,
        freeze_backbone: bool,
        embed_proj: bool,
        kd_stu_source: str,
        kd_tec_source: str,
        max_kd_len: int,
        fast_kd: bool,
        additional_text_kd: bool,
        grounding_gather: bool,
        syn_super: bool,
        kd_super: bool,
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        # for name, param in self.vision_clip.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # for name, param in self.backbone.named_parameters():
        #     if param.requires_grad:
        #         print('aaaaaaa')
        # exit()
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.text_encoder = text_encoder
        self.grounding_head = grounding_head if grounding_super else None
        self.grounding_super = grounding_super
        self.region_dropout = region_dropout
        self.vision_feature = vision_feature
        self.embed_dim = embed_dim
        self.grounding_gather = grounding_gather
        self.plain_class_super = plain_class_super
        self.syn_class_type = syn_class_type
        self.syn_process_each = (self.syn_class_type == 'thresRand')
        self.syn_super = syn_super
        self.kd_super = kd_super
        assert 0 <= self.region_dropout < 1

        if hasattr(self.backbone, 'init_weights'):
            self.backbone.init_weights()
        self.text_encoder.init_weights()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.region_supervision = region_supervision
        if self.region_supervision:
            self.non_object_embedding = nn.Parameter(
                torch.empty(1, self.text_encoder.text_projection.shape[-1])
            )
            nn.init.normal_(
                self.non_object_embedding.data,
                std=self.text_encoder.transformer.width ** -0.5,
            )

        empty_weight = torch.ones(1) * no_object_weight
        self.register_buffer("empty_weight", empty_weight)

        self.learned_temp = learned_temp
        if self.learned_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.vision_feature != "resAtten" and hasattr(self.backbone, 'remove_attenpool'):
            self.backbone.remove_attenpool()
        
        attn_shape = self.backbone.output_shape()['attn'][0] if 'attn' in self.backbone.output_shape() else 0
        vision_in_dim = {
            "resAtten": attn_shape,
            "res5": self.backbone.output_shape()['res5'][0],
            "pixelDecoder": self.sem_seg_head.pixel_decoder.mask_dim,
            "queryClass": self.sem_seg_head.predictor.hidden_dim,
        }
        # if self.vision_feature != "query-class":
        in_dim = vision_in_dim[self.vision_feature]
        if self.vision_feature == 'resAtten' and not embed_proj:
            self.embed_proj = None
        else:
            self.embed_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_dim, embed_dim)
            )

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # KD
        if self.kd_super:
            self.vision_clip = vision_clip
            # if hasattr(self.vision_clip, 'init_weights'):
            self.vision_clip.init_weights()
            for param in self.vision_clip.parameters():
                param.requires_grad = False
            self.kd_stu_source = kd_stu_source
            self.kd_tec_source = kd_tec_source

            assert self.kd_tec_source in ['res5', 'attn_g', 'attn_l']
            if self.kd_tec_source == 'res5':
                tec_dim = vision_in_dim['res5']
            else:
                tec_dim = vision_in_dim['resAtten']
            # import pdb; pdb.set_trace()
            assert self.kd_stu_source in ['prior_relu', 'posterior_relu', 'prior', 'posterior', 'direct']
            if self.kd_stu_source == 'prior_relu':
                self.kd_proj = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(in_dim, tec_dim)
                )
            elif self.kd_stu_source == 'posterior_relu':
                self.kd_proj = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(embed_dim, tec_dim)
                )
            elif self.kd_stu_source == 'prior':
                self.kd_proj = nn.Linear(in_dim, tec_dim)
            elif self.kd_stu_source == 'posterior':
                self.kd_proj = nn.Linear(embed_dim, tec_dim)
            else:
                self.kd_proj = None

            self.max_kd_len = max_kd_len
            self.fast_kd = fast_kd
            self.text_kd_logit_scale = nn.Parameter(torch.ones([]))
            self.additional_text_kd = additional_text_kd
            

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_out_shape = backbone.output_shape()
        if 'attn' in backbone_out_shape:
            backbone_out_shape.pop('attn')
        sem_seg_head = build_sem_seg_head(cfg, backbone_out_shape)

        text_encoder = CLIPTextEncoder(cfg)

        vision_clip = CLIPResNetWithAttention(cfg)

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        grounding_weight = cfg.MODEL.GROUNDING.LOSS_WEIGHT
        # kd_weight = cfg.MODEL.KD.LOSS_WEIGHT

        # building criterion
        region_supervision = cfg.MODEL.PER_REGION.ENABLED
        cost_class = 2 if cfg.MODEL.PER_REGION.PLAIN_CLASS else -1
        matcher = HungarianMatcher(
            cost_class=cost_class,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight}
        # if region_supervision:
        #     weight_dict['loss_ce'] = class_weight
        if cfg.MODEL.PER_REGION.PLAIN_CLASS:
            weight_dict['loss_ce'] = cfg.MODEL.PER_REGION.PLAIN_LOSS_WEIGHT
        if cfg.MODEL.PER_REGION.SYN_CLASS.ENABLED:
            weight_dict['loss_syn'] = cfg.MODEL.PER_REGION.SYN_CLASS.LOSS_WEIGHT
        if cfg.MODEL.GROUNDING.ENABLED:
            weight_dict['loss_grounding'] = grounding_weight
        if cfg.MODEL.KD.ENABLED:
            weight_dict['loss_kd'] = cfg.MODEL.KD.KD_WEIGHT
            if cfg.MODEL.KD.TEXT_KD_WEIGHT > 0:
                weight_dict['loss_text_kd'] = cfg.MODEL.KD.TEXT_KD_WEIGHT

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items() if k in ['loss_mask', 'loss_dice']})
            weight_dict.update(aux_weight_dict)

        losses = ["masks"]
        # if region_supervision:
        #     losses.append("labels")

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        grounding_head = GroundingHead(cfg)

        return {
            "backbone": backbone,
            "vision_clip": vision_clip,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "text_encoder" : text_encoder,
            "grounding_head": grounding_head,
            "grounding_super": cfg.MODEL.GROUNDING.ENABLED,
            "region_supervision": region_supervision,
            "learned_temp": cfg.MODEL.LEARNED_TEMP,
            "region_dropout": cfg.MODEL.PER_REGION.DROPOUT,
            "plain_class_super": cfg.MODEL.PER_REGION.PLAIN_CLASS,
            "syn_class_type": cfg.MODEL.PER_REGION.SYN_CLASS.TYPE,
            "vision_feature": cfg.MODEL.VISION_FEATURE,
            "embed_dim": cfg.MODEL.TEXT_ENCODER.EMBED_DIM,
            "no_object_weight": no_object_weight,
            "freeze_backbone": cfg.MODEL.FREEZE_BACKBONE,
            "embed_proj": cfg.MODEL.PROJECTION,
            "kd_stu_source": cfg.MODEL.KD.STU_SOURCE,
            "kd_tec_source": cfg.MODEL.KD.TEC_SOURCE,
            "max_kd_len": cfg.MODEL.KD.MAX_LIST_LEN,
            "fast_kd": cfg.MODEL.KD.FAST_KD,
            "additional_text_kd": cfg.MODEL.KD.TEXT_KD_WEIGHT > 0,
            "grounding_gather": cfg.MODEL.GROUNDING.GATHER,
            "syn_super": cfg.MODEL.PER_REGION.SYN_CLASS.ENABLED,
            "kd_super": cfg.MODEL.KD.ENABLED,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }


    @property
    def device(self):
        return self.pixel_mean.device

    # @profile_every(1)
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        # images.tensor: (B,C,H,W)

        outputs, seg_feature = self.sem_seg_head(features)

        if self.grounding_super:        
            text = torch.stack([x["caption"].to(self.device) for x in batched_inputs])
            text_mask = torch.stack([x["caption_mask"].to(self.device) for x in batched_inputs])
            text_ebmd = self.text_encoder(text)
        if self.plain_class_super:
            plain_class = torch.stack([x["all_caption"].to(self.device) for x in batched_inputs])
            plain_class_ebmd = self.text_encoder(plain_class)
        if self.syn_process_each:
            class_caption = [x["syn_class"] for x in batched_inputs]
            _, c_dim = class_caption[0].shape
            cap_len_list = [len(c_i) for c_i in class_caption]
            self.max_cap_num = max(cap_len_list)

            all_class = torch.zeros((len(class_caption), self.max_cap_num, c_dim), dtype=class_caption[0].dtype)
            for i, cap in enumerate(class_caption):
                all_class[i, :cap_len_list[i]] = cap
            all_class = all_class.to(self.device)

            syn_class_ebmd = self.text_encoder(all_class)
        else:
            syn_class = torch.stack([x["syn_class"].to(self.device) for x in batched_inputs])
            syn_class_ebmd = self.text_encoder(syn_class)
            syn_class_slice = [torch.LongTensor(slc).to(self.device) for slc in batched_inputs[0]['all_syn_slice']]


        if self.learned_temp:
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = 10

        if self.vision_feature == 'resAtten':
            vision_feature = features['attn']
        elif self.vision_feature == 'res5':
            vision_feature = features['res5']
        elif self.vision_feature == 'pixelDecoder':
            vision_feature = seg_feature['pixel_feature']
            vision_feature = F.interpolate(vision_feature, size=(48, 48), mode='bilinear')
        elif self.vision_feature == 'queryClass':
            vision_feature = seg_feature['query_embed'] #b, 100, c
        else:
            raise NotImplementedError

        # print(features['attn'].shape, features['res5'].shape, seg_feature['pixel_feature'].shape)#, seg_feature['query_embed'].shape)
        # import pdb;pdb.set_trace()

        bsz, channel, *_ = vision_feature.shape

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None


            # if self.region_supervision:
                # import pdb;pdb.set_trace()
            if self.vision_feature != "queryClass":
                # vision_ebmd, valid_mask = mask_avg_pool(vision_feature, outputs["pred_masks"], use_gt=False, return_mask=True)
                vision_ebmd, valid_mask = get_sup_propotype(vision_feature, outputs["pred_masks"], use_gt=False, return_mask=True)
            else:
                vision_ebmd = vision_feature
                valid_mask = torch.ones((bsz, channel))
            if self.embed_proj is not None:
                vision_ebmd = self.embed_proj(vision_ebmd)
            vision_ebmd_norm = vision_ebmd / (vision_ebmd.norm(dim=-1, keepdim=True) + 1e-7)

            if self.plain_class_super:
                # vision_ebmd, valid_mask = mask_avg_pool(features['attn'], outputs["pred_masks"], use_gt=False, return_mask=True)
                # vision (bsz, 100, dim_c), text (bsz, 133, dim_c)
                # similarity (bsz, 100, 133)
                plain_class_ebmd = torch.cat([plain_class_ebmd,
                    self.non_object_embedding[None, :, :].repeat(len(batched_inputs), 1, 1)], dim=1)
                # import pdb; pdb.set_trace()
                plain_class_ebmd = plain_class_ebmd / (plain_class_ebmd.norm(dim=-1, keepdim=True) + 1e-7)
                plain_class_ebmd = plain_class_ebmd.permute(0, 2, 1)
                cls_pred = torch.bmm(vision_ebmd_norm * logit_scale, plain_class_ebmd)
                # cls_pred = torch.einsum('bvc,btc->bvt', vision_ebmd * logit_scale, all_class_ebmd)
                # cls_pred = similarity * 10 # temperature
                # import pdb;pdb.set_trace()
                outputs['pred_logits'] = cls_pred

            # bipartite matching-based loss
            losses, indices = self.criterion(outputs, targets, get_indice=True)

            if self.plain_class_super:
                cls_loss = self.loss_labels(cls_pred, targets, indices, valid_mask)
                plain_loss = {"loss_ce": cls_loss}

                losses.update(plain_loss)

            if self.syn_super:
                syn_class_ebmd = torch.cat([syn_class_ebmd,
                    self.non_object_embedding[None, :, :].repeat(len(batched_inputs), 1, 1)], dim=1)

                syn_class_ebmd = syn_class_ebmd / (syn_class_ebmd.norm(dim=-1, keepdim=True) + 1e-7)
                syn_class_ebmd = syn_class_ebmd.permute(0, 2, 1)
                cls_pred = torch.bmm(vision_ebmd_norm * logit_scale, syn_class_ebmd)
                if self.syn_process_each:
                    # TODO get gt
                    syn_classes = [x["syn_classes"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets_syn(gt_instances, images, syn_classes)

                    cls_loss = self.loss_labels_process_each(cls_pred, targets, indices, valid_mask)
                    syn_loss = {"loss_syn": cls_loss}

                    losses.update(syn_loss)
                else:
                    # 14, 1024, 281
                    _, _, total_cls_num = cls_pred.shape
                    syn_class_slice.append(torch.LongTensor([total_cls_num - 1]).to(self.device))
                    if self.syn_class_type == 'clusterMax':
                        cls_pred = torch.cat([cls_pred.index_select(2, ind).max(dim=2, keepdim=True)[0] for ind in syn_class_slice], dim=2)
                    elif self.syn_class_type == 'clusterAvg':
                        cls_pred = torch.cat([cls_pred.index_select(2, ind).mean(dim=2, keepdim=True) for ind in syn_class_slice], dim=2)
                    else:
                        raise NotImplementedError
                    # import pdb;pdb.set_trace()
                    cls_loss = self.loss_labels(cls_pred, targets, indices, valid_mask)
                    syn_loss = {"loss_syn": cls_loss}

                    losses.update(syn_loss)

            # KDKDKDKDKDKDKDKDKDKDKDKDKDKD
            if self.kd_super:
                if 'prior' in self.kd_stu_source:
                    kd_vision_feature = self.kd_proj(vision_feature)
                elif 'posterior' in self.kd_stu_source:
                    kd_vision_feature = self.kd_proj(vision_ebmd)
                else:
                    kd_vision_feature = vision_ebmd

                if self.syn_process_each:
                    kd_text_ebmd = syn_class_ebmd
                elif self.plain_class_super:
                    kd_text_ebmd = plain_class_ebmd
                else:
                    raise NotImplementedError

                if self.fast_kd:
                    kd_v_idx_b, kd_v_idx_i = self.criterion._get_src_permutation_idx(indices)
                    kd_vision_embed = kd_vision_feature[kd_v_idx_b, kd_v_idx_i]    # (n1+n2+...+nn), dim
                    kd_vision_embed = F.normalize(kd_vision_embed, p=2, dim=1)

                    # import pdb;pdb.set_trace()

                    kd_t_idx_i = torch.cat([targets[b]["labels"][i : i+1] for b, (_, tgt) in enumerate(indices) for i in tgt])
                    kd_text_ebmd = kd_text_ebmd.permute(0, 2, 1)[kd_v_idx_b, kd_t_idx_i]

                    masked_images_list = []
                    if "kd_image" in batched_inputs[0]:
                        # masked_images = torch.cat([x["kd_image"].to(self.device) for x in batched_inputs if x["kd_image"] is not None])
                        for b in range(len(indices)):
                            for n in indices[b][1]:
                                masked_images_list.append(batched_inputs[b]["kd_image"][n].to(self.device))
                    else:
                        for b in range(len(indices)):
                            # for n in range(targets[b]["masks"].shape[0]):
                            for n in indices[b][1]: # following the matched indice order
                                gt_mask = targets[b]["masks"][n:n+1]    # 1,h,w
                                masked_image = gt_mask * images.tensor[b]
                                masked_images_list.append(masked_image)

                    masked_images = torch.stack(masked_images_list)  # N,3,h,w

                    cur_len = len(masked_images)

                    ins_num_list = [len(indices[b][1]) for b in range(len(indices))]
                    accm_num_list = list(accumulate([0] + ins_num_list))
                    if cur_len > self.max_kd_len:
                        kd_idx = np.random.choice(np.arange(cur_len), size=self.max_kd_len, replace=False)
                        kd_vision_embed = kd_vision_embed[kd_idx]
                        kd_text_ebmd = kd_text_ebmd[kd_idx]
                        masked_images = masked_images[kd_idx]
                        filter_ins_mask = np.zeros(cur_len)
                        filter_ins_mask[kd_idx] = 1
                        ins_num_list = [filter_ins_mask[accm_num_list[i] : accm_num_list[i + 1]].sum() for i in range(len(indices))]
                        accm_num_list = list(accumulate([0] + ins_num_list))

                    with torch.no_grad():
                        masked_clip_features = self.vision_clip(masked_images)
                    masked_clip_features = masked_clip_features[self.kd_tec_source]
                    if len(masked_clip_features.shape) > 2:
                        masked_clip_features = F.adaptive_avg_pool2d(masked_clip_features, (1, 1)).squeeze()
                    masked_clip_features = F.normalize(masked_clip_features, p=2, dim=1)

                    plain_kd_loss = ((kd_vision_embed - masked_clip_features) ** 2).sum(dim=1)

                    # text
                    text_kd_loss = []
                    for i in range(len(indices)):
                        idx_0, idx_1 = int(accm_num_list[i]), int(accm_num_list[i + 1])
                        if idx_0 == idx_1:
                            continue
                        kd_v_embed_b = kd_vision_embed[idx_0 : idx_1]
                        kd_c_embed_b = masked_clip_features[idx_0 : idx_1]
                        kd_t_embed_b = kd_text_ebmd[idx_0 : idx_1] * self.text_kd_logit_scale

                        v_dis = torch.einsum("al,bl->ab", kd_v_embed_b, kd_c_embed_b).flatten()
                        t_dis = torch.einsum("al,bl->ab", kd_t_embed_b, kd_t_embed_b).flatten()

                        text_kd_loss_b = (t_dis - v_dis) ** 2
                        text_kd_loss.append(text_kd_loss_b)

                    if self.additional_text_kd:
                        plain_kd_loss = plain_kd_loss.mean()
                        text_kd_loss = torch.cat(text_kd_loss).mean()
                        losses.update({"loss_kd" : plain_kd_loss, "loss_text_kd" : text_kd_loss})
                    else:
                        text_kd_loss.append(plain_kd_loss)
                        kd_loss = torch.cat(text_kd_loss).mean()
                        losses.update({"loss_kd" : kd_loss})
                # elif self.fast_kd == "clip":
                #     masked_images_list = []
                #     if "kd_image" in batched_inputs[0]:
                #         # masked_images = torch.cat([x["kd_image"].to(self.device) for x in batched_inputs if x["kd_image"] is not None])
                #         for b in range(len(indices)):
                #             for n in range(targets[b]["masks"].shape[0]):
                #             # for n in indices[b][1]:
                #                 masked_images_list.append(batched_inputs[b]["kd_image"][n].to(self.device))
                #     else:
                #         for b in range(len(indices)):
                #             for n in range(targets[b]["masks"].shape[0]):
                #             # for n in indices[b][1]: # following the matched indice order
                #                 gt_mask = targets[b]["masks"][n:n+1]    # 1,h,w
                #                 masked_image = gt_mask * images.tensor[b]
                #                 masked_images_list.append(masked_image)

                #     masked_images = torch.stack(masked_images_list)  # N,3,h,w

                #     cur_len = len(masked_images)

                #     ins_num_list = [len(indices[b][1]) for b in range(len(indices))]
                #     accm_num_list = list(accumulate([0] + ins_num_list))

                #     with torch.no_grad():
                #         masked_clip_features = self.vision_clip(masked_images)
                #     masked_clip_features = masked_clip_features[self.kd_tec_source]
                #     if len(masked_clip_features.shape) > 2:
                #         masked_clip_features = F.adaptive_avg_pool2d(masked_clip_features, (1, 1)).squeeze()
                #     masked_clip_features = F.normalize(masked_clip_features, p=2, dim=1)

                #     for b in range(len(indices)):
                else:
                    masked_images_list = []
                    kd_loss = 0
                    kd_num = 0
                    # kd_vision_feature = self.kd_proj(vision_feature)
                    for b in range(len(indices)):
                        masked_images_list = []
                        # print(len(indices[b][1]), 'gt')
                        # print(len(indices[b][0]), 'pred')
                        # print(targets[b]["masks"].shape[0], 'aaa')
                        # exit()
                        for n in range(targets[b]["masks"].shape[0]):
                            gt_mask = targets[b]["masks"][n:n+1]   # 1,h,w
                            masked_image = gt_mask * images.tensor[b]
                            masked_images_list.append(masked_image)
                        if not masked_images_list:
                            continue
                        masked_images = torch.stack(masked_images_list)  # N,3,h,w
                        masked_clip_features = self.vision_clip(masked_images)
                        # masked_clip_features = F.adaptive_avg_pool2d(masked_clip_features['attn_l'], (1,1))   # TODO
                        masked_clip_features = masked_clip_features[self.kd_tec_source]
                        # import pdb;pdb.set_trace()
                        if len(masked_clip_features.shape) > 2:
                            masked_clip_features = F.adaptive_avg_pool2d(masked_clip_features, (1, 1)).squeeze(-1).squeeze(-1)
                        masked_clip_features = F.normalize(masked_clip_features, p=2, dim=1)
                        # masked_clip_features['attn]: (n,1024,h,w)
                        for n in range(len(indices[b][1])):
                            per_clip_vision_feature = masked_clip_features[indices[b][1][n]]    # (c, 1, 1) -> (c,)
                            per_clip_vision_feature = F.normalize(per_clip_vision_feature, p=2, dim=0)
                            per_pred_vision_feature = kd_vision_feature[b][indices[b][0][n]]   # (c, )
                            per_pred_vision_feature = F.normalize(per_pred_vision_feature, p=2, dim=0)
                            kd_loss += sum((per_pred_vision_feature - per_clip_vision_feature) ** 2)
                            kd_num += 1
                            # New
                            if len(indices[b][1]) > 1:
                                # Select id
                                # copy_indices = list(indices[b][1])
                                # copy_indices.pop(n)
                                # select_indice = choice(copy_indices)
                                current_class_embd = kd_text_ebmd[b, :, targets[b]["labels"][n:n+1]] * self.text_kd_logit_scale  # (c, 1)
                                for select_indice in list(indices[b][1]):
                                    select_clip_feature = masked_clip_features[indices[b][1][select_indice]]    # (c, 1, 1) -> (c,)
                                    select_clip_feature = F.normalize(select_clip_feature, p=2, dim=0)
                                # all_class_ebmd: (b, c, num_class)
                                # targets[b]["labels"]: (N,)
                                # class name embedding
                                    select_class_embd = kd_text_ebmd[b:b+1, :, targets[b]["labels"][indices[b][1][select_indice]]] * self.text_kd_logit_scale    # (1, c)
                                    text_similar = torch.mm(select_class_embd, current_class_embd).squeeze()

                                    vision_similar = torch.mm(per_pred_vision_feature.unsqueeze(0), select_clip_feature.unsqueeze(1)).squeeze()
                                    kd_loss += (text_similar - vision_similar) ** 2
                                    kd_num += 1
                            # End

                    kd_loss = kd_loss / (kd_num + 1e-5)
                    # kd_loss = kd_loss / (kd_num + 1e-5)
                    losses.update({'loss_kd': kd_loss})
                    # import pdb;pdb.set_trace()

            if self.grounding_super:
                # select matched mask
                # mask_idx = self.criterion._get_src_permutation_idx(indices)
                # mask_idx = self.criterion._get_tgt_permutation_idx(indices)
                
                # import pdb;pdb.set_trace()

                vision_ebmd_mask = None
                # vision_ebmd = mask_avg_pool(features['attn'], pool_mask, use_gt) # (Bs, P, C)
                if self.vision_feature != "queryClass":
                    use_gt = True
                    if use_gt:
                        mask_idx = [tgt for _, tgt in indices]
                        mask = [t["masks"] for t in targets]
                    else:
                        mask_idx = [src for src, _ in indices]
                        mask = torch.clone(outputs["pred_masks"]).detach()
                    max_mask_num = max([len(m_i) for m_i in mask_idx])

                    bsz = len(batched_inputs)
                    _, mask_h, mask_w = mask[0].shape
                    pool_mask = torch.zeros((bsz, max_mask_num, mask_h, mask_w), dtype=mask[0].dtype)
                    vision_ebmd_mask = torch.zeros((bsz, max_mask_num))
                    for ii, idx in enumerate(mask_idx):
                        len_ = len(idx)
                        pool_mask[ii, :len_, :, :] = mask[ii][idx, :, :]
                        vision_ebmd_mask[ii, :len_] = 1
                    
                    vision_ebmd_mask = vision_ebmd_mask.to(self.device)
                    pool_mask = pool_mask.to(self.device)

                    # vision_ebmd = mask_avg_pool(vision_feature, pool_mask, use_gt)
                    vision_g_ebmd = get_sup_propotype(vision_feature, pool_mask, use_gt)
                    if self.embed_proj is not None:
                        vision_g_ebmd = self.embed_proj(vision_g_ebmd)
                else:
                    max_g_v_num = 25
                    bsz, _, dim = vision_ebmd.shape
                    vision_g_ebmd = torch.zeros((bsz, max_g_v_num, dim), dtype=vision_ebmd.dtype).to(self.device)
                    vision_ebmd_mask = torch.zeros((bsz, max_g_v_num), dtype=text_mask.dtype).to(self.device)
                    for b, ind in enumerate(indices):
                        src_idx = ind[0]
                        if len(src_idx) > max_g_v_num:
                            src_idx_idx = np.random.choice(np.arange(len(src_idx)), size=max_g_v_num, replace=False)
                            src_idx = src_idx[src_idx_idx]
                        vision_g_ebmd[b, :len(src_idx)] = vision_ebmd[b][src_idx]
                        vision_ebmd_mask[b, :len(src_idx)] = 1
                    vision_g_ebmd = vision_g_ebmd.to(self.device)

                # import pdb;pdb.set_trace()
                if self.grounding_gather:
                    word_size = dist.get_world_size()
                    cur_rank = dist.get_rank()
                    
                    gather_text_ebmd = [torch.ones_like(text_ebmd)
                        for _ in range(word_size)]
                    dist.all_gather(gather_text_ebmd, text_ebmd, async_op=False)
                    gather_text_ebmd[cur_rank] = text_ebmd
                    gather_text_ebmd = torch.cat(gather_text_ebmd, dim=0)
                    
                    gather_vision_ebmd = [torch.ones_like(vision_g_ebmd)
                        for _ in range(word_size)]
                    dist.all_gather(gather_vision_ebmd, vision_g_ebmd, async_op=False)
                    gather_vision_ebmd[cur_rank] = vision_g_ebmd
                    gather_vision_ebmd = torch.cat(gather_vision_ebmd, dim=0)

                    gather_text_mask = [torch.ones_like(text_mask)
                        for _ in range(word_size)]
                    dist.all_gather(gather_text_mask, text_mask, async_op=False)
                    # gather_text_mask[cur_rank] = text_mask
                    gather_text_mask = torch.cat(gather_text_mask, dim=0)

                    # if vision_ebmd_mask is None:
                    #     gather_vision_ebmd_mask = None
                    # else:
                    gather_vision_ebmd_mask = [torch.ones_like(vision_ebmd_mask)
                        for _ in range(word_size)]
                    dist.all_gather(gather_vision_ebmd_mask, vision_ebmd_mask, async_op=False)
                    # gather_vision_ebmd_mask[cur_rank] = vision_ebmd_mask
                    gather_vision_ebmd_mask = torch.cat(gather_vision_ebmd_mask, dim=0)

                    # for i, t_e in enumerate(gather_text_ebmd):
                    #     print(dist.get_rank(), text_ebmd.mean(), i, t_e.shape, t_e.mean(), t_e.requires_grad)
                    grounding_loss = self.grounding_head(gather_text_ebmd, gather_vision_ebmd, gather_text_mask, gather_vision_ebmd_mask, temperature=logit_scale)
                else:
                    grounding_loss = self.grounding_head(text_ebmd, vision_g_ebmd, text_mask, vision_ebmd_mask, temperature=logit_scale)
                losses.update(grounding_loss)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            # training sample
            # overfit/
            # mask_cls_results = outputs["pred_logits"]
            cls_id = batched_inputs[0]['name2id'].long().to(self.device)

            if self.vision_feature != "queryClass":
                # vision_ebmd = mask_avg_pool(vision_feature, outputs["pred_masks"])
                vision_ebmd = get_sup_propotype(vision_feature, outputs["pred_masks"])
            else:
                vision_ebmd = vision_feature
            if self.embed_proj is not None:
                vision_ebmd = self.embed_proj(vision_ebmd)

            self.infer_num_class = batched_inputs[0]['num_class']
            # vision_ebmd = mask_avg_pool(features['attn'], outputs["pred_masks"])

            if self.region_supervision:
                text_ebmd = all_class_ebmd
                text_ebmd_0 = torch.cat([text_ebmd, self.non_object_embedding[None, :, :]], dim=1)

                vision_ebmd = vision_ebmd / (vision_ebmd.norm(dim=-1, keepdim=True) + 1e-7)
                text_ebmd_0 = text_ebmd_0 / (text_ebmd_0.norm(dim=-1, keepdim=True) + 1e-7)

                text_ebmd_0 = text_ebmd_0.permute(0, 2, 1)
                mask_cls_results = torch.bmm(vision_ebmd * logit_scale, text_ebmd_0)
                cls_pred = mask_cls_results[0]

                ###########
                # text_ebmd_0 = all_class_ebmd

                # vision_ebmd = vision_ebmd / (vision_ebmd.norm(dim=-1, keepdim=True) + 1e-7)
                # text_ebmd_0 = text_ebmd_0 / (text_ebmd_0.norm(dim=-1, keepdim=True) + 1e-7)

                # text_ebmd_0 = text_ebmd_0.permute(0, 2, 1)
                # mask_cls_results = torch.bmm(vision_ebmd * logit_scale, text_ebmd_0)

                # _, num_mask, _ = vision_ebmd.shape
                # non_pred = (torch.ones((1, num_mask, 1)) * (-1e7)).to(self.device)
                # mask_cls_results = torch.cat([mask_cls_results, non_pred], dim=2)

                # cls_pred = mask_cls_results[0]
                #--------------

                # cls_pred = self.get_sim_logits(vision_ebmd[0], text_ebmd_0, logit_scale)
                # mask_cls_results = cls_pred.unsqueeze(0)

                # import pdb;pdb.set_trace()

            else:
                cls_pred = self.get_sim_logits(vision_ebmd[0], text_ebmd[0], 0.1)
                num_mask, _ = cls_pred.shape
                # mask_cls_results = cls_pred.softmax(dim=-1)
                mask_cls_results = torch.ones((1, num_mask, self.infer_num_class + 1)).to(self.device) * (-1e7)
                # mask_cls_results = torch.ones((1, num_mask, self.sem_seg_head.num_classes + 1)).to(self.device) * (-1e7)
                # TODO single class, multi name
                mask_cls_results[0, :, cls_id] = cls_pred

            #-----------------------------
            #### debug

            debug = True
            debug = False
            if debug:
                import os
                import imageio
                # import numpy as np

                # import pdb;pdb.set_trace()
                cls_name = batched_inputs[0]['cls_name']
                cls_name.append("nothing")
                valid_idx = torch.where(cls_pred.sum(dim=1) != 0)[0]
                value, cls_name_idx = cls_pred.softmax(dim=-1).max(dim=-1)
                ### batched_inputs[0]['instances']._fields['gt_classes']
                ### debug commanc, cls_name, cls_name_idx[valid_idx], np.array(cls_name)[cls_name_idx[valid_idx].cpu()]
                def save_visual(sub_path):
                    if not os.path.exists("tmp/" + sub_path):
                        os.mkdir("tmp/" + sub_path)
                    imageio.imwrite("tmp/{}/img.jpg".format(sub_path), np.array(batched_inputs[0]['image'].permute(1, 2, 0)))
                    for i in range(100):
                        if i in valid_idx:
                            _name = cls_name[cls_name_idx[i]]
                        else:
                            _name = "invalid"
                        mask = outputs["pred_masks"][0, i].sigmoid()
                        if torch.sum(mask > 0.3) > 30:
                            imageio.imwrite("tmp/{}/mask_{}_{}_{}.png".format(sub_path, i, _name, value[i]), 
                                np.array(mask.cpu().detach()))

                def save_gt(sub_path):
                    if not os.path.exists(f"tmp/{sub_path}"):
                        os.mkdir(f"tmp/{sub_path}")
                    sem_gt = batched_inputs[0]['sem_seg']
                    # gt_mask = torch.zeros((1, len(sem_gt.unique()), *sem_gt.shape), dtype=sem_gt.dtype).to(self.device)
                    for i, idx in enumerate(sem_gt.unique()):
                        gt_mask = torch.zeros_like(sem_gt)
                        gt_mask[sem_gt == idx] = 1
                        imageio.imwrite("tmp/{}/seg_{}_gt.png".format(sub_path, idx), np.array(gt_mask))
                
                import pdb;pdb.set_trace()
            #############################

            # if some config TODO
            # import pdb;pdb.set_trace()
            if self.infer_num_class != len(cls_id):
                _cls_id = torch.cat((cls_id, torch.LongTensor([self.infer_num_class]).to(self.device)))
                # import pdb;pdb.set_trace()
                mask_cls_results = torch.cat([mask_cls_results.index_select(2, (_cls_id == i).nonzero().flatten()).max(dim=2, keepdim=True)[0] for i in range(self.infer_num_class + 1)], dim=2)

            # vision_ebmd = vision_ebmd / (vision_ebmd.norm(dim=-1, keepdim=True) + 1e-7)
            # text_ebmd = text_ebmd / (text_ebmd.norm(dim=-1, keepdim=True) + 1e-7)
            # cls_pred = self.get_sim_logits(vision_ebmd[0], text_ebmd[0])

            # import pdb;pdb.set_trace()

            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results


    def loss_labels_process_each(self, src_logits, targets, indices, ignore_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert "pred_logits" in outputs
        # src_logits = outputs["pred_logits"].float()

        idx = self.criterion._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.max_cap_num, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_classes[ignore_masks < 1] = -100

        if self.region_dropout > 0:
            drop = np.random.rand(len(idx[0]))
            # drop_idx = torch.clone(idx)
            drop_idx_idx = drop < self.region_dropout
            drop_idx_0 = idx[0][drop_idx_idx]
            drop_idx_1 = idx[1][drop_idx_idx]

            # import pdb;pdb.set_trace()

            target_classes[(drop_idx_0, drop_idx_1)] = -100

        # import pdb;pdb.set_trace()
        clss_weight = torch.ones(self.max_cap_num + 1).to(self.device)
        clss_weight[-1] = self.empty_weight

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, clss_weight)
        # losses = {"loss_ce": loss_ce}
        return loss_ce


    def loss_labels(self, src_logits, targets, indices, ignore_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert "pred_logits" in outputs
        # src_logits = outputs["pred_logits"].float()

        idx = self.criterion._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.criterion.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_classes[ignore_masks < 1] = -100

        if self.region_dropout > 0:
            drop = np.random.rand(len(idx[0]))
            # drop_idx = torch.clone(idx)
            drop_idx_idx = drop < self.region_dropout
            drop_idx_0 = idx[0][drop_idx_idx]
            drop_idx_1 = idx[1][drop_idx_idx]

            # import pdb;pdb.set_trace()

            target_classes[(drop_idx_0, drop_idx_1)] = -100

        # import pdb;pdb.set_trace()
        clss_weight = torch.ones(self.criterion.num_classes + 1).to(self.device)
        clss_weight[-1] = self.empty_weight

        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.criterion.empty_weight)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, clss_weight)
        # losses = {"loss_ce": loss_ce}
        return loss_ce


    def get_sim_logits(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        temperature: float = 100,
    ):
        return temperature * image_features @ text_features.T


    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets


    def prepare_targets_syn(self, targets, images, syn_class):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image, syn in zip(targets, syn_class):
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": syn,
                    "masks": padded_masks,
                }
            )
        return new_targets


    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

        # import os
        # import imageio
        # import numpy as np
        # # from .data.datasets.register_vspw import VSPW_CATEGORIES

        # def save_seg(path):
        #     if not os.path.exists(f"tmp/{path}"):
        #         os.mkdir(f"tmp/{path}")
        #     for i, seg_i in enumerate(semseg):
        #         if torch.max(seg_i) > 0.3:
        #             imageio.imwrite("./tmp/{}/seg_{}_{}.png".format(path, i, COCO_CATEGORIES[i]['name']), np.array(seg_i.cpu()))

        # import pdb;pdb.set_trace()

        return semseg


    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        # keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        keep = labels.ne(self.infer_num_class) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            
            # print("YOU didn't detect any mask :)")
            
            # import pdb;pdb.set_trace()

            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            # import pdb;pdb.set_trace()

            return panoptic_seg, segments_info


    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        labels = torch.arange(self.infer_num_class, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        # topk_indices = topk_indices // self.sem_seg_head.num_classes
        topk_indices = topk_indices // self.infer_num_class
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        # import pdb;pdb.set_trace()

        return result
