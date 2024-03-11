# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
# from numpy.lib.utils import get_include

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
# from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.visualizer import Visualizer, ColorMode

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

from zs3.dataloaders import make_data_loader
from zs3.modeling.deeplab import DeepLab
from zs3.modeling.aspp import build_aspp
from zs3.modeling.backbone import build_backbone
from zs3.modeling.decoder import build_decoder

from zs3.modeling.sync_batchnorm.replicate import patch_replication_callback
from zs3.dataloaders.datasets import DATASETS_DIRS
from zs3.utils.calculate_weights import calculate_weigths_labels
from zs3.utils.loss import SegmentationLosses
from zs3.utils.lr_scheduler import LR_Scheduler
from zs3.utils.metrics import Evaluator
from zs3.utils.saver import Saver
from zs3.utils.summaries import TensorboardSummary
from zs3.parsing import get_parser
from zs3.exp_data import CLASSES_NAMES
from zs3.base_trainer import BaseTrainer
from zs3.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.nn.functional as F

@META_ARCH_REGISTRY.register()
class OpenSegDeeplabv3Text(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: nn.Module,
        aspp: nn.Module,
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
        grounding_super: bool,
        learned_temp: bool,
        region_dropout: float,
        plain_class_super: bool,
        syn_class_type: str,
        vision_feature: str,
        embed_dim: int,
        freeze_backbone: bool,
        embed_proj: bool,
        kd_stu_source: str,
        kd_tec_source: str,
        max_kd_len: int,
        fast_kd: bool,
        additional_text_kd: bool,
        grounding_gather: bool,
        dec_layers: int,
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
        self.aspp = aspp
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
        # self.grounding_head = grounding_head if grounding_super else None
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
        # self.kd_type = kd_type
        assert 0 <= self.region_dropout < 1

        if hasattr(self.backbone, 'init_weights'):
            self.backbone.init_weights()
        self.text_encoder.init_weights()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        
        self.non_object_embedding = nn.Parameter(
            torch.empty(1, self.text_encoder.text_projection.shape[-1])
        )
        nn.init.normal_(
            self.non_object_embedding.data,
            std=self.text_encoder.transformer.width ** -0.5,
        )

        # empty_weight = torch.ones(1) * no_object_weight
        # self.register_buffer("empty_weight", empty_weight)

        self.learned_temp = learned_temp
        if self.learned_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # if self.vision_feature != "resAtten" and hasattr(self.backbone, 'remove_attenpool'):
        #     self.backbone.remove_attenpool()
        
        # attn_shape = self.backbone.output_shape()['attn'] if 'attn' in self.backbone.output_shape() else 1024
        # vision_in_dim = {
        #     "resAtten": attn_shape,
        #     "res5": self.backbone.output_shape()['res5'],
        #     "pixelDecoder": self.sem_seg_head.pixel_decoder.mask_dim,
        #     "queryClass": self.sem_seg_head.predictor.hidden_dim,
        # }
        # if self.vision_feature != "query-class":
        # in_dim = vision_in_dim[self.vision_feature]
        # if self.vision_feature == 'resAtten' and not embed_proj:
        #     self.embed_proj = None
        # else:
        in_dim = 256
        self.embed_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_dim, embed_dim)
        )
        #     for i in range(dec_layers - 1):
        #         aux_embed_proj = nn.Sequential(
        #             nn.ReLU(),
        #             nn.Linear(in_dim, embed_dim)
        #         )
        #         setattr(self, "aux_embed_porj_{}".format(i), aux_embed_proj)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # KD
        
        

    @classmethod
    def from_config(cls, cfg):
        # backbone = build_backbone(cfg)
        # backbone_out_shape = backbone.output_shape()
        # if 'attn' in backbone_out_shape:
        #     backbone_out_shape.pop('attn')
        
        global_avg_pool_bn = True
        output_stride=16
        BatchNorm = nn.BatchNorm2d
        num_classes = 133+1
        backbone = build_backbone(
            output_stride,
            BatchNorm,
            pretrained=False,
            imagenet_pretrained_path="",
        )
        
        aspp = build_aspp(output_stride, BatchNorm, global_avg_pool_bn)
        # sem_seg_head = build_sem_seg_head(cfg, backbone_out_shape)
        sem_seg_head = build_decoder(num_classes, BatchNorm)
        # sem_seg_head_out_shape = sem_seg_head.output_shape()

        text_encoder = CLIPTextEncoder(cfg)

        # vision_clip = CLIPResNetWithAttention(cfg)

        # # Loss parameters:
        # deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        # no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # # loss weights
        # class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        # dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        # mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # grounding_weight = cfg.MODEL.GROUNDING.LOSS_WEIGHT
        # # class_weight = 0
        # # dice_weight = 0
        # # mask_weight = 0
        # # grounding_weight = 0
        # # kd_weight = cfg.MODEL.KD.LOSS_WEIGHT

        # # building criterion
        # region_supervision = cfg.MODEL.PER_REGION.ENABLED
        # cost_class = 2 if cfg.MODEL.PER_REGION.PLAIN_CLASS else -1
        # matcher = HungarianMatcher(
        #     cost_class=cost_class,
        #     cost_mask=mask_weight,
        #     cost_dice=dice_weight,
        #     num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        # )

        # kd_type = cfg.MODEL.KD.KD_TYPE

        # weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight}
        # dense_list = ['loss_mask', 'loss_dice', 'loss_ce', 'loss_grounding']
        # # weight_dict = {}
        # # dense_list = []
        # # if region_supervision:
        # #     weight_dict['loss_ce'] = class_weight

        # if cfg.MODEL.PER_REGION.PLAIN_CLASS:
        #     weight_dict['loss_ce'] = cfg.MODEL.PER_REGION.PLAIN_LOSS_WEIGHT
        #     # weight_dict['loss_ce'] = 0

        # if deep_supervision:
        #     dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        #     aux_weight_dict = {}
        #     for i in range(dec_layers - 1):
        #         aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items() if k in dense_list})
        #     weight_dict.update(aux_weight_dict)

        # losses = ["masks"]
        # # if region_supervision:
        # #     losses.append("labels")

        # criterion = SetCriterion(
        #     sem_seg_head.num_classes,
        #     matcher=matcher,
        #     weight_dict=weight_dict,
        #     eos_coef=no_object_weight,
        #     losses=losses,
        #     num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        #     oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
        #     importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        # )
        criterion = SegmentationLosses(weight=None, cuda=True).build_loss(
            mode='ce'
        )


        # grounding_head = GroundingHead(cfg)

        return {
            "backbone": backbone,
            "aspp": aspp,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "text_encoder" : text_encoder,
            "grounding_super": cfg.MODEL.GROUNDING.ENABLED,
            "learned_temp": cfg.MODEL.LEARNED_TEMP,
            "region_dropout": cfg.MODEL.PER_REGION.DROPOUT,
            "plain_class_super": cfg.MODEL.PER_REGION.PLAIN_CLASS,
            "syn_class_type": cfg.MODEL.PER_REGION.SYN_CLASS.TYPE,
            "vision_feature": cfg.MODEL.VISION_FEATURE,
            "embed_dim": cfg.MODEL.TEXT_ENCODER.EMBED_DIM,
            "freeze_backbone": cfg.MODEL.FREEZE_BACKBONE,
            "embed_proj": cfg.MODEL.PROJECTION,
            "kd_stu_source": cfg.MODEL.KD.STU_SOURCE,
            "kd_tec_source": cfg.MODEL.KD.TEC_SOURCE,
            "max_kd_len": cfg.MODEL.KD.MAX_LIST_LEN,
            "fast_kd": cfg.MODEL.KD.FAST_KD,
            "additional_text_kd": cfg.MODEL.KD.TEXT_KD_WEIGHT > 0,
            "grounding_gather": cfg.MODEL.GROUNDING.GATHER,
            "dec_layers": cfg.MODEL.MASK_FORMER.DEC_LAYERS,
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


        features,low_level_feat = self.backbone(images.tensor)
        # import pdb; pdb.set_trace()
        # torch.save(outs, "Deeplab_subnetwork.pt")

        # from thop import profile
        # flops, params = profile(self.aspp, inputs=(features,))
        # print('FLOPs = ' + str(flops/1000**3) + 'G')
        # print('Params = ' + str(params/1000**2) + 'M')
        # import numpy as np
        # params3 = sum(p.numel() for p in self.aspp.parameters())
        # import pdb; pdb.set_trace()
        # images.tensor: (B,C,H,W)
        features = self.aspp(features)

        outputs = self.sem_seg_head(features,low_level_feat)

        # import pdb; pdb.set_trace()
        # # from thop import profile
        # flops, params = profile(self.sem_seg_head, inputs=(features,low_level_feat, ))
        # print('FLOPs = ' + str(flops/1000**3) + 'G')
        # print('Params = ' + str(params/1000**2) + 'M')
        # import numpy as np
        # params3 = sum(p.numel() for p in self.aspp.parameters())
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # 测试一下下面的语句到底是否需要
        # outputs = F.interpolate(outputs, size=images.tensor.size()[2:],mode="bilinear", align_corners=True)
        vision_feature = outputs
        # # loss_band
        # print(self.grounding_super)
        # print(self.plain_class_super)
        # print(self.grounding_super)
        # self.grounding_super = False
        # self.plain_class_super = False
        # self.grounding_super = False

        if self.grounding_super:
            text = torch.stack([x["caption"].to(self.device) for x in batched_inputs])
            text_mask = torch.stack([x["caption_mask"].to(self.device) for x in batched_inputs])
            text_ebmd = self.text_encoder(text)
        if self.plain_class_super:
            plain_class = torch.stack([x["all_caption"].to(self.device) for x in batched_inputs])
            plain_class_ebmd = self.text_encoder(plain_class)
            all_class_ebmd = plain_class_ebmd

        if self.learned_temp:
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = 10

        # if self.vision_feature == 'resAtten':
        #     vision_feature = features['attn']
        # elif self.vision_feature == 'res5':
        #     vision_feature = features['res5']
        # elif self.vision_feature == 'pixelDecoder':
        #     vision_feature = seg_feature['pixel_feature']
        #     vision_feature = F.interpolate(vision_feature, size=(48, 48), mode='bilinear')
        # elif self.vision_feature == 'queryClass':
        #     vision_feature = seg_feature['query_embed'] #b, 100, c
        # else:
        #     raise NotImplementedError

        # print(features['attn'].shape, features['res5'].shape, seg_feature['pixel_feature'].shape)#, seg_feature['query_embed'].shape)
        # import pdb;pdb.set_trace()

        bsz, channel, *_ = vision_feature.shape

        if self.training:

            # mask classification target
            # if "instances" in batched_inputs[0]:
            #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            #     targets = self.prepare_targets(gt_instances, images)
            # else:
            #     targets = None

            targets = [x["onehot"].to(self.device) for x in batched_inputs]


            # if self.region_supervision:
                # import pdb;pdb.set_trace()
            # if self.vision_feature != "queryClass":
            #     # vision_ebmd, valid_mask = mask_avg_pool(vision_feature, outputs["pred_masks"], use_gt=False, return_mask=True)
            #     vision_ebmd, valid_mask = get_sup_propotype(vision_feature, outputs["pred_masks"], use_gt=False, return_mask=True)
            # else:
            #     vision_ebmd = vision_feature
            #     valid_mask = torch.ones((bsz, channel))
            # 合并空间维度操作
            # import pdb; pdb.set_trace()
            # [256,128,128]->[256,16384]
            vision_ebmd = torch.reshape(vision_feature,(bsz,256,-1))
            vision_ebmd = vision_ebmd.permute(0, 2, 1)
            # 把w放到第一位
            # vision_ebmd = vision_feature
            # import pdb; pdb.set_trace()
            
            if self.embed_proj is not None:
                vision_ebmd = self.embed_proj(vision_ebmd)
            vision_ebmd_norm = vision_ebmd / (vision_ebmd.norm(dim=-1, keepdim=True) + 1e-7)
            # aux_vision_ebmd = []
            # for i, aux_query in enumerate(seg_feature['query_feature']):
            #     aux_embed_proj = getattr(self, "aux_embed_porj_{}".format(i))
            #     aux_query = aux_embed_proj(aux_query)
            #     aux_vision_ebmd.append(aux_query)

            # if self.plain_class_super:
                # vision_ebmd, valid_mask = mask_avg_pool(features['attn'], outputs["pred_masks"], use_gt=False, return_mask=True)
                # vision (bsz, 100, dim_c), text (bsz, 133, dim_c)
                # similarity (bsz, 100, 133)
            # 把w放到第一位
            # vision_ebmd_norm = vision_ebmd_norm.permute(0,2,1)
            # # import pdb; pdb.set_trace()
            # vision_ebmd_norm = torch.reshape(vision_ebmd_norm,(bsz, 1024, 512, 512))
            # # (512, bsz, 512, 1024)
            # vision_ebmd_norm = vision_ebmd_norm.permute(3, 0, 2, 1)
            # ebmds = torch.chunk(vision_ebmd_norm, vision_ebmd_norm.size(0), dim=0)

            plain_class_ebmd = torch.cat([plain_class_ebmd,
                self.non_object_embedding[None, :, :].repeat(len(batched_inputs), 1, 1)], dim=1)
            # import pdb; pdb.set_trace()
            plain_class_ebmd = plain_class_ebmd / (plain_class_ebmd.norm(dim=-1, keepdim=True) + 1e-7)
            plain_class_ebmd = plain_class_ebmd.permute(0, 2, 1)
            # # original
            # import pdb; pdb.set_trace()
            cls_pred = torch.bmm(vision_ebmd_norm * logit_scale, plain_class_ebmd)
            # [bsz,16384,134]->[bsz,128,128,134]
            cls_pred = torch.reshape(cls_pred,(bsz,128,128,134))
            # outputs['pred_logits'] = cls_pred
            # import pdb; pdb.set_trace()
            ## cuda out of memory
            # for i, ebmd in enumerate(ebmds) :
            #     ebmd = ebmd.squeeze()
            #     if i == 0:
            #         cls_pred = torch.bmm(ebmd * logit_scale, plain_class_ebmd)
            #         cls_pred = cls_pred.unsqueeze(0)
            #     else:
            #         cls_pred_new = torch.bmm(ebmd * logit_scale, plain_class_ebmd)
            #         cls_pred_new = cls_pred_new.unsqueeze(0)
            #         cls_pred = torch.cat((cls_pred, cls_pred_new), dim=0)
            # import pdb; pdb.set_trace()
            
            cls_pred = cls_pred.permute(0, 3, 1, 2) #确认一下，和targets对齐
            cls_pred = F.interpolate(cls_pred, scale_factor=4, mode='bilinear', align_corners=True)
            
            # targets = torch.stack(targets, dim=0).permute(0,3,1,2)
            # targets = targets.float()
            targets = [x["gt_label"].to(self.device) for x in batched_inputs]
            targets = torch.stack(targets, dim=0)
            targets = targets.float()
            #only ce_loss

            losses = self.criterion(cls_pred, targets)

                # aux_pred_logits = []
                # for i, aux_query in enumerate(aux_vision_ebmd):
                #     aux_query_norm = aux_query / (aux_query.norm(dim=-1, keepdim=True) + 1e-7)
                #     aux_cls_pred = torch.bmm(aux_query_norm * logit_scale, plain_class_ebmd)

                #     outputs['aux_outputs'][i]['pred_logits'] = aux_cls_pred
                #     aux_pred_logits.append(aux_cls_pred)

            # bipartite matching-based loss
            # losses, indices, aux_indices = self.criterion(outputs, targets, get_all_indice=True)

            # if self.plain_class_super:
            #     cls_loss = self.loss_labels(cls_pred, targets, indices, valid_mask)
            #     losses.update({"loss_ce": cls_loss})

            #     for i, aux_pred in enumerate(aux_pred_logits):
            #         cls_loss = self.loss_labels(aux_pred, targets, aux_indices[i], valid_mask)
            #         losses.update({"loss_ce_{}".format(i) : cls_loss})

            # KDKDKDKDKDKDKDKDKDKDKDKDKDKD
            


            # # grouding supervision
            # for k in list(losses.keys()):
            #     if k in self.criterion.weight_dict:
            #         losses[k] *= self.criterion.weight_dict[k]
            #     else:
            #         # remove this loss if not specified in `weight_dict`
            #         losses.pop(k)

            return losses
        else:
            # training sample
            # overfit/
            # mask_cls_results = outputs["pred_logits"]
            cls_id = batched_inputs[0]['name2id'].long().to(self.device)

            # import pdb; pdb.set_trace()
            bsz, channel, h, w = vision_feature.shape
            vision_ebmd = vision_feature
             # [256,128,128]->[256,16384]
            vision_ebmd = torch.reshape(vision_feature,(bsz,256,-1))
            vision_ebmd = vision_ebmd.permute(0, 2, 1)
            
            if self.embed_proj is not None:
                vision_ebmd = self.embed_proj(vision_ebmd)

            self.infer_num_class = batched_inputs[0]['num_class']
            # vision_ebmd = mask_avg_pool(features['attn'], outputs["pred_masks"])

            # if self.region_supervision:
            # import pdb;pdb.set_trace()
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

            # else:
            #     cls_pred = self.get_sim_logits(vision_ebmd[0], text_ebmd[0], 0.1)
            #     num_mask, _ = cls_pred.shape
            #     # mask_cls_results = cls_pred.softmax(dim=-1)
            #     mask_cls_results = torch.ones((1, num_mask, self.infer_num_class + 1)).to(self.device) * (-1e7)
            #     # mask_cls_results = torch.ones((1, num_mask, self.sem_seg_head.num_classes + 1)).to(self.device) * (-1e7)
            #     # TODO single class, multi name
            #     mask_cls_results[0, :, cls_id] = cls_pred

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

            outdir = '/opt/data/private/xjx/data/sup/pc-59-DeeplabV3'
            processed_results = []
            for input_per_image, image_size in zip(
                batched_inputs, images.image_sizes
            ):
                processed_results.append({})
                
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                cls_pred = F.softmax(cls_pred, dim=-1)[..., :-1]
                a,c = cls_pred.shape
                r = torch.reshape(cls_pred, (h,w,c))
                r = r.permute(2, 0, 1)
                # [133, 200, 304]
                if not self.sem_seg_postprocess_before_inference:
                    # [133, 426, 640]
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    # vis
                    import cv2
                    gt = input_per_image["sem_seg"]
                    im = cv2.imread(input_per_image['file_name'])
                    gt = F.interpolate(gt.float().unsqueeze(0).unsqueeze(0), size=(im.shape[0], im.shape[1]), mode="nearest").squeeze().long().cpu()
                    _,pre = r.max(dim = 0)
                    import os
                    outpath = os.path.join(outdir, input_per_image['file_name'][-12:-4])
                    if not os.path.exists(outpath):
                        os.makedirs(outpath)
                    gt_vis = Visualizer(im, self.metadata).draw_sem_seg(gt, alpha=0.6)
                    pre_vis = Visualizer(im, self.metadata).draw_sem_seg(pre.cpu(), alpha=0.6)

                    cv2.imwrite(os.path.join(outpath, 'im.png'), im)
                    cv2.imwrite(os.path.join(outpath, 'gt.png'), gt_vis.get_image())
                    cv2.imwrite(os.path.join(outpath, 'pre-ft.png'), pre_vis.get_image())


                
                processed_results[-1]["sem_seg"] = r

            # vision_ebmd = vision_ebmd / (vision_ebmd.norm(dim=-1, keepdim=True) + 1e-7)
            # text_ebmd = text_ebmd / (text_ebmd.norm(dim=-1, keepdim=True) + 1e-7)
            # cls_pred = self.get_sim_logits(vision_ebmd[0], text_ebmd[0])

            # import pdb;pdb.set_trace()

            # mask_pred_results = outputs["pred_masks"]
            # # upsample masks
            # mask_pred_results = F.interpolate(
            #     mask_pred_results,
            #     size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            #     mode="bilinear",
            #     align_corners=False,
            # )

            # del outputs

            # processed_results = []
            # for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
            #     mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            # ):
            #     height = input_per_image.get("height", image_size[0])
            #     width = input_per_image.get("width", image_size[1])
            #     processed_results.append({})

            #     if self.sem_seg_postprocess_before_inference:
            #         mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
            #             mask_pred_result, image_size, height, width
            #         )
            #         mask_cls_result = mask_cls_result.to(mask_pred_result)

            #     # semantic segmentation inference
            #     if self.semantic_on:
            #         r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
            #         if not self.sem_seg_postprocess_before_inference:
            #             r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
            #         processed_results[-1]["sem_seg"] = r

            #     # panoptic segmentation inference
            #     if self.panoptic_on:
            #         panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
            #         processed_results[-1]["panoptic_seg"] = panoptic_r
                
            #     # instance segmentation inference
            #     if self.instance_on:
            #         instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
            #         processed_results[-1]["instances"] = instance_r

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


    def loss_syn_label(self, cls_pred, targets, indices, valid_mask, syn_class_slice=None):
        if self.syn_process_each:
            cls_loss = self.loss_labels_process_each(cls_pred, targets, indices, valid_mask)
            # syn_loss = {"loss_syn": cls_loss}
        else:
            # 14, 1024, 281
            _, _, total_cls_num = cls_pred.shape
            if self.syn_class_type == 'clusterMax':
                cls_pred = torch.cat([cls_pred.index_select(2, ind).max(dim=2, keepdim=True)[0] for ind in syn_class_slice], dim=2)
            elif self.syn_class_type == 'clusterAvg':
                cls_pred = torch.cat([cls_pred.index_select(2, ind).mean(dim=2, keepdim=True) for ind in syn_class_slice], dim=2)
            else:
                raise NotImplementedError
            # import pdb;pdb.set_trace()
            cls_loss = self.loss_labels(cls_pred, targets, indices, valid_mask)
            # syn_loss = {"loss_syn": cls_loss}

        return cls_loss



    def indice2main(self, indice, target_indice):
        # import pdb;pdb.set_trace()
        tgt2src_list = [{t.item() : s.item() for (s, t) in zip(src, tgt)} for (src, tgt) in indice]
        new_indice = [(torch.tensor([tgt2src_list[i][t.item()] for t in tgt], dtype=torch.long), torch.tensor([t.item() for t in tgt], dtype=torch.long)) for i, (_, tgt) in enumerate(target_indice)]

        return new_indice


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
    
    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.sem_seg_head]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p