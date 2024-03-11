# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.MASK_FORMER.TEST.VOCA_MULTINAME = False
    cfg.MODEL.MASK_FORMER.TEST.VOCA_FULL_CLASS = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # Open voca
    cfg.MODEL.MASK_FORMER.MASK_CLASSIFICATION = True
    cfg.MODEL.LEARNED_TEMP = False
    cfg.MODEL.VISION_FEATURE = 'resAtten' # 'res5' 'resAtten' 'pixelDecoder' 'queryClass'
    cfg.MODEL.PROJECTION = True
    cfg.MODEL.FREEZE_BACKBONE = False

    cfg.MODEL.PER_REGION = CN()
    cfg.MODEL.PER_REGION.ENABLED = False
    cfg.MODEL.PER_REGION.DROPOUT = 0.
    cfg.MODEL.PER_REGION.DENSE_SUPERVISE = False
    cfg.MODEL.PER_REGION.PLAIN_CLASS = True             # 'person' ....
    cfg.MODEL.PER_REGION.PLAIN_LOSS_WEIGHT = 10.0
    cfg.MODEL.PER_REGION.SYN_CLASS = CN()
    cfg.MODEL.PER_REGION.SYN_CLASS.ENABLED = False
    cfg.MODEL.PER_REGION.SYN_CLASS.LOSS_WEIGHT = 10.0
    cfg.MODEL.PER_REGION.SYN_CLASS.TYPE = 'thresRand'   # ['thresRand', 'clusterMax', 'clusterAvg']


    # CLIP
    cfg.MODEL.TEXT_ENCODER = CN()
    cfg.MODEL.TEXT_ENCODER.CONTEXT_LENGTH = 77
    cfg.MODEL.TEXT_ENCODER.VOCAB_SIZE = 49408
    cfg.MODEL.TEXT_ENCODER.TRANSFORMER_WIDTH = 512
    cfg.MODEL.TEXT_ENCODER.TRANSFORMER_HEADS = 8
    cfg.MODEL.TEXT_ENCODER.TRANSFORMER_LAYERS = 12
    cfg.MODEL.TEXT_ENCODER.EMBED_DIM = 1024
    cfg.MODEL.TEXT_ENCODER.OUT_DIM = 256
    cfg.MODEL.TEXT_ENCODER.PRETRAINED = None
    cfg.MODEL.TEXT_ENCODER.INPUT = 'caption'
    cfg.MODEL.TEXT_ENCODER.PROMPT = '{}'

    # KD
    cfg.MODEL.KD = CN()
    cfg.MODEL.KD.STU_SOURCE = 'prior' # prior, posterior, direct
    cfg.MODEL.KD.TEC_SOURCE = 'attn_g' # attn_g, attn_l, res5
    cfg.MODEL.KD.MAX_LIST_LEN = 200
    cfg.MODEL.KD.FAST_KD = True
    cfg.MODEL.KD.TEXT_KD_WEIGHT = -1.0
    cfg.MODEL.KD.KD_WEIGHT = 10.0
    cfg.MODEL.KD.ENABLED = True
    cfg.MODEL.KD.KD_TYPE = "vision"

    cfg.MODEL.VIT = CN()
    cfg.MODEL.VIT.INPUT_RESOLUTION = 512
    cfg.MODEL.VIT.PATCH_SIZE = 16
    cfg.MODEL.VIT.WIDTH = 768
    cfg.MODEL.VIT.LAYERS = 12
    cfg.MODEL.VIT.HEADS = 12
    cfg.MODEL.VIT.OUTPUT_DIM = 512
    cfg.MODEL.VIT.DROP_PATH_RATE = 0.1
    cfg.MODEL.VIT.OUT_INDICES = [3, 5, 7, 11]
    cfg.MODEL.VIT.PRETRAINED = None
    cfg.MODEL.VIT.OUT_FEATURES = ["res2", "res3", "res4", "res5", "attn"]
    # cfg.MODEL.VIT.GET_EMBEDDINGS = False

    cfg.MODEL.RESNETS.LAYERS = [3, 4, 6, 3]
    cfg.MODEL.RESNETS.OUTPUT_DIM = 1024
    cfg.MODEL.RESNETS.INPUT_RESOLUTION = 224
    cfg.MODEL.RESNETS.WIDTH = 64
    cfg.MODEL.RESNETS.PRETRAINED = None

    cfg.MODEL.GROUNDING = CN()
    cfg.MODEL.GROUNDING.ENABLED = True
    # Use dot product for grounding. This could be cosine or euclidean too.
    cfg.MODEL.GROUNDING.LOCAL_METRIC = "dot"
    # After aligning words to regions, sum the local distances to compute global distance.
    cfg.MODEL.GROUNDING.GLOBAL_METRIC = "aligned_local"
    # Use softmax to softly align each word to regions, and vice versa. 
    # This could be for instance hardmax, which aligns to the most similar
    cfg.MODEL.GROUNDING.ALIGNMENT = "softmax"
    # Typical good values are 100.0 for euclidean, 10.0 for dot, 0.01 for cosine
    cfg.MODEL.GROUNDING.ALIGNMENT_TEMPERATURE = 10.0
    # This loss is to choose the right caption out of all captions in the batch, 
    # And similarly choose the right image. Could be triplet loss instead.
    cfg.MODEL.GROUNDING.LOSS = "cross_entropy"
    # Whether to find a region for each word
    cfg.MODEL.GROUNDING.ALIGN_WORDS_TO_REGIONS = True
    # Whether to find a word for a region
    # At least one of these two should be True
    cfg.MODEL.GROUNDING.ALIGN_REGIONS_TO_WORDS = True
    cfg.MODEL.GROUNDING.NEGATIVE_MINING = "random"
    cfg.MODEL.GROUNDING.TRIPLET_MARGIN = 1.0
    cfg.MODEL.GROUNDING.LOSS_WEIGHT = 1.0
    cfg.MODEL.GROUNDING.GATHER = False

    cfg.INPUT.SCALE_RESIZE = CN()
    cfg.INPUT.SCALE_RESIZE.ENABLED = False
    cfg.INPUT.SCALE_RESIZE.SCALE_MIN = 0.8
    cfg.INPUT.SCALE_RESIZE.SCALE_MAX = 1.2

    cfg.RANK = CN()
    cfg.RANK.rank = [97, 125, 119, 116, 121, 0, 32, 96, 94, 129, 131, 100, 122, 102, 124, 84, 19, 103, 90, 91, 71, 81, 123, 24, 117, 115, 50, 22, 38, 132, 18, 126, 106, 80, 64, 104, 63, 56, 105, 37, 2, 20, 98, 92, 62, 51, 3, 46, 26, 82, 68, 49, 35, 33, 120, 13, 47, 25, 42, 118, 54, 101, 111, 17, 86, 74, 8, 28, 41, 14, 1, 75, 93, 130, 44, 40, 60, 58, 61, 45, 108, 113, 107, 30, 79, 87, 7, 88, 99, 31, 66, 5, 9, 34, 127, 16, 85, 43, 112, 4, 67, 23, 109, 89, 39, 11, 29, 53, 114, 27, 83, 6, 55, 128, 110, 57, 70, 76, 69, 48, 52, 77, 15, 73, 95, 59, 65, 12, 72, 36, 21, 10, 78]

    cfg.MODEL.FC_CLIP = CN()
    cfg.MODEL.FC_CLIP.CLIP_MODEL_NAME = "convnext_large_d_320"
    cfg.MODEL.FC_CLIP.CLIP_PRETRAINED_WEIGHTS = "laion2b_s29b_b131k_ft_soup"
    cfg.MODEL.FC_CLIP.EMBED_DIM = 768
    cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA = 0.4
    cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA = 0.8
    cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK = False
