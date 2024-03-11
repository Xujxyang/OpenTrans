#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
def median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid_index = n // 2  # 整除运算符，结果为整数
    if n % 2 == 1:  # 列表长度为奇数
        median_value = sorted_lst[mid_index]
    else:  # 列表长度为偶数
        median_value = (sorted_lst[mid_index - 1] + sorted_lst[mid_index]) / 2
    return median_value

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from typing import Any, Dict, List, Set

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    launch,
)
# from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    DETRPanopticCaptionDatasetMapper,
    DETRPanopticCaptionDesDatasetMapper,
    DETRPanopticCaptionSynDatasetMapper,
    DETRPanopticCaptionSynKDDatasetMapper,
    DETRPanopticCaptionNegDatasetMapper,
    DETRPanopticCaptionKDDatasetMapper,
    DatasetMapperVoca,
    PeriodicCheckpointerLastest,
)

from detectron2.data import build_detection_test_loader

logger = logging.getLogger("detectron2")


# def get_evaluator(cfg, dataset_name, output_folder=None):
#     """
#     Create evaluator(s) for a given dataset.
#     This uses the special metadata "evaluator_type" associated with each builtin dataset.
#     For your own dataset, you can simply create an evaluator manually in your
#     script and do not have to worry about the hacky if-else logic here.
#     """
#     if output_folder is None:
#         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#     evaluator_list = []
#     evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
#     if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
#         evaluator_list.append(
#             SemSegEvaluator(
#                 dataset_name,
#                 distributed=True,
#                 output_dir=output_folder,
#             )
#         )
#     if evaluator_type in ["coco", "coco_panoptic_seg"]:
#         evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
#     if evaluator_type == "coco_panoptic_seg":
#         evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
#     if evaluator_type == "cityscapes_instance":
#         return CityscapesInstanceEvaluator(dataset_name)
#     if evaluator_type == "cityscapes_sem_seg":
#         return CityscapesSemSegEvaluator(dataset_name)
#     if evaluator_type == "pascal_voc":
#         return PascalVOCDetectionEvaluator(dataset_name)
#     if evaluator_type == "lvis":
#         return LVISEvaluator(dataset_name, cfg, True, output_folder)
#     if len(evaluator_list) == 0:
#         raise NotImplementedError(
#             "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
#         )
#     if len(evaluator_list) == 1:
#         return evaluator_list[0]
#     return DatasetEvaluators(evaluator_list)


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    # semantic segmentation
    if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    # instance segmentation
    if evaluator_type == "coco":
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    # panoptic segmentation
    if evaluator_type in [
        "coco_panoptic_seg",
        "ade20k_panoptic_seg",
        "cityscapes_panoptic_seg",
        "mapillary_vistas_panoptic_seg",
    ]:
        if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    # COCO
    if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
        evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))        #这个需要
    # Mapillary Vistas
    if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
        evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
        evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
    # Cityscapes
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "cityscapes_panoptic_seg":
        if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
    # ADE20K
    if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
        evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
    # LVIS
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def build_test_loader(cfg, dataset_name):
    mapper = DatasetMapperVoca(cfg, False)
    return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def build_train_loader(cfg):
    # Semantic segmentation dataset mapper
    if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
        mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    # Panoptic segmentation dataset mapper
    elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
        mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    # Instance segmentation dataset mapper
    elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
        mapper = MaskFormerInstanceDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    # coco instance segmentation lsj new baseline
    elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
        mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    # coco panoptic segmentation lsj new baseline
    elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
        mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic_caption":
        mapper = DETRPanopticCaptionDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic_caption_syn":
        mapper = DETRPanopticCaptionSynDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic_caption_des":
        mapper = DETRPanopticCaptionDesDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic_caption_neg":
        mapper = DETRPanopticCaptionNegDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic_caption_kd":
        mapper = DETRPanopticCaptionKDDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic_caption_syn_kd":
        mapper = DETRPanopticCaptionSynKDDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    else:
        mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

def build_optimizer(cfg, model):
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

    defaults = {}
    defaults["lr"] = cfg.SOLVER.BASE_LR
    defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                # print(module_name)
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if "backbone" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
            if (
                "relative_position_bias_table" in module_param_name
                or "absolute_pos_embed" in module_param_name
            ):
                print(module_param_name)
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            params.append({"params": [value], **hyperparams})

    def maybe_add_full_model_gradient_clipping(optim):
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.SOLVER.BASE_LR
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_test_loader(cfg, dataset_name)
        evaluator = build_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=True):
    model.train()
    # print(comm.get_local_rank(), '369')
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    # print(comm.get_local_rank(), '375')
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    # import time
    # start_time = time.time()
    # layer selctivte!!!
    from utils import net_esd_estimator, get_module_names_shapes, get_hf_first_commit_date
    metrics = net_esd_estimator(
                model.module.backbone,
                EVALS_THRESH=0.00001,
                bins=100,
                # fix_fingers=None if args.fix_fingers=="DKS" else args.fix_fingers,
                # fix_fingers = 'xmin_mid',
                fix_fingers='xmin_peak',
                # xmin_pos=args.xmin_pos,
                # filter_zeros=args.filter_zeros=='True'
                filter_zeros=True,
            )
    # import pdb; pdb.set_trace()
    mid = median(metrics['alpha'])
    # k = math.floor((iteration/max_iter)*55*0.85)
    # kth_largest = np.partition(metrics['alpha'], k)[k]
    # import pdb; pdb.set_trace()
    frozen = []
    # frozen = metrics['longname'][-28:]
    for i in range(len(metrics['alpha'])):
        if metrics['alpha'][i] < mid:
            frozen.append(metrics['longname'][i])
    # import pdb; pdb.set_trace()
    print(frozen)
    # # for name, param in model.backbone.named_parameters():
    # #     param.requires_grad == False

    # import pdb; pdb.set_trace()
    for name, param in model.module.backbone.named_parameters():
        # import pdb; pdb.set_trace()
        for layer_name in frozen:
            if name.startswith(layer_name):
                param.requires_grad = False
                break
            else:
                param.requires_grad = True


    for name, param in model.module.backbone.named_parameters():
        if param.requires_grad == False:
            print(name)
    # import pdb; pdb.set_trace()

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    # import time
    # start_time = time.time()
    # total_backward_time = 0
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            # start_time3 = time.time()
            storage.iter = iteration
            # if iteration % 100 == 0:
            #     from utils import net_esd_estimator, get_module_names_shapes, get_hf_first_commit_date
            #     metrics = net_esd_estimator(
            #                 model.module.backbone,
            #                 EVALS_THRESH=0.00001,
            #                 bins=100,
            #                 # fix_fingers=None if args.fix_fingers=="DKS" else args.fix_fingers,
            #                 # fix_fingers = 'xmin_mid',
            #                 fix_fingers='xmin_peak',
            #                 # xmin_pos=args.xmin_pos,
            #                 # filter_zeros=args.filter_zeros=='True'
            #                 filter_zeros=True,
            #             )
            #     # import pdb; pdb.set_trace()
            #     mid = median(metrics['alpha'])
            #     # k = math.floor((iteration/max_iter)*55*0.85)
            #     # kth_largest = np.partition(metrics['alpha'], k)[k]
            #     # import pdb; pdb.set_trace()
            #     frozen = []
            #     # frozen = metrics['longname'][-28:]
            #     for i in range(len(metrics['alpha'])):
            #         if metrics['alpha'][i] < mid:
            #             frozen.append(metrics['longname'][i])
            #     # import pdb; pdb.set_trace()
            #     print(frozen)
            #     # for name, param in model.backbone.named_parameters():
            #     #     param.requires_grad == False

            #     # import pdb; pdb.set_trace()
            #     for name, param in model.module.backbone.named_parameters():
            #         # import pdb; pdb.set_trace()
            #         for layer_name in frozen:
            #             if name.startswith(layer_name):
            #                 param.requires_grad = False
            #                 break
            #             else:
            #                 param.requires_grad = True


            #     for name, param in model.module.backbone.named_parameters():
            #         if param.requires_grad == False:
            #             print(name)
            # # CKA
            # if iteration == 1:
            #     save_name_2 = 'original'
            #     checkpointer2 = DetectionCheckpointer(model, save_dir="original_clip")
            #     checkpointer2.save(save_name_2)

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            # start_time1 = time.time()
            optimizer.zero_grad()
            # start_time2 = time.time()
            losses.backward()
            # end_time2 = time.time()
            # print("backward time:")
            # one_backward_time = end_time2-start_time2
            # print(one_backward_time)
            # total_backward_time += one_backward_time
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            # end_time1 = time.time()
            # print("backward+optimizer time:")
            # print(end_time1-start_time1)

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
            # end_time3 = time.time()
            # print("one iter:")
            # print(end_time3-start_time3)
            # import pdb; pdb.set_trace()
    # print("total_backward_time")
    # print(total_backward_time)
    # end_time = time.time()
    # training_time = end_time - start_time
    # print(training_time)
    # training_time_str = str(training_time)
    # with open("time.txt", 'a') as file:
    #     file.write("freeze:" + training_time_str +"\n")



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def main(args):
    # print('111671')
    cfg = setup(args)
    # print('1111')
    model = build_model(cfg)
    # import torchvision
    # model = torchvision.models.resnet.resnet18()
    # model.to('cuda')
    # print('2222')
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = do_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
        )

    return do_train(cfg, model, resume=True)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
