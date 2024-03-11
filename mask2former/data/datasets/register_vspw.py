# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

VSPW_CATEGORIES = [
    {"id": 0, "name": "others"},
    {"id": 1, "name": "wall"},
    {"id": 2, "name": "ceiling"},
    {"id": 3, "name": "door"},
    {"id": 4, "name": "stair"},
    {"id": 5, "name": "ladder"},
    {"id": 6, "name": "escalator"},
    {"id": 7, "name": "Playground_slide"},
    {"id": 8, "name": "handrail_or_fence"},
    {"id": 9, "name": "window"},
    {"id": 10, "name": "rail"},
    {"id": 11, "name": "goal"},
    {"id": 12, "name": "pillar"},
    {"id": 13, "name": "pole"},
    {"id": 14, "name": "floor"},
    {"id": 15, "name": "ground"},
    {"id": 16, "name": "grass"},
    {"id": 17, "name": "sand"},
    {"id": 18, "name": "athletic_field"},
    {"id": 19, "name": "road"},
    {"id": 20, "name": "path"},
    {"id": 21, "name": "crosswalk"},
    {"id": 22, "name": "building"},
    {"id": 23, "name": "house"},
    {"id": 24, "name": "bridge"},
    {"id": 25, "name": "tower"},
    {"id": 26, "name": "windmill"},
    {"id": 27, "name": "well_or_well_lid"},
    {"id": 28, "name": "other_construction"},
    {"id": 29, "name": "sky"},
    {"id": 30, "name": "mountain"},
    {"id": 31, "name": "stone"},
    {"id": 32, "name": "wood"},
    {"id": 33, "name": "ice"},
    {"id": 34, "name": "snowfield"},
    {"id": 35, "name": "grandstand"},
    {"id": 36, "name": "sea"},
    {"id": 37, "name": "river"},
    {"id": 38, "name": "lake"},
    {"id": 39, "name": "waterfall"},
    {"id": 40, "name": "water"},
    {"id": 41, "name": "billboard_or_Bulletin_Board"},
    {"id": 42, "name": "sculpture"},
    {"id": 43, "name": "pipeline"},
    {"id": 44, "name": "flag"},
    {"id": 45, "name": "parasol_or_umbrella"},
    {"id": 46, "name": "cushion_or_carpet"},
    {"id": 47, "name": "tent"},
    {"id": 48, "name": "roadblock"},
    {"id": 49, "name": "car"},
    {"id": 50, "name": "bus"},
    {"id": 51, "name": "truck"},
    {"id": 52, "name": "bicycle"},
    {"id": 53, "name": "motorcycle"},
    {"id": 54, "name": "wheeled_machine"},
    {"id": 55, "name": "ship_or_boat"},
    {"id": 56, "name": "raft"},
    {"id": 57, "name": "airplane"},
    {"id": 58, "name": "tyre"},
    {"id": 59, "name": "traffic_light"},
    {"id": 60, "name": "lamp"},
    {"id": 61, "name": "person"},
    {"id": 62, "name": "cat"},
    {"id": 63, "name": "dog"},
    {"id": 64, "name": "horse"},
    {"id": 65, "name": "cattle"},
    {"id": 66, "name": "other_animal"},
    {"id": 67, "name": "tree"},
    {"id": 68, "name": "flower"},
    {"id": 69, "name": "other_plant"},
    {"id": 70, "name": "toy"},
    {"id": 71, "name": "ball_net"},
    {"id": 72, "name": "backboard"},
    {"id": 73, "name": "skateboard"},
    {"id": 74, "name": "bat"},
    {"id": 75, "name": "ball"},
    {"id": 76, "name": "cupboard_or_showcase_or_storage_rack"},
    {"id": 77, "name": "box"},
    {"id": 78, "name": "traveling_case_or_trolley_case"},
    {"id": 79, "name": "basket"},
    {"id": 80, "name": "bag_or_package"},
    {"id": 81, "name": "trash_can"},
    {"id": 82, "name": "cage"},
    {"id": 83, "name": "plate"},
    {"id": 84, "name": "tub_or_bowl_or_pot"},
    {"id": 85, "name": "bottle_or_cup"},
    {"id": 86, "name": "barrel"},
    {"id": 87, "name": "fishbowl"},
    {"id": 88, "name": "bed"},
    {"id": 89, "name": "pillow"},
    {"id": 90, "name": "table_or_desk"},
    {"id": 91, "name": "chair_or_seat"},
    {"id": 92, "name": "bench"},
    {"id": 93, "name": "sofa"},
    {"id": 94, "name": "shelf"},
    {"id": 95, "name": "bathtub"},
    {"id": 96, "name": "gun"},
    {"id": 97, "name": "commode"},
    {"id": 98, "name": "roaster"},
    {"id": 99, "name": "other_machine"},
    {"id": 100, "name": "refrigerator"},
    {"id": 101, "name": "washing_machine"},
    {"id": 102, "name": "Microwave_oven"},
    {"id": 103, "name": "fan"},
    {"id": 104, "name": "curtain"},
    {"id": 105, "name": "textiles"},
    {"id": 106, "name": "clothes"},
    {"id": 107, "name": "painting_or_poster"},
    {"id": 108, "name": "mirror"},
    {"id": 109, "name": "flower_pot_or_vase"},
    {"id": 110, "name": "clock"},
    {"id": 111, "name": "book"},
    {"id": 112, "name": "tool"},
    {"id": 113, "name": "blackboard"},
    {"id": 114, "name": "tissue"},
    {"id": 115, "name": "screen_or_television"},
    {"id": 116, "name": "computer"},
    {"id": 117, "name": "printer"},
    {"id": 118, "name": "Mobile_phone"},
    {"id": 119, "name": "keyboard"},
    {"id": 120, "name": "other_electronic_product"},
    {"id": 121, "name": "fruit"},
    {"id": 122, "name": "food"},
    {"id": 123, "name": "instrument"},
    {"id": 124, "name": "train"},
]


def _get_vspw_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in VSPW_CATEGORIES]
    assert len(stuff_ids) == 125, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in VSPW_CATEGORIES[1:]]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_vspw(root):
    # root = os.path.join(root, "VSPW")
    meta = _get_vspw_meta()
    # for name, image_dirname, sem_seg_dirname in [
    #     ("train", "images/training", "annotations/training"),
    #     ("test", "images/validation", "annotations/validation"),
    # ]:
    # for name, dirname in [("train", "train"), ("val", "val")]:
    #     image_dir = os.path.join(root, dirname, 'imgs')
    #     gt_dir = os.path.join(root, dirname, 'masks')
    #     name = f"vspw_{name}"
    for name, image_dirname, sem_seg_dirname in [
        ("val", "VSPW_minival", "VSPW_minival"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"vspw_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )
    # for name, image_dirname, sem_seg_dirname in [
    #     ("val", "VSPW_minival", "VSPW_minival"),
    # ]:
    #     image_dir = os.path.join(root, image_dirname)
    #     gt_dir = os.path.join(root, sem_seg_dirname)
    #     name = f"vspw_{name}_sem_seg"
    #     DatasetCatalog.register(
    #         name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
    #     )
    #     MetadataCatalog.get(name).set(
    #         image_root=image_dir,
    #         sem_seg_root=gt_dir,
    #         evaluator_type="sem_seg",
    #         ignore_label=255,
    #         **meta,
    #     )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_vspw(_root)