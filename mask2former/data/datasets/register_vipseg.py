import os
import json
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

__all__ = [""]

VIPSeg_CATEGORIES = [
    { "id": 0, "name": "wall", "isthing": 0,},
    { "id": 1, "name": "ceiling", "isthing": 0,},
    { "id": 2, "name": "door", "isthing": 1,},
    { "id": 3, "name": "stair", "isthing": 0,},
    { "id": 4, "name": "ladder", "isthing": 1,},
    { "id": 5, "name": "escalator", "isthing": 0,},
    { "id": 6, "name": "Playground_slide", "isthing": 0,},
    { "id": 7, "name": "handrail_or_fence", "isthing": 0,},
    { "id": 8, "name": "window", "isthing": 1,},
    { "id": 9, "name": "rail", "isthing": 0,},
    { "id": 10, "name": "goal", "isthing": 1,},
    { "id": 11, "name": "pillar", "isthing": 0,},
    { "id": 12, "name": "pole", "isthing": 0,},
    { "id": 13, "name": "floor", "isthing": 0,},
    { "id": 14, "name": "ground", "isthing": 0,},
    { "id": 15, "name": "grass", "isthing": 0,},
    { "id": 16, "name": "sand", "isthing": 0,},
    { "id": 17, "name": "athletic_field", "isthing": 0,},
    { "id": 18, "name": "road", "isthing": 0,},
    { "id": 19, "name": "path", "isthing": 0,},
    { "id": 20, "name": "crosswalk", "isthing": 0,},
    { "id": 21, "name": "building", "isthing": 0,},
    { "id": 22, "name": "house", "isthing": 0,},
    { "id": 23, "name": "bridge", "isthing": 0,},
    { "id": 24, "name": "tower", "isthing": 0,},
    { "id": 25, "name": "windmill", "isthing": 0,},
    { "id": 26, "name": "well_or_well_lid", "isthing": 0,},
    { "id": 27, "name": "other_construction", "isthing": 0,},
    { "id": 28, "name": "sky", "isthing": 0,},
    { "id": 29, "name": "mountain", "isthing": 0,},
    { "id": 30, "name": "stone", "isthing": 0,},
    { "id": 31, "name": "wood", "isthing": 0,},
    { "id": 32, "name": "ice", "isthing": 0,},
    { "id": 33, "name": "snowfield", "isthing": 0,},
    { "id": 34, "name": "grandstand", "isthing": 0,},
    { "id": 35, "name": "sea", "isthing": 0,},
    { "id": 36, "name": "river", "isthing": 0,},
    { "id": 37, "name": "lake", "isthing": 0,},
    { "id": 38, "name": "waterfall", "isthing": 0,},
    { "id": 39, "name": "water", "isthing": 0,},
    { "id": 40, "name": "billboard_or_Bulletin_Board", "isthing": 0,},
    { "id": 41, "name": "sculpture", "isthing": 1,},
    { "id": 42, "name": "pipeline", "isthing": 0,},
    { "id": 43, "name": "flag", "isthing": 1,},
    { "id": 44, "name": "parasol_or_umbrella", "isthing": 1,},
    { "id": 45, "name": "cushion_or_carpet", "isthing": 0,},
    { "id": 46, "name": "tent", "isthing": 1,},
    { "id": 47, "name": "roadblock", "isthing": 1,},
    { "id": 48, "name": "car", "isthing": 1,},
    { "id": 49, "name": "bus", "isthing": 1,},
    { "id": 50, "name": "truck", "isthing": 1,},
    { "id": 51, "name": "bicycle", "isthing": 1,},
    { "id": 52, "name": "motorcycle", "isthing": 1,},
    { "id": 53, "name": "wheeled_machine", "isthing": 0,},
    { "id": 54, "name": "ship_or_boat", "isthing": 1,},
    { "id": 55, "name": "raft", "isthing": 1,},
    { "id": 56, "name": "airplane", "isthing": 1,},
    { "id": 57, "name": "tyre", "isthing": 0,},
    { "id": 58, "name": "traffic_light", "isthing": 0,},
    { "id": 59, "name": "lamp", "isthing": 0,},
    { "id": 60, "name": "person", "isthing": 1,},
    { "id": 61, "name": "cat", "isthing": 1,},
    { "id": 62, "name": "dog", "isthing": 1,},
    { "id": 63, "name": "horse", "isthing": 1,},
    { "id": 64, "name": "cattle", "isthing": 1,},
    { "id": 65, "name": "other_animal", "isthing": 1,},
    { "id": 66, "name": "tree", "isthing": 0,},
    { "id": 67, "name": "flower", "isthing": 0,},
    { "id": 68, "name": "other_plant", "isthing": 0,},
    { "id": 69, "name": "toy", "isthing": 0,},
    { "id": 70, "name": "ball_net", "isthing": 0,},
    { "id": 71, "name": "backboard", "isthing": 0,},
    { "id": 72, "name": "skateboard", "isthing": 1,},
    { "id": 73, "name": "bat", "isthing": 0,},
    { "id": 74, "name": "ball", "isthing": 1,},
    { "id": 75, "name": "cupboard_or_showcase_or_storage_rack", "isthing": 0,},
    { "id": 76, "name": "box", "isthing": 1,},
    { "id": 77, "name": "traveling_case_or_trolley_case", "isthing": 1,},
    { "id": 78, "name": "basket", "isthing": 1,},
    { "id": 79, "name": "bag_or_package", "isthing": 1,},
    { "id": 80, "name": "trash_can", "isthing": 0,},
    { "id": 81, "name": "cage", "isthing": 0,},
    { "id": 82, "name": "plate", "isthing": 1,},
    { "id": 83, "name": "tub_or_bowl_or_pot", "isthing": 1,},
    { "id": 84, "name": "bottle_or_cup", "isthing": 1,},
    { "id": 85, "name": "barrel", "isthing": 1,},
    { "id": 86, "name": "fishbowl", "isthing": 1,},
    { "id": 87, "name": "bed", "isthing": 1,},
    { "id": 88, "name": "pillow", "isthing": 1,},
    { "id": 89, "name": "table_or_desk", "isthing": 1,},
    { "id": 90, "name": "chair_or_seat", "isthing": 1,},
    { "id": 91, "name": "bench", "isthing": 1,},
    { "id": 92, "name": "sofa", "isthing": 1,},
    { "id": 93, "name": "shelf", "isthing": 0,},
    { "id": 94, "name": "bathtub", "isthing": 0,},
    { "id": 95, "name": "gun", "isthing": 1,},
    { "id": 96, "name": "commode", "isthing": 1,},
    { "id": 97, "name": "roaster", "isthing": 1,},
    { "id": 98, "name": "other_machine", "isthing": 0,},
    { "id": 99, "name": "refrigerator", "isthing": 1,},
    { "id": 100, "name": "washing_machine", "isthing": 1,},
    { "id": 101, "name": "Microwave_oven", "isthing": 1,},
    { "id": 102, "name": "fan", "isthing": 1,},
    { "id": 103, "name": "curtain", "isthing": 0,},
    { "id": 104, "name": "textiles", "isthing": 0,},
    { "id": 105, "name": "clothes", "isthing": 0,},
    { "id": 106, "name": "painting_or_poster", "isthing": 1,},
    { "id": 107, "name": "mirror", "isthing": 1,},
    { "id": 108, "name": "flower_pot_or_vase", "isthing": 1,},
    { "id": 109, "name": "clock", "isthing": 1,},
    { "id": 110, "name": "book", "isthing": 0,},
    { "id": 111, "name": "tool", "isthing": 0,},
    { "id": 112, "name": "blackboard", "isthing": 0,},
    { "id": 113, "name": "tissue", "isthing": 0,},
    { "id": 114, "name": "screen_or_television", "isthing": 1,},
    { "id": 115, "name": "computer", "isthing": 1,},
    { "id": 116, "name": "printer", "isthing": 1,},
    { "id": 117, "name": "Mobile_phone", "isthing": 1,},
    { "id": 118, "name": "keyboard", "isthing": 1,},
    { "id": 119, "name": "other_electronic_product", "isthing": 0,},
    { "id": 120, "name": "fruit", "isthing": 0,},
    { "id": 121, "name": "food", "isthing": 0,},
    { "id": 122, "name": "instrument", "isthing": 1,},
    { "id": 123, "name": "train", "isthing": 1,}
]

def load_vipseg_json(json_file, data_root, panoptic_root, sem_root=None, dataset_name=None, meta=None):
    with open(json_file) as f:
        json_file = json.load(f)

    def _convert_category_id(segment_info, meta):
        segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][segment_info["category_id"]]

        return segment_info

    dataset_dicts = []
    for video in json_file["annotations"]:
        video_id = video["video_id"]
        # video_anno = video["annotations"]

        file_name_list, frame_objs = [], []
        for frame in video["annotations"]:
            record = {}
            record["video_id"] = video_id
            frame_id = frame["image_id"]
            file_name = os.path.join(data_root, video_id, frame_id + ".jpg")
            file_name_list.append(file_name)

            _frame_obj = {
                "file_name": file_name,
                "image_id": frame_id,
                "segments_info": [_convert_category_id(x, meta) for x in frame["segments_info"]],
                "pan_seg_file_name": os.path.join(panoptic_root, video_id, frame_id + ".png")
            }
            if sem_root is not None:
                sem_seg_file_name = os.path.join(sem_root, video_id, frame_id + ".png")
                _frame_obj['sem_seg_file_name'] = sem_seg_file_name

            # frame_objs.append(_frame_obj)

            record["dataset"] = dataset_name
            # record["file_names"] = file_name_list
            # record["annotations"] = frame_objs
            record.update(_frame_obj)

            dataset_dicts.append(record)
    
    return dataset_dicts

def _get_vipseg_meta(split='base'):
    novel_clsID = [78, 94, 112, 76, 102, 32, 122, 116, 80, 57, 34, 11]
    if split == 'novel':
        VIPSeg_SPLIT_CATEGORIES = [k for k in VIPSeg_CATEGORIES if k["id"] in novel_clsID]
    elif split == 'base':
        VIPSeg_SPLIT_CATEGORIES = [k for k in VIPSeg_CATEGORIES if k["id"] not in novel_clsID]
    else:
        VIPSeg_SPLIT_CATEGORIES = VIPSeg_CATEGORIES

    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in VIPSeg_SPLIT_CATEGORIES]
    # assert len(stuff_ids) == 124, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    # stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_dataset_id_to_contiguous_id = {}
    dataset_contiguous_id2name = {}
    for i, cat in enumerate(VIPSeg_SPLIT_CATEGORIES):
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i
        dataset_contiguous_id2name[i] = cat["name"]

    stuff_classes = [k["name"] for k in VIPSeg_SPLIT_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "dataset_contiguous_id2name": dataset_contiguous_id2name,
    }
    return ret

_PREDEFINED_SPLITS_VIPSEG_PANOPTIC_CAPTION = {
    'vipseg_train_base_ignore_image' : (
        'VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_train_base_ignore.json',
        'VIPSeg/VIPSeg_720P/images',
        'VIPSeg/VIPSeg_720P/panomasksRGB',
        None,
        "base",
    ),
    'vipseg_train_base_delete_image' : (
        'VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_train_base_delete.json',
        'VIPSeg/VIPSeg_720P/images',
        'VIPSeg/VIPSeg_720P/panomasksRGB',
        None,
        "base",
    ),
    'vipseg_val_base_image' : (
        'VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val_base.json',
        'VIPSeg/VIPSeg_720P/images',
        'VIPSeg/VIPSeg_720P/panomasksRGB',
        'VIPSeg/VIPSeg_720P/panoptic_semseg_val_base',
        "base",
    ),
    'vipseg_val_novel_image' : (
        'VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val_novel.json',
        'VIPSeg/VIPSeg_720P/images',
        'VIPSeg/VIPSeg_720P/panomasksRGB',
        'VIPSeg/VIPSeg_720P/panoptic_semseg_val_novel',
        "novel",
    ),
    'vipseg_val_image' : (
        'VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json',
        'VIPSeg/VIPSeg_720P/images',
        'VIPSeg/VIPSeg_720P/panomasksRGB',
        'VIPSeg/VIPSeg_720P/panoptic_semseg_val',
        "all",
    )
}

def register_vipseg(dataset_name, anno_path, image_root, pan_root, sem_root, metadata):
    DatasetCatalog.register(
        dataset_name,
        lambda: load_vipseg_json(anno_path, image_root, pan_root, sem_root, dataset_name, metadata),
    )
    MetadataCatalog.get(dataset_name).set(
        panoptic_root=pan_root,
        image_root=image_root,
        panoptic_json=anno_path,
        evaluator_type='sem_seg',
        ignore_label=255,
        **metadata,
    )

def register_all_vipseg(root):
    for (
        dataset_name,
        (anno_path, image_root, pan_root, sem_root, split)
    ) in _PREDEFINED_SPLITS_VIPSEG_PANOPTIC_CAPTION.items():
        register_vipseg(
            dataset_name,
            os.path.join(root, anno_path),
            os.path.join(root, image_root),
            os.path.join(root, pan_root),
            os.path.join(root, sem_root) if sem_root is not None else None,
            _get_vipseg_meta(split),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_vipseg(_root)