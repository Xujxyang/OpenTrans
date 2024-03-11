#!/bin/bash
cd $(dirname $0)
# pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
# pip install --user git+https://github.com/cocodataset/panopticapi.git
# pip install scipy
# python3 nltk_install.py
cd mask2former/modeling/pixel_decoder/ops
sudo chmod 777 /usr/local/lib/python3.7/dist-packages/
python3 setup.py build install
cd ../../../..

hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/package
cd package
pip install scipy-1.7.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
unzip -q panopticapi-master.zip
pip install -e panopticapi-master
# mv nltk_data.tar.gz /home/tiger/
tar -xzf nltk_data.tar.gz -C /home/tiger/
cd ..

mkdir -p data/coco
cd data/coco
export DETECTRON2_DATASETS=$(realpath ..)
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/coco_data/*
# curl "http://images.cocodataset.org/zips/train2017.zip" --output train2017.zip
# curl "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" --output annotations_trainval2017.zip
# curl "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip" --output panoptic_annotations_trainval2017.zip
# curl "http://images.cocodataset.org/zips/val2017.zip" --output val2017.zip
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip
unzip -q panoptic_annotations_trainval2017.zip
unzip -q annotations/panoptic_train2017.zip
unzip -q annotations/panoptic_val2017.zip
tar -xzf zero_shot_annotation.tar.gz
# unzip -q stuffthingmaps_trainval2017.zip -d stuffthingmaps
cd ../..

# python3 datasets/prepare_coco_stuff_164k_sem_seg.py data/coco/
python3 datasets/prepare_coco_semantic_annos_from_panoptic_annos.py

hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/voca/RN50.pt pretrained/
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/voca/RN101.pt pretrained/
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/voca/ViT-B-16.pt pretrained/
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/voca/R-101.pkl pretrained/
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/voca/R-50.pkl pretrained/

sh prepare_dataset.sh &
# python3 train_net.py --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml --num-gpus 8
# python3 train_net.py --config-file configs/coco/panoptic-segmentation/$1 --num-gpus ${@:3}

export BYTED_TORCH_FX='O0'
export OMP_NUM_THREADS='1'

resume=$3
echo 
if test ${#resume} -gt 6
then
    mkdir output
    cd output
    hdfs dfs -get $resume model_resume.pth
    echo "model_resume.pth" > last_checkpoint
    cd ..
    python3 train_net.py --resume --num-gpus $ARNOLD_WORKER_GPU --config-file configs/coco/panoptic-segmentation/$1 ${@:4} 
else
    python3 train_net.py --num-gpus $ARNOLD_WORKER_GPU --config-file configs/coco/panoptic-segmentation/$1 ${@:4} 
fi
# torchrun --nproc_per_node=$ARNOLD_WORKER_GPU --nnodes=$ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr=$METIS_WORKER_0_HOST --master_port=40000 train_net.py --config-file configs/coco/panoptic-segmentation/$1 ${@:3} 
hdfs dfs -put -f output/model_lastest.pth hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/voca/$2

# python3 train_net.py --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml --eval-only MODEL.WEIGHTS output/model_lastest.pth
python3 train_net.py --config-file configs/coco/panoptic-segmentation/$1 --eval-only DATASETS.TEST '("cityscapes_fine_panoptic_val", "coco_2017_val_panoptic_caption", "ade20k_sem_seg_val", "ade20k_full_sem_seg_val", "pc_val_sem_seg", "pc_full_val_sem_seg","openvocab_pascal20_sem_seg_val",)' MODEL.META_ARCHITECTURE OpenSegMaskFormer MODEL.WEIGHTS output/model_lastest.pth

# python3 train_net.py --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml --num-gpus 8
# MODEL.PER_REGION.DROPOUT 0.75 SOLVER.IMS_PER_BATCH 56
# python3 train_net.py --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca.yaml --num-gpus 8
# sleep infinity

# SOLVER.IMS_PER_BATCH 2 DATALOADER.NUM_WORKERS 0
# cd ky_open_voca/
# export DETECTRON2_DATASETS='/opt/tiger/debug/ky_open_voca/data'
# CUDA_VISIBLE_DEVICES=
python3 train_net.py --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion_syn.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKDSyn OUTPUT_DIR test_1 MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_syn_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.PER_REGION.SYN_CLASS.LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 2.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.PER_REGION.SYN_CLASS.ENABLED True MODEL.KD.ENABLED True
python3 train_net.py --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKD OUTPUT_DIR test_1 MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 2.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.KD.ENABLED True
python3 train_net.py --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion_syn.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKDSyn OUTPUT_DIR test_1 MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_syn_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.PER_REGION.SYN_CLASS.LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 2.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.PER_REGION.SYN_CLASS.ENABLED False MODEL.KD.ENABLED True
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml --eval-only DATASETS.TEST '("pc_full_val_sem_seg",)' MODEL.META_ARCHITECTURE OpenSegMaskFormer MODEL.WEIGHTS pruned_nokd_text/seg_clip_imp10.pth
CUDA_VISIBLE_DEVICES=1 python3 plain_train_net.py --resume --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKD OUTPUT_DIR rebuttla_clip MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 2.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.KD.ENABLED True 
CUDA_VISIBLE_DEVICES=3 python3 plain_train_net_imp.py --resume --num-gpus 1 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKD OUTPUT_DIR rebuttal_test MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 2.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.KD.ENABLED False MODEL.WEIGHTS ratio_0.7_iter_1000/seg_clip_imp2.pth
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 plain_train_net_imp_entropy.py --resume --num-gpus 1 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKD OUTPUT_DIR test_imp_2 MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 0.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.KD.ENABLED False MODEL.WEIGHTS pruned19/seg_clip_imp3.pth
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 1 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml --eval-only DATASETS.TEST '("coco_2017_val_panoptic_caption",)' MODEL.META_ARCHITECTURE OpenSegMaskFormer MODEL.WEIGHTS pruned_nokd_text/seg_clip_imp10.pth
CUDA_VISIBLE_DEVICES=0 python3 caculate_periamge_entropy.py --num-gpus 1 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKDCaculateEntropy OUTPUT_DIR test_entropy MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 0.0 MODEL.KD.KD_WEIGHT 0.0 MODEL.GROUNDING.LOSS_WEIGHT 0.0 MODEL.KD.ENABLED False MODEL.WEIGHTS test_plain_3/model_0049999.pth
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 plain_train_net_per_image_entropy.py --resume --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKD OUTPUT_DIR test_per_image MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 0.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.KD.ENABLED False MODEL.WEIGHTS pruned19/seg_clip_imp3.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_DISTRIBUTED_DEBUG=DETAIL python3 plain_train_net_selective.py --resume --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKD OUTPUT_DIR test_selective_2 MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 0.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.KD.ENABLED False MODEL.WEIGHTS pruned_nokd_text/seg_clip_imp2.pth
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 plain_train_net.py --resume --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKDLora OUTPUT_DIR lora_TEST MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 2.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.KD.ENABLED True MODEL.WEIGHTS test_plain_3/model_0009999.pth
python3 train_net.py --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml --eval-only DATASETS.TEST '("cityscapes_fine_panoptic_val",)' MODEL.META_ARCHITECTURE OpenSegMaskFormer MODEL.WEIGHTS selective_0.25/model_0049999.pth

# Deeplab without prune
CUDA_VISIBLE_DEVICES=1 python3 plain_train_net_deeplab.py --resume --num-gpus 4 --config-file configs/coco/panoptic-segmentation/Deeplabv3_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegDeeplabv3Text OUTPUT_DIR test MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME deeplab_panoptic_caption_syn_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.WEIGHTS pruned_base/seg_clip_imp22.pth