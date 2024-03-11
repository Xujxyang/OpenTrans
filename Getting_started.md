When you have prepared the environment and dataset according to the description of fc-clip, you also need to configure the method for running the selective layer based on the requirements we provide.\
Let's get started!\
Please use the following command to load the data path:\
export DETECTRON2_DATASETS= ******\
Then, you can run our pruning method using the following statement:\
python3 plain_train_net_imp.py --resume --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKD OUTPUT_DIR rebuttal_test MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 2.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.KD.ENABLED True\
If it is the first step of training and you need to reference mask2former/maskformer_model_voca_dense_kd.py or use the command line to disable other unnecessary losses and only use kd_loss.\
You can use the following command to perform layer-wise training:\
python3 plain_train_net.py --resume --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKD OUTPUT_DIR rebuttla_clip MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 2.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.KD.ENABLED True\
If you want to activate both strategies simultaneously, you can enable layer-wise in plain_train_net_imp.py:\
The specific algorithms for these two strategies can be found in the pruning.py and utils.py files. You can adjust them according to your specific requirements.\
