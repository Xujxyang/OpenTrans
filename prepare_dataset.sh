export DETECTRON2_DATASETS=$(realpath data)
cd data
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/cityscapes
unzip -q cityscapes/gtFine_trainvaltest.zip -d cityscapes/
unzip -oq cityscapes/leftImg8bit_trainvaltest.zip -d cityscapes/
unzip -q ../package/cityscapesScripts-master.zip
cd cityscapesScripts-master/
pip install --user -e ./
CITYSCAPES_DATASET=$(realpath ../cityscapes) python3 cityscapesscripts/preparation/createTrainIdLabelImgs.py
CITYSCAPES_DATASET=$(realpath ../cityscapes) python3 cityscapesscripts/preparation/createPanopticImgs.py --set-names val
cd ..

# ade20k 150
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/ade/ADEChallengeData2016.zip
unzip -q ADEChallengeData2016.zip
cd ..
DETECTRON2_DATASETS=$(realpath data) python3 datasets/prepare_ade20k_sem_seg.py 

# pascal context
cd data
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/data/VOCtrainval_03-May-2010.tar
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/data/detail-api-master.zip
tar -xf VOCtrainval_03-May-2010.tar
unzip detail-api-master.zip
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/data/trainval_merged.json VOCdevkit/VOC2010/
cd detail-api-master/PythonAPI
python3 setup.py build_ext --inplace
rm -rf build
python3 setup.py build_ext install
rm -rf build
cd ../../..
python3 datasets/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json

# ade20k full
cd data
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/data/kunyang36_17172dc7.zip
unzip -q kunyang36_17172dc7.zip
cd ..
python3 datasets/prepare_ade20k_full_sem_seg.py

# pascal contest full
cd data/VOCdevkit/VOC2010
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/kunyanghan/data/trainval.tar.gz
tar -xzf trainval.tar.gz
cd ../../..
python3 datasets/pascal_context_mat2png.py
