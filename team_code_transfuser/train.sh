export WORK_DIR=/home/simon/Documents/Studium/8.Semester/Bachelorarbeit_Autonomes_Fahren/transfuser_github/transfuser

python3 train.py \
--batch_size=10 \
--logdir=${WORK_DIR}/results/training/training_log \
--parallel_training=0 \
--root_dir=${WORK_DIR}/results/training/training_dataset \
--load_file=${WORK_DIR}/model/model_seed1_39.pth
