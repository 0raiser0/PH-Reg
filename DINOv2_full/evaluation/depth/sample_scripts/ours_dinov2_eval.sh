conda activate env_cuda11

run_name="our-dinov2_full_reg"
model="dinov2_full"

# NYUv2
echo "start evalution on NYUv2"
config="evaluation/configs/vitb_nyu_linear_config.py"
workdir="./work_dirs_eval/othermodels/${run_name}/nyu/"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port 29511  \
                                    evaluate_dense_tasks_ours_model.py  \
                                    ${config} \
                                    --backbone-type $model \
                                    --task depth \
                                    --work-dir ${workdir} \