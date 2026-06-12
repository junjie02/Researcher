set -x

# 缓存重定向：HF / ModelScope / pip / torch 全部下到数据盘
source "$(dirname "$0")/../env.sh"

# torchrun 同样需要 researcher 环境（verl 是该环境下 editable 安装）
CONDA_BASE_PATH="${CONDA_EXE%/bin/conda}"
if [ -z "$CONDA_BASE_PATH" ] || [ ! -f "$CONDA_BASE_PATH/etc/profile.d/conda.sh" ]; then
    CONDA_BASE_PATH="/root/miniconda3"
fi
source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
conda activate researcher

export WANDB_API_KEY="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" # replace wandb api key

nproc_per_node=$1

save_path="./checkpoints/o2searcher/coldstart/"
model_path="./Qwen2.5-3B-Instruct"
train_files="./o2searcher/data/coldstart/train.json"
val_files="./o2searcher/data/coldstart/test.json"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=8 \
    data.micro_batch_size=1 \
    data.max_length=10240 \
    data.truncation=right\
    model.partial_pretrain=$model_path \
    model.enable_gradient_checkpointing=True\
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.1\
    trainer.default_local_dir=$save_path \
    trainer.project_name=researcher \
    trainer.experiment_name=qwen2.5-3b-sft \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=2