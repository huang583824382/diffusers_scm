export MODEL_NAME="/mnt/afs_james/zhiwei/latent-consistency-model/stabilityai/stable-diffusion-xl-base-1.0"
# export DATASET_NAME="lambdalabs/naruto-blip-captions"
# export DATASET_NAME="pixparse/cc12m-wds"
export VAE_PATH="/mnt/afs_james/zhiwei/latent-consistency-model/madebyollin/sdxl-vae-fp16-fix"
export OUTPUT_DIR="narutos-lora-lcm-sdxl"
export HF_ENDPOINT="https://hf-mirror.com"

accelerate launch train_lcm_distill_lora_sdxl_wds.py \
  --pretrained_teacher_model=${MODEL_NAME} \
  --pretrained_vae_model_name_or_path=${VAE_PATH} \
  --output_dir="CC12M-lora-scm-sdxl-wds" \
  --mixed_precision=fp16 \
  --resolution=1024 \
  --lora_rank=64 \
  --learning_rate=1e-4 --loss_type="huber" --use_fix_crop_and_size --adam_weight_decay=0.0 \
  --max_train_steps=1000 \
  --max_train_samples=4000000 \
  --dataloader_num_workers=8 \
  --train_shards_path_or_url="pipe:curl -L -s https://hf-mirror.com/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01099}.tar?download=true" \
  --validation_steps=200 \
  --checkpointing_steps=200 --checkpoints_total_limit=10 \
  --train_batch_size=12 \
  --gradient_checkpointing --enable_xformers_memory_efficient_attention \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --resume_from_checkpoint=latest \
  --report_to=wandb \
  --seed=453645634 
