#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,2 \
python -m torch.distributed.launch --master_port=29512 --nproc_per_node=2 main_pretrain.py \
 --batch_size 8  --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 300 --warmup_epochs 40 \
 --resume /home/aarjav/scratch/mae/output_dir_rgb_train_covid/checkpoint-199.pth  \
 --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/COVID-imagenet/train   \
 --output_dir ./output_dir_glcm_covid_ssim --log_dir ./output_dir_glcm_covid_ssim
