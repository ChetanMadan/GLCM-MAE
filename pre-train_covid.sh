#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
 --batch_size 8  --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
 --resume /home/aarjav/scratch/mae/output_dir_rgb_train_covid/checkpoint-199.pth  \
 --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/COVID-imagenet/train/ \
 --output_dir ./output_dir_glcm_ssim_rgb_covid --log_dir ./output_dir_glcm_ssim_rgb_covid_new2