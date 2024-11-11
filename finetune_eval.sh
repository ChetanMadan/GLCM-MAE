#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=6 python main_finetune_gbcu.py --eval \
 --resume /home/aarjav/scratch/v01_scratch4/mae/output_finetune_glcm_ssim_only_gbcu10fold/fold_9/checkpoint-80.pth \
 --model vit_base_patch16 --batch_size 16 \
 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_9 \
 --nb_classes 3


