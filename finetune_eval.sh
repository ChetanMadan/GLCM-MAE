#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=6 python main_finetune_gbcu.py --eval \
#  --resume /home/aarjav/scratch/v01_scratch5/mae/output_finetune_rgb_glcm_ssim_busi5fold_final/fold_4/checkpoint-55.pth \
#  --model vit_base_patch16 --batch_size 16 \
#  --data_path /home/aarjav/scratch/busi5fold_final/fold_4 \
#  --nb_classes 3

CUDA_VISIBLE_DEVICES=5 python main_finetune.py --eval \
 --resume /home/aarjav/scratch/mae/output_finetune_Her2neu_Grounding_DINO_mae/fold_4/checkpoint-76.pth \
 --model vit_base_patch16 --batch_size 16 \
 --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs/imgs \
 --fold fold_4 \
 --nb_classes 2


