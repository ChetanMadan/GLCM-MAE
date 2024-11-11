# !/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
 --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 64  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_ssim_glcm_busi5fold_fold_0/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch4/busi5fold/fold_0 \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_busi5fold_fold_0 --log_dir ./output_finetune_rgb_ssim_glcm_busi5fold_fold_0

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
 --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 64  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_ssim_glcm_busi5fold_fold_1/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch4/busi5fold/fold_1 \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_busi5fold_fold_1 --log_dir ./output_finetune_rgb_ssim_glcm_busi5fold_fold_1

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
 --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 64  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_ssim_glcm_busi5fold_fold_2/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch4/busi5fold/fold_2 \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_busi5fold_fold_2 --log_dir ./output_finetune_rgb_ssim_glcm_busi5fold_fold_2

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
 --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 64  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_ssim_glcm_busi5fold_fold_3/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch4/busi5fold/fold_3 \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_busi5fold_fold_3 --log_dir ./output_finetune_rgb_ssim_glcm_busi5fold_fold_3

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
 --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 64  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_ssim_glcm_busi5fold_fold_4/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch4/busi5fold/fold_4 \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_busi5fold_fold_4 --log_dir ./output_finetune_rgb_ssim_glcm_busi5fold_fold_4
