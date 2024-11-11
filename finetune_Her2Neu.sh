#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
python -m torch.distributed.launch --master_port=29500 --nproc_per_node=5 main_pretrain_2.py \
 --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
 --resume /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_decoder_8_gbcu10fold/checkpoint-199.pth  \
 --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch2/gbcu_data/ \
 --output_dir ./output_dir_glcm_ssim_only_gbcu10fold --log_dir ./output_dir_glcm_ssim_only_gbcu10fold

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch \
 --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_glcm_ssim_only_gbcu10fold/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_0/ \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_0 --log_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_0

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch \
 --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_glcm_ssim_only_gbcu10fold/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_1/ \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_1 --log_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_1

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch \
 --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_glcm_ssim_only_gbcu10fold/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_2/ \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_2 --log_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_2

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch \
 --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_glcm_ssim_only_gbcu10fold/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_3/ \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_3 --log_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_3

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch \
 --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_glcm_ssim_only_gbcu10fold/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_4/ \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_4 --log_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_4

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch \
 --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_glcm_ssim_only_gbcu10fold/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_5/ \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_5 --log_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_5

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch \
 --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_glcm_ssim_only_gbcu10fold/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_6/ \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_6 --log_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_6

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch \
 --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_glcm_ssim_only_gbcu10fold/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_7/ \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_7 --log_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_7

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch \
 --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_glcm_ssim_only_gbcu10fold/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_8/ \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_8 --log_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_8

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch \
 --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_glcm_ssim_only_gbcu10fold/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_9/ \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_9 --log_dir ./output_finetune_glcm_ssim_only_gbcu10fold/fold_9
