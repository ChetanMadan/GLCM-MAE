# !/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain.py \
#  --batch_size 128 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch8/mae/MAE_pth/pretained_ViT_base/mae_pretrain_vit_base_full.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch8/gbcu_data/ \
#  --mask_ratio 0.70 \
#  --output_dir ./output_dir_rgb_mask_70_gbcu10fold --log_dir ./output_dir_rgb_mask_70_gbcu10fold

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch8/mae/output_dir_rgb_mask_70_gbcu10fold/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch8/gbcu_data/ \
#  --mask_ratio 0.70 \
#  --output_dir ./output_dir_rgb_glcm_ssim_mask_70_gbcu10fold --log_dir ./output_dir_rgb_glcm_ssim_mask_70_gbcu10fold
# CUDA_VISIBLE_DEVICES=3,6,7 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=3 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16_d10 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch7/mae/output_dir_rgb_decoder_10_gbcu10fold/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch7/gbcu_data/ \
#  --output_dir ./output_dir_rgb_glcm_ssim_decoder_10_gbcu10fold --log_dir ./output_dir_rgb_glcm_ssim_decoder_10_gbcu10fold


# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain.py \
#  --batch_size 128 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/MAE_pth/pretained_ViT_base/mae_pretrain_vit_base_full.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_2/train\
#  --output_dir ./output_dir_rgb_gbcu10fold_fold_2 --log_dir ./output_dir_rgb_gbcu10fold_fold_2

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain.py \
#  --batch_size 128 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/MAE_pth/pretained_ViT_base/mae_pretrain_vit_base_full.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_3/train\
#  --output_dir ./output_dir_rgb_gbcu10fold_fold_3 --log_dir ./output_dir_rgb_gbcu10fold_fold_3

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain.py \
#  --batch_size 128 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/MAE_pth/pretained_ViT_base/mae_pretrain_vit_base_full.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_4/train\
#  --output_dir ./output_dir_rgb_gbcu10fold_fold_4 --log_dir ./output_dir_rgb_gbcu10fold_fold_4

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain.py \
#  --batch_size 128 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/MAE_pth/pretained_ViT_base/mae_pretrain_vit_base_full.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_5/train\
#  --output_dir ./output_dir_rgb_gbcu10fold_fold_5 --log_dir ./output_dir_rgb_gbcu10fold_fold_5

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain.py \
#  --batch_size 128 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/MAE_pth/pretained_ViT_base/mae_pretrain_vit_base_full.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_6/train\
#  --output_dir ./output_dir_rgb_gbcu10fold_fold_6 --log_dir ./output_dir_rgb_gbcu10fold_fold_6

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain.py \
#  --batch_size 128 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/MAE_pth/pretained_ViT_base/mae_pretrain_vit_base_full.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_7/train\
#  --output_dir ./output_dir_rgb_gbcu10fold_fold_7 --log_dir ./output_dir_rgb_gbcu10fold_fold_7

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain.py \
#  --batch_size 128 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/MAE_pth/pretained_ViT_base/mae_pretrain_vit_base_full.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_8/train\
#  --output_dir ./output_dir_rgb_gbcu10fold_fold_8 --log_dir ./output_dir_rgb_gbcu10fold_fold_8

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain.py \
#  --batch_size 128 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/MAE_pth/pretained_ViT_base/mae_pretrain_vit_base_full.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_9/train\
#  --output_dir ./output_dir_rgb_gbcu10fold_fold_9 --log_dir ./output_dir_rgb_gbcu10fold_fold_9

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_0/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_0/train\
#  --output_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_0 --log_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_0

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_1/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_1/train\
#  --output_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_1 --log_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_1

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_2/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_2/train\
#  --output_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_2 --log_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_2

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_3/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_3/train\
#  --output_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_3 --log_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_3

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_4/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_4/train\
#  --output_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_4 --log_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_4

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_5/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_5/train\
#  --output_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_5 --log_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_5

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=8 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_ssim_glcm_gbcu10fold_fold_6/checkpoint-280.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_6/train\
#  --output_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_6 --log_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_6

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_7/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_7/train\
#  --output_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_7 --log_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_7

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_8/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_8/train\
#  --output_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_8 --log_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_8

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_9/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_9/train\
#  --output_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_9 --log_dir ./output_dir_rgb_ssim_glcm_gbcu10fold_fold_9

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_0/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_0/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_gbcu10fold_fold_0 --log_dir ./output_finetune_rgb_gbcu10fold_fold_0

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_1/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_1/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_gbcu10fold_fold_1 --log_dir ./output_finetune_rgb_gbcu10fold_fold_1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_2/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_2/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_gbcu10fold_fold_2 --log_dir ./output_finetune_rgb_gbcu10fold_fold_2

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_3/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_3/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_gbcu10fold_fold_3 --log_dir ./output_finetune_rgb_gbcu10fold_fold_3

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_4/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_4/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_gbcu10fold_fold_4 --log_dir ./output_finetune_rgb_gbcu10fold_fold_4

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_5/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_5/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_gbcu10fold_fold_5 --log_dir ./output_finetune_rgb_gbcu10fold_fold_5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_6/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_6/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_gbcu10fold_fold_6 --log_dir ./output_finetune_rgb_gbcu10fold_fold_6

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_7/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_7/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_gbcu10fold_fold_7 --log_dir ./output_finetune_rgb_gbcu10fold_fold_7

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_8/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_8/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_gbcu10fold_fold_8 --log_dir ./output_finetune_rgb_gbcu10fold_fold_8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_gbcu10fold_fold_9/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_9/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_gbcu10fold_fold_9 --log_dir ./output_finetune_rgb_gbcu10fold_fold_9

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_decoder_8_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_6/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_gbcu10fold_fold_6 --log_dir ./output_finetune_rgb_gbcu10fold_fold_6

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_decoder_4_gbcu10fold_fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_6/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_4_gbcu10fold/fold_6 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_4_gbcu10fold/fold_6

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_decoder_6_gbcu10fold_fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_6/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_6_gbcu10fold/fold_6 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_6_gbcu10fold/fold_6

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_decoder_10_gbcu10fold_fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_6/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_10_gbcu10fold/fold_6 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_10_gbcu10fold/fold_6

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_decoder_10_gbcu10fold_fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_6/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_10_gbcu10fold/fold_6 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_10_gbcu10fold/fold_6

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_decoder_10_gbcu10fold_fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_6/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_10_gbcu10fold/fold_6 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_10_gbcu10fold/fold_6

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_mask_70_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_1/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_mask_70_gbcu10fold/fold_1 --log_dir ./output_finetune_rgb_ssim_glcm_mask_70_gbcu10fold/fold_1

#  CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_mask_70_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_8/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_mask_70_gbcu10fold/fold_8 --log_dir ./output_finetune_rgb_ssim_glcm_mask_70_gbcu10fold/fold_8

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_mask_70_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_9/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_mask_70_gbcu10fold/fold_9 --log_dir ./output_finetune_rgb_ssim_glcm_mask_70_gbcu10fold/fold_9

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_mask_80_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_1/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_mask_80_gbcu10fold/fold_1 --log_dir ./output_finetune_rgb_ssim_glcm_mask_80_gbcu10fold/fold_1

#  CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_mask_80_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_8/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_mask_80_gbcu10fold/fold_8 --log_dir ./output_finetune_rgb_ssim_glcm_mask_80_gbcu10fold/fold_8

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_mask_80_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_9/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_mask_80_gbcu10fold/fold_9 --log_dir ./output_finetune_rgb_ssim_glcm_mask_80_gbcu10fold/fold_9

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_mask_85_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_1/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_mask_85_gbcu10fold/fold_1 --log_dir ./output_finetune_rgb_ssim_glcm_mask_85_gbcu10fold/fold_1

#  CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_mask_85_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_8/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_mask_85_gbcu10fold/fold_8 --log_dir ./output_finetune_rgb_ssim_glcm_mask_85_gbcu10fold/fold_8

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_mask_85_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_9/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_mask_85_gbcu10fold/fold_9 --log_dir ./output_finetune_rgb_ssim_glcm_mask_85_gbcu10fold/fold_9



# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python -m torch.distributed.launch --master_port=29501 --nproc_per_node=4 main_pretrain.py \
#  --batch_size 128 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/MAE_pth/pretained_ViT_base/mae_pretrain_vit_base_full.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/busi_final/imgs\
#  --output_dir ./output_dir_rgb_busi5fold_pretrain --log_dir ./output_dir_rgb_busi5fold_pretrain

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python -m torch.distributed.launch --master_port=29500 --nproc_per_node=8 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_busi5fold_pretrain/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/busi_final/imgs\
#  --output_dir ./output_dir_rgb_glcm_ssim_busi5fold_pretrain --log_dir ./output_dir_rgb_glcm_ssim_busi5fold_pretrain



# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch5/mae/output_dir_rgb_glcm_ssim_busi5fold_pretrain/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/busi5fold_final/fold_0 \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_busi5fold_final/fold_0 --log_dir ./output_finetune_rgb_glcm_ssim_busi5fold_final/fold_0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch5/mae/output_dir_rgb_glcm_ssim_busi5fold_pretrain/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/busi5fold_final/fold_1 \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_busi5fold_final/fold_1 --log_dir ./output_finetune_rgb_glcm_ssim_busi5fold_final/fold_1

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch5/mae/output_dir_rgb_glcm_ssim_busi5fold_pretrain/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/busi5fold_final/fold_2 \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_busi5fold_final/fold_2 --log_dir ./output_finetune_rgb_glcm_ssim_busi5fold_final/fold_2

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch5/mae/output_dir_rgb_glcm_ssim_busi5fold_pretrain/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/busi5fold_final/fold_3 \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_busi5fold_final/fold_3 --log_dir ./output_finetune_rgb_glcm_ssim_busi5fold_final/fold_3

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --blr 5e-4 --weight_decay 0.05 --min_lr 3e-6\
#  --finetune /home/aarjav/scratch/v01_scratch5/mae/output_dir_rgb_glcm_ssim_busi5fold_pretrain/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/busi5fold_final/fold_1 \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_busi5fold_final/fold_1 --log_dir ./output_finetune_rgb_glcm_ssim_busi5fold_final/fold_1

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
 --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --blr 1e-4 --weight_decay 0.05 --min_lr 1e-7\
 --finetune /home/aarjav/scratch/v01_scratch5/mae/output_dir_rgb_glcm_ssim_busi5fold_pretrain/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/busi5fold_final/fold_1 \
 --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_busi5fold_final/fold_1 --log_dir ./output_finetune_rgb_glcm_ssim_busi5fold_final/fold_1
