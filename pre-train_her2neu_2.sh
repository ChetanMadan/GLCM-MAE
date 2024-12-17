# !/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0,1,2,5 \
# python -m torch.distributed.launch --master_port=29501 --nproc_per_node=4 main_pretrain.py \
#  --batch_size 128 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/mae/MAE_pth/pretained_ViT_base/mae_pretrain_vit_base_full.pth \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs \
#  --output_dir ./output_dir_Her2neu_Grounding_DINO_mae --log_dir ./output_dir_Her2neu_Grounding_DINO_mae

CUDA_VISIBLE_DEVICES=0,1,2,5 \
python -m torch.distributed.launch --master_port=29501 --nproc_per_node=4 main_pretrain_2.py \
 --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
 --resume /home/aarjav/scratch/mae/output_dir_Her2neu_Grounding_DINO_mae/checkpoint-199.pth \
 --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs \
 --output_dir ./output_dir_Her2neu_Grounding_DINO_glcm_mae --log_dir ./output_dir_Her2neu_Grounding_DINO_glcm_mae

# CUDA_VISIBLE_DEVICES=0,1,2,5 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/mae/output_dir_Her2neu_Grounding_DINO_mae/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs/imgs/ --fold fold_0 \
#  --nb_classes 2 --dist_eval --output_dir ./output_finetune_Her2neu_Grounding_DINO_mae/fold_0 --log_dir ./output_finetune_Her2neu_Grounding_DINO_mae/fold_0

#  CUDA_VISIBLE_DEVICES=0,1,2,5 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/mae/output_dir_Her2neu_Grounding_DINO_mae/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs/imgs/ --fold fold_1 \
#  --nb_classes 2 --dist_eval --output_dir ./output_finetune_Her2neu_Grounding_DINO_mae/fold_1 --log_dir ./output_finetune_Her2neu_Grounding_DINO_mae/fold_1

# CUDA_VISIBLE_DEVICES=0,1,2,5 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/mae/output_dir_Her2neu_Grounding_DINO_mae/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs/imgs/ --fold fold_2 \
#  --nb_classes 2 --dist_eval --output_dir ./output_finetune_Her2neu_Grounding_DINO_mae/fold_2 --log_dir ./output_finetune_Her2neu_Grounding_DINO_mae/fold_2

# CUDA_VISIBLE_DEVICES=0,1,2,5 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/mae/output_dir_Her2neu_Grounding_DINO_mae/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs/imgs/ --fold fold_3 \
#  --nb_classes 2 --dist_eval --output_dir ./output_finetune_Her2neu_Grounding_DINO_mae/fold_3 --log_dir ./output_finetune_Her2neu_Grounding_DINO_mae/fold_3

# CUDA_VISIBLE_DEVICES=0,1,2,5 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/mae/output_dir_Her2neu_Grounding_DINO_mae/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs/imgs/ --fold fold_4 \
#  --nb_classes 2 --dist_eval --output_dir ./output_finetune_Her2neu_Grounding_DINO_mae/fold_4 --log_dir ./output_finetune_Her2neu_Grounding_DINO_mae/fold_4

CUDA_VISIBLE_DEVICES=0,1,2,5 python -m torch.distributed.launch \
 --master_port=29501 --nproc_per_node=4 main_finetune.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/mae/output_dir_Her2neu_Grounding_DINO_glcm_mae/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs/imgs/ --fold fold_0 \
 --nb_classes 2 --dist_eval --output_dir ./output_finetune_Her2neu_Grounding_DINO_mae_glcm/fold_0 --log_dir ./output_finetune_Her2neu_Grounding_DINO_mae_glcm/fold_0

 CUDA_VISIBLE_DEVICES=0,1,2,5 python -m torch.distributed.launch \
 --master_port=29501 --nproc_per_node=4 main_finetune.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/mae/output_dir_Her2neu_Grounding_DINO_glcm_mae/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs/imgs/ --fold fold_1 \
 --nb_classes 2 --dist_eval --output_dir ./output_finetune_Her2neu_Grounding_DINO_mae_glcm/fold_1 --log_dir ./output_finetune_Her2neu_Grounding_DINO_mae_glcm/fold_1

CUDA_VISIBLE_DEVICES=0,1,2,5 python -m torch.distributed.launch \
 --master_port=29501 --nproc_per_node=4 main_finetune.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/mae/output_dir_Her2neu_Grounding_DINO_glcm_mae/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs/imgs/ --fold fold_2 \
 --nb_classes 2 --dist_eval --output_dir ./output_finetune_Her2neu_Grounding_DINO_mae_glcm/fold_2 --log_dir ./output_finetune_Her2neu_Grounding_DINO_mae_glcm/fold_2

CUDA_VISIBLE_DEVICES=0,1,2,5 python -m torch.distributed.launch \
 --master_port=29501 --nproc_per_node=4 main_finetune.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/mae/output_dir_Her2neu_Grounding_DINO_glcm_mae/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs/imgs/ --fold fold_3 \
 --nb_classes 2 --dist_eval --output_dir ./output_finetune_Her2neu_Grounding_DINO_mae_glcm/fold_3 --log_dir ./output_finetune_Her2neu_Grounding_DINO_mae_glcm/fold_3

CUDA_VISIBLE_DEVICES=0,1,2,5 python -m torch.distributed.launch \
 --master_port=29501 --nproc_per_node=4 main_finetune.py\
 --batch_size 128  --model vit_base_patch16 --epochs 100 \
 --finetune /home/aarjav/scratch/mae/output_dir_Her2neu_Grounding_DINO_glcm_mae/checkpoint-399.pth \
 --data_path /home/aarjav/scratch/Her2Neu_slices/Grounding_DINO_imgs/imgs/ --fold fold_4 \
 --nb_classes 2 --dist_eval --output_dir ./output_finetune_Her2neu_Grounding_DINO_mae_glcm/fold_4 --log_dir ./output_finetune_Her2neu_Grounding_DINO_mae_glcm/fold_4


# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch/mae/output_dir_rgb_glcm_ssim_no_curriculum_L5_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch/gbcu10fold/gbcu_fold_1/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_no_curriculum_L5_gbcu10fold/fold_1 --log_dir ./output_finetune_rgb_glcm_ssim_no_curriculum_L5_gbcu10fold/fold_1

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch/mae/output_dir_rgb_glcm_ssim_no_curriculum_L5_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch/gbcu10fold/gbcu_fold_2/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_no_curriculum_L5_gbcu10fold/fold_2 --log_dir ./output_finetune_rgb_glcm_ssim_no_curriculum_L5_gbcu10fold/fold_2

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch/mae/output_dir_rgb_glcm_ssim_no_curriculum_L5_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch/gbcu10fold/gbcu_fold_3/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_no_curriculum_L5_gbcu10fold/fold_3 --log_dir ./output_finetune_rgb_glcm_ssim_no_curriculum_L5_gbcu10fold/fold_3

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch/mae/output_dir_rgb_glcm_ssim_no_curriculum_L5_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch/gbcu10fold/gbcu_fold_4/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_no_curriculum_L5_gbcu10fold/fold_4 --log_dir ./output_finetune_rgb_glcm_ssim_no_curriculum_L5_gbcu10fold/fold_4

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_large_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_vit_large_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_2/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_2 --log_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_2

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_large_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_vit_large_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_3/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_3 --log_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_3

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_large_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_vit_large_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_4/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_4 --log_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_4

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_large_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_vit_large_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_5/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_5 --log_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_5

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_large_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_vit_large_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_6/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_6 --log_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_6

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_large_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_vit_large_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_7/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_7 --log_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_7

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_large_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_vit_large_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_8/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_8 --log_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_8

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 64  --model vit_large_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_vit_large_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_9/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_9 --log_dir ./output_finetune_rgb_glcm_ssim_vit_large_gbcu10fold/fold_9

# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_no_curriculum_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_4/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_no_curriculum_gbcu10fold/fold_4 --log_dir ./output_finetune_rgb_ssim_glcm_no_curriculum_gbcu10fold/fold_4

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_no_curriculum_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_8/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_no_curriculum_gbcu10fold/fold_8 --log_dir ./output_finetune_rgb_ssim_glcm_no_curriculum_gbcu10fold/fold_8

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#  --master_port=29501 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_no_curriculum_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_9/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_no_curriculum_gbcu10fold/fold_9 --log_dir ./output_finetune_rgb_ssim_glcm_no_curriculum_gbcu10fold/fold_9

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_decoder_8_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_1/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_8_gbcu10fold/fold_1 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_8_gbcu10fold/fold_1

#  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_decoder_8_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_8/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_8_gbcu10fold/fold_8 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_8_gbcu10fold/fold_8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch2/mae/output_dir_rgb_glcm_ssim_decoder_8_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch2/gbcu10fold/gbcu_fold_9/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_8_gbcu10fold/fold_9 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_8_gbcu10fold/fold_9

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_only_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_1/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_only_gbcu10fold/fold_1 --log_dir ./output_finetune_rgb_glcm_only_gbcu10fold/fold_1

#  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_only_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_8/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_only_gbcu10fold/fold_8 --log_dir ./output_finetune_rgb_glcm_only_gbcu10fold/fold_8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_only_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_9/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_glcm_only_gbcu10fold/fold_9 --log_dir ./output_finetune_rgb_glcm_only_gbcu10fold/fold_9

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_glcm_only_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_1/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_only_gbcu10fold/fold_1 --log_dir ./output_finetune_glcm_only_gbcu10fold/fold_1

#  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_glcm_only_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_8/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_only_gbcu10fold/fold_8 --log_dir ./output_finetune_glcm_only_gbcu10fold/fold_8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_glcm_only_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_9/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_glcm_only_gbcu10fold/fold_9 --log_dir ./output_finetune_glcm_only_gbcu10fold/fold_9

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_decoder_4_gbcu10fold_fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_1/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_4_gbcu10fold/fold_1 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_4_gbcu10fold/fold_1

#  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_decoder_4_gbcu10fold_fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_8/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_4_gbcu10fold/fold_8 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_4_gbcu10fold/fold_8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_decoder_4_gbcu10fold_fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_9/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_4_gbcu10fold/fold_9 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_4_gbcu10fold/fold_9

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_decoder_6_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_1/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_6_gbcu10fold/fold_1 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_6_gbcu10fold/fold_1

#  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_decoder_6_gbcu10fold_fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_8/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_6_gbcu10fold/fold_8 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_6_gbcu10fold/fold_8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_decoder_6_gbcu10fold_fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_9/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_6_gbcu10fold/fold_9 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_6_gbcu10fold/fold_9

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_decoder_10_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_1/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_10_gbcu10fold/fold_1 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_10_gbcu10fold/fold_1

#  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_decoder_10_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_8/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_10_gbcu10fold/fold_8 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_10_gbcu10fold/fold_8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch4/mae/output_dir_rgb_glcm_ssim_decoder_10_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/v01_scratch4/gbcu10fold/gbcu_fold_9/ \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_rgb_ssim_glcm_decoder_10_gbcu10fold/fold_9 --log_dir ./output_finetune_rgb_ssim_glcm_decoder_10_gbcu10fold/fold_9

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch8/mae/output_finetune_rgb_glcm_ssim_no_curriculum_L15_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/gbcu10fold_final/gbcu_fold_1 \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_tgb_glcm_ssim_L15_gbcu_final/fold_1 --log_dir ./output_finetune_tgb_glcm_ssim_L15_gbcu_final/fold_1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch8/mae/output_finetune_rgb_glcm_ssim_no_curriculum_L15_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/gbcu10fold_final/gbcu_fold_2 \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_tgb_glcm_ssim_L15_gbcu_final/fold_2 --log_dir ./output_finetune_tgb_glcm_ssim_L15_gbcu_final/fold_2

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch8/mae/output_finetune_rgb_glcm_ssim_no_curriculum_L15_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/gbcu10fold_final/gbcu_fold_5 \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_tgb_glcm_ssim_L15_gbcu_final/fold_5 --log_dir ./output_finetune_tgb_glcm_ssim_L15_gbcu_final/fold_5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch8/mae/output_finetune_rgb_glcm_ssim_no_curriculum_L15_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/gbcu10fold_final/gbcu_fold_8 \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_tgb_glcm_ssim_L15_gbcu_final/fold_8 --log_dir ./output_finetune_tgb_glcm_ssim_L15_gbcu_final/fold_8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#  --master_port=29500 --nproc_per_node=4 main_finetune_gbcu.py\
#  --batch_size 128  --model vit_base_patch16 --epochs 100 \
#  --finetune /home/aarjav/scratch/v01_scratch8/mae/output_finetune_rgb_glcm_ssim_no_curriculum_L15_gbcu10fold/checkpoint-399.pth \
#  --data_path /home/aarjav/scratch/gbcu10fold_final/gbcu_fold_9 \
#  --nb_classes 3 --dist_eval --output_dir ./output_finetune_tgb_glcm_ssim_L15_gbcu_final/fold_9 --log_dir ./output_finetune_tgb_glcm_ssim_L15_gbcu_final/fold_9

# CUDA_VISIBLE_DEVICES=1,2,3,4 \
# python -m torch.distributed.launch --master_port=29501 --nproc_per_node=4 main_pretrain_2.py \
#  --batch_size 8 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 \
#  --resume /home/aarjav/scratch/v01_scratch8/mae/output_dir_rgb_decoder_8_gbcu10fold/checkpoint-199.pth  \
#  --blr 1.5e-4 --weight_decay 0.05 --data_path /home/aarjav/scratch/v01_scratch8/gbcu_data/ \
#  --output_dir ./output_finetune_rgb_glcm_ssim_no_curriculum_L15_gbcu10fold --log_dir ./output_finetune_rgb_glcm_ssim_no_curriculum_L15_gbcu10fold
