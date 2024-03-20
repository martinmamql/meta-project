#!/bin/bash

#SBATCH -c8
#SBATCH -p high 
#SBATCH --mem=16g
#SBATCH --gres=gpu:2080Ti:1
CUDA_VISIBLE_DEVICES=0 python ../run_main.py --dataset_name recola_arousal --header_type residual --header_training_paradigm simultaneous --batch_size_contrastive 32 --batch_size_residual 256 --lr_contrastive 0.01 --mm_contrastive 0.5 --wd_contrastive 1e-06 --optim_contrastive sgd --feature_extractor_depth 2 --lr_unimodal 0.01 --lr_bimodal 0.01 --lr_trimodal 0.01 --mm_unimodal 0.5 --mm_bimodal 0.5 --mm_trimodal 0.5 --wd_unimodal 1e-06 --wd_bimodal 1e-06 --wd_trimodal 1e-06 --optim_unimodal sgd --optim_bimodal sgd --optim_trimodal sgd --prediction_head_depth 2 --num_epochs_contrastive 100 --num_epochs_residual 200 --patience_contrastive 20 --patience_residual 40 --warmup_epochs 1 --seed 1706 --dataset_dir /work/siyuanwu/meta/ --output_dir ../output/
