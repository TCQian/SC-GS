#!/bin/bash

#SBATCH --job-name=SC-GS-hypernerf-no-synthesis    # Job name
#SBATCH --time=6:00:00                   # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1             # must use this GPU, since pytorch3d relied on it
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications
#SBATCH --mem=16G                        # Request 16GB of memory

source ~/.bashrc
conda activate scgs

CUDA_VISIBLE_DEVICES=0 python train_gui.py --source_path ./data/hypernerf/interp/aleks-teapot --model_path outputs/aleks-teapot-no-synthesis --deform_type node --node_num 512 --hyper_dim 2 --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ./data/hypernerf/interp/aleks-teapot --model_path outputs/aleks-teapot-no-synthesis --deform_type node --node_num 512 --hyper_dim 2 --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800
