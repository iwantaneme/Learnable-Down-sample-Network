# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 4
downscale_factor = 4
#the number of channel(1 or 3)
num_channel =1
# Current configuration parameter method
mode = "valid"
# Experiment name, easy to save weights and log files
exp_name = "fsrcnn_x4"

if mode == "train":
    # Dataset
    # train_image_dir = f"data/DIV2K/train/"
    # train_image_dir = f"data/NWPU VHR-10 dataset/positive image set/"
    train_image_dir = f"data/canon/train/4/HR/"
    # train_image_dir_lr = f"data/DIV2K/train_lr/"
    train_image_dir_lr= f"data/canon/train/4/LR/"

    valid_image_dir = f"data/DIV2K/valid/"
    # test_lr_image_dir = f"data/Set5/LRbicx{downscale_factor}/"
    # test_hr_image_dir = f"data/Set5/GTmod12/"
    test_lr_image_dir = f"data/canon/test/4/LR/"
    test_hr_image_dir = f"data/canon/test/4/HR/"

    image_size = 200
    batch_size = 8
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 1
    resume = "samples/fsrcnn_x4/epoch_5.pth"

    # Total number of epochs
    epochs = 300

    # SGD optimizer parameter

    model_lr = 1e-3
    model_momentum = 0.9
    model_weight_decay = 1e-4
    model_nesterov = False

    print_frequency = 200

if mode == "valid":
    # Test data address
    # lr_dir = f"data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"results/test/test1/"
    # hr_dir = f"data/Set5/GTmod12"
    # lr_dir = f"data/canon/test/4/LR/"
    # hr_dir = f"data/canon/test/4/HR/"

    lr_dir = f"data/test/"
    hr_dir = f"data/test/"

    model_path = f"samples/{exp_name}/epoch_35.pth"
