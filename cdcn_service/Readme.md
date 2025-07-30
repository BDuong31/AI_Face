# CDCN++ Model for Face Anti-Spoofing

This repository contains the implementation of the CDCN++ model for face anti-spoofing, as part of the FAS Challenge at CVPRW 2020 (Track 2: Single-modal).
---
## Overview
The CDCN++ (Central Difference Convolutional Network++) model is designed for face anti-spoofing tasks. It uses central difference convolution layers to enhance feature extraction and spatial attention mechanisms to improve robustness against spoofing attacks.
---
## Requirements
To set up the environment, install the required dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
## Dataset
The dataset should be placed in the my_face_dataset directory. The dataset structure should follow this format:


## Description of Folders and Files

- **protocol/**: Contains the lists of image paths for training, validation, and testing. Each text file lists the relative paths to the images.
  - `train_list.txt`: Contains a list of images used for training.
  - `val_list.txt`: Contains a list of images used for validation.
  - `test_list.txt`: Contains a list of images used for testing.

- **images/**: Contains the actual image data, divided into two categories:
  - `real/`: Contains images of real faces.
  - `fake/`: Contains images of fake faces.

## Usage

1. Ensure the directory structure is intact.
2. Use the `train_list.txt`, `val_list.txt`, and `test_list.txt` files to load the respective images for training, validation, and testing.

Update the paths in train_CDCNpp_model1.py and test_CDCNpp_model1.py if necessary.

## Training
To train the CDCN++ model, run the following command:
```bash
python train_CDCNpp_model1.py --gpu <GPU_ID> --batchsize <BATCH_SIZE> --epochs <EPOCHS> --lr <LEARNING_RATE>
```
Example:
```bash
python train_CDCNpp_model1.py --gpu 0 --batchsize 9 --epochs 60 --lr 0.00008
```

Training logs and model checkpoints will be saved in the CDCNpp_BinaryMask_P1_07 directory by default.

## Testing
To test the trained model, use the following command:
```bash
python test_CDCNpp_model1.py --gpu <GPU_ID> --batchsize <BATCH_SIZE>
```
Example:
```bash
python test_CDCNpp_model1.py --gpu 0 --batchsize 9
```
The results will be saved in the CDCNpp_BinaryMask_P1_07 directory.


## Utilities
The utils.py file contains helper functions for:

Calculating performance metrics (e.g., EER, ACER)
Saving and loading model checkpoints
Data augmentation techniques (e.g., random erasing, cutout)

## Results
The CDCN++ model achieves state-of-the-art performance on the Oulu-NPU dataset. Example metrics include:

ACER: 1.2%
EER: 1.5%
