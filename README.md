# SKIN LESION MEDICAL IMAGE SEGMENTATION USING MULTIRESUNET ARCHITECTURE
This project applies the MultiResUNet architecture for skin lesion segmentation to aid in early skin cancer diagnosis. This repository contains the code, documentation, and web-based deployment model for clinical use.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Deployment](#deployment)
7. [Future Work](#future-work)

---

### Introduction

Skin cancer is a growing concern worldwide. Traditional methods for skin lesion analysis face challenges due to variability in lesion characteristics. This project leverages deep learning, specifically the MultiResUNet architecture, to enhance skin lesion segmentation accuracy.

### Project Overview

- **Objective**: Improve skin lesion segmentation accuracy using MultiResUNet.
- **Dataset**: ISIC Challenge 2016 with 2,558 annotated images.
- **Technologies**: Python, TensorFlow, Keras.

### Dataset

The ISIC 2016 dataset consists of 2,558 high-quality, annotated dermatological images.

### Methodology

#### Preprocessing
- Resize images to 256x256 pixels.
- Normalize pixel values to [0, 1].
- Data augmentation: horizontal and vertical flips, rotation, zoom, and shearing.

#### Model Architecture
MultiResUNet, an enhanced U-Net model with multi-resolution blocks and residual connections, is used to capture the complexity of skin lesions.

### Results

- **Dice Similarity Coefficient (DSC)**: 0.8724
- **Intersection-over-Union (IoU)**: Achieved high accuracy in lesion segmentation.

### Deployment

A web-based interface allows real-time skin lesion analysis for healthcare providers. Upload an image to get the segmented output in seconds.

