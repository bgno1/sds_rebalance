# Optimizing Drone-Captured Maritime Rescue Image Object Detection through Dataset Rebalancing under Sample Constraints

## Overview

This repository contains the code supporting the findings in the paper *"Optimizing Drone-Captured Maritime Rescue Image Object Detection through Dataset Rebalancing under Sample Constraints,"* which is currently under peer review. The SeaDronesSee dataset [1], a highly valuable and unique resource for drone-based maritime search and rescue research, provides critical data for advancing object detection models in this field. However, due to the challenging nature of collecting data in such specialized and difficult environments, the dataset inherently faces sample constraints and imbalances in the distribution of certain categories.

Our work seeks to address these limitations by proposing a greedy algorithm-based method for re-partitioning the dataset. This approach aims to enhance the generalization and training efficiency of detection models while making the most of the invaluable data contained within the SeaDronesSee dataset. Ultimately, our goal is to maximize the impact of this precious resource, ensuring it can be fully leveraged to improve drone-based maritime rescue operations.

## SeaDronesSee Dataset

The SeaDronesSee dataset is available on the official GitHub repository: [SeaDronesSee GitHub](https://github.com/Ben93kie/SeaDronesSee).  
The `instances_train.json` and `instances_val.json` files used in this code can be found in the official SeaDronesSee dataset, which is downloadable from the following link: [Download SeaDronesSee Dataset](https://cloud.cs.uni-tuebingen.de/index.php/s/aJQPHLGnke68M52).

## User Guide

### Requirements

The code has been tested in a **Windows environment** using **Python 3.8**.

### How to Run

To execute the rebalancing process, run the following command in your terminal:

```bash
python greedy_resplit.py
```

### Input Files

This script requires the original SeaDronesSee dataset annotation files as inputs:

- `./instances_train.json` – Original training set annotations from the SeaDronesSee dataset.
- `./instances_val.json` – Original validation set annotations from the SeaDronesSee dataset.

Ensure these files are downloaded and placed in the correct directory before running the script. You can obtain them from the [dataset link](https://cloud.cs.uni-tuebingen.de/index.php/s/aJQPHLGnke68M52).

### Output Files

After running the script, you will find the rebalanced annotation files in the same directory as the input files:

- `./instances_train_balanced.json` – Rebalanced training set annotations.
- `./instances_val_balanced.json` – Rebalanced validation set annotations.

## About Model Validation

- The Faster R-CNN and Cascade R-CNN models, integrated with a ResNeXt-101 backbone and FPN, can be implemented and validated using the [MMDetection framework](https://github.com/open-mmlab/mmdetection). This framework provides the flexibility and tools needed to test these models, supporting the experiments presented in our paper.

- Experimental results for YOLOv8s can be validated using the official [Ultralytics YOLOv8]([Releases · ultralytics/ultralytics · GitHub](https://github.com/ultralytics/ultralytics/releases)) repository, which offers state-of-the-art performance for real-time object detection and plays a key role in validating our findings.

- All models are trained using the original and re-partitioned training and validation annotations. Validation results, evaluated on the [SeaDronesSee Leaderboard](https://macvi.org/leaderboard/airborne/seadronessee/object-detection), demonstrate noticeable performance improvements.

## References

[1] Varga, Leon Amadeus, et al. "SeaDronesSee: A Maritime Benchmark for Detecting Humans in Open Water." *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 2022. 
