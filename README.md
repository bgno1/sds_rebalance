# Improving Drone-captured Maritime Rescue Image Object Detection with Dataset Re-balancing under Sample Constraints

## Overview

This repository contains the code supporting the result in the paper *"Improving Drone-captured Maritime Rescue Image Object Detection with Dataset Re-balancing under Sample Constraints"*, which is currently under peer review. The study addresses the issue of sample imbalance and scarcity in the SeaDronesSee dataset, and proposes a greedy algorithm-based method for re-partitioning the dataset to improve model training and generalization.

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