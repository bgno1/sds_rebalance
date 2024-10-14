[![DOI](https://zenodo.org/badge/864396002.svg)](https://doi.org/10.5281/zenodo.13894509)

# Optimizing Drone-Captured Maritime Rescue Image Object Detection through Dataset Rebalancing under Sample Constraints

## 1. Overview

This repository contains the code supporting the findings in the paper *"Optimizing Drone-Captured Maritime Rescue Image Object Detection through Dataset Rebalancing under Sample Constraints,"* which is currently under peer review. The SeaDronesSee dataset [1], a highly valuable and unique resource for drone-based maritime search and rescue research, provides critical data for advancing object detection models in this field. However, due to the challenging nature of collecting data in such specialized and difficult environments, the dataset inherently faces sample constraints and imbalances in the distribution of certain categories.

Our work seeks to address these limitations by proposing a greedy algorithm-based method for re-partitioning the dataset. This approach aims to enhance the generalization and training efficiency of detection models while making the most of the invaluable data contained within the SeaDronesSee dataset. Ultimately, our goal is to maximize the impact of this precious resource, ensuring it can be fully leveraged to improve drone-based maritime rescue operations.

## 2. SeaDronesSee Dataset

The SeaDronesSee dataset, essential for maritime search and rescue object detection tasks, can be downloaded directly from the official repository: [Download SeaDronesSee Dataset](https://cloud.cs.uni-tuebingen.de/index.php/s/aJQPHLGnke68M52). For further details please refer to the official SeaDronesSee website: [https://macvi.org/](https://macvi.org/).

## 3. Dataset re-partitioning based on the greedy algorithm

The code has been tested using Python 3.8.

To execute the rebalancing process, run the `greedy_resplit.py` script located in the `greedy_resplit` folder of this repository using the following command in your terminal

```bash
python greedy_resplit.py
```

**Input Files**

This script requires the original SeaDronesSee dataset annotation files as inputs:

- `./instances_train.json` – Original training set annotations from the SeaDronesSee dataset.
- `./instances_val.json` – Original validation set annotations from the SeaDronesSee dataset.

Ensure these files are downloaded and placed in the correct directory before running the script. You can obtain them from the [dataset link](https://cloud.cs.uni-tuebingen.de/index.php/s/aJQPHLGnke68M52).

**Output Files**

After running the script, you will find the rebalanced annotation files in the same directory as the input files:

- `./instances_train_balanced.json` – Rebalanced training set annotations.
- `./instances_val_balanced.json` – Rebalanced validation set annotations.

## 4. Model Validation

### 4.1 Dataset Preparation

- Download the SeaDronesSee dataset from [this link](https://cloud.cs.uni-tuebingen.de/index.php/s/aJQPHLGnke68M52) and copy the contents of the `images` folder into the `./dataset/SeaDronesSee` directory of this repository.

- Create a `dataset/SeaDronesSee/images/train_val` directory and copy all the images from the `train` and `val` folders into this newly created folder.

The directory structure should look like this:

```md
- dataset/
    - SeaDronesSee/
        - annotations/    # Annotation files in JSON format
        - images/         # Image data
            - test/
            - train/
            - val/
            - train_val/  # Contains all images from both train and val folders
        - labels/         # (for YOLOv8) Annotations for images
        - test.txt        # (for YOLOv8) List of test images
        - train.txt       # (for YOLOv8) List of training images
        - val.txt         # (for YOLOv8) List of validation images
```

- Run the `./greedy_resplit/copy_images.py` script from this repository. This script will properly partition the training and validation images from the `train_val` folder into the `train` and `val` folders under `./dataset/SeaDronesSee_balanced`, based on the rebalanced annotation files.

- Copy the `images/test` folder from the official SeaDronesSee dataset to the `./dataset/SeaDronesSee_balanced/images` directory in this repository.

The final directory structure of the dataset should look as follows:

```md
- dataset/
    - SeaDronesSee/           # Original SeaDronesSee dataset
        - annotations/        # Annotation files in JSON format
        - images/             # Image data
            - test/           
            - train/          # original training set
            - val/            # original validation set
        - labels/             # (for YOLOv8) Annotations for images
        - test.txt            # (for YOLOv8) List of test images
        - train.txt           # (for YOLOv8) List of training images
        - val.txt             # (for YOLOv8) List of validation images
    - SeaDronesSee_balanced/  # Rebalanced version of the SeaDronesSee dataset
        - annotations/        # Annotation files in JSON format
        - images/             # Image data
            - test/
            - train/          # rebalanced training set
            - val/            # rebalanced validation set
        - labels/             # (for YOLOv8) Annotations for images
        - test.txt            # (for YOLOv8) List of test images
        - train.txt           # (for YOLOv8) List of training images
        - val.txt             # (for YOLOv8) List of validation images
```

The contents of the `SeaDronesSee/images` and `SeaDronesSee_balanced/images` folders need to be downloaded and generated through the steps mentioned above. All other files and folders are already included in this repository.

### 4.2 Faster R-CNN and Cascade R-CNN

#### 4.2.1 Training

- Download the MMDetection project from the [official MMDetection GitHub repository](https://github.com/open-mmlab/mmdetection), and copy the contents of the `mmdetection-main` folder into this repository's `mmdetection` folder. Follow the official guidelines to install the necessary dependencies.

The relevant directory structure will look as follows:

```md
- mmdetection
    - tools/
    - configs/
    ...
    frx.py
    frx_bl.py
    cas.py
    cas_bl.py
```

Among these files, `frx.py`, `frx_bl.py`, `cas.py`, and `cas_bl.py` are custom model training scripts provided in this GitHub repository to support the experiments in the paper. These scripts can be found in the `validation/mmdetection/` directory. Specifically:

- `frx.py` and `frx_bl.py` are used to train the Faster R-CNN models on the original and rebalanced datasets, respectively.
- `cas.py` and `cas_bl.py` are used to train the Cascade R-CNN models  on the original and rebalanced datasets, respectively.

**Example usage:**

- To train the Faster R-CNN model on the original dataset using `frx.py`, navigate to the `mmdetection` directory and execute the following command:

```bash
python ./tools/train.py ./frx.py
```

- To train the Faster R-CNN model on the rebalanced dataset using `frx_bl.py`, navigate to the `mmdetection` directory and execute the following command:

```bash
python ./tools/train.py ./frx_bl.py
```

#### 4.2.2 Test on SeaDronesSee Leaderboard

- Use MMDetection’s test script to generate prediction files in PKL format. For example, to generate predictions using the model trained with `frx.py`, run the following command:

```bash
python ./tools/test.py ./frx.py ./work_dirs/frx/epoch_12.pth --out ./test.pkl
```

- Convert the `test.pkl` file generated in the previous step to the JSON format required by the SeaDronesSee Leaderboard using the `mmdetection/to_sds_json.py` script provided in this repository.

- Validate the model’s generalization performance by submitting the JSON file to the SeaDronesSee Leaderboard server at [SeaDronesSee Leaderboard](https://macvi.org/leaderboard/airborne/seadronessee/object-detection).

### 4.3 YOLOv8

#### 4.3.1 Training

Download the YOLOv8 project from the [official Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics), the following directory structure will be available:

```md
- ultralytics
    - docs/
    ...
    - ultralytics/
        - cfg/
        - data/
        ...
```

Using the Ultralytics official documentation as a reference, you can create a Python training script in the `ultralytics` directory with the following structure:

```python
model = YOLO(r'./yolo_sds.yaml') # YOLOv8模型文件
results = model.train(
    data = 'sds_balanced.yaml',
    workers=12,
    batch=48
)
```

Key details:

- The `workers` and `batch` parameters should be adjusted based on your hardware, particularly the GPU. Setting these values too high on low-spec machines may lead to crashes. Additionally, these parameters impact both training speed and model performance.

- The `yolo_sds.yaml` file represents the improved YOLOv8 model used in our experiments and can be found in the `validation/yolov8` directory of this repository.

- The `sds_balanced.yaml` is the dataset used for training and can either be `sds.yaml` (original split) or `sds_balanced.yaml` (rebalanced split). Both YAML files are available in the `dataset/yolov8` directory of this repository.

In addition, YOLOv8 uses the CIoU loss function by default. To use the WIoU loss function as mentioned in the paper, the following changes should be made:

- Import `wiou_loss` in `ultralytics/utils/loss.py`:

- Modify the `forward` function in the `BboxLoss` class in `ultralytics/utils/loss.py` by adding the following code:
  
  ```python
  if config.bbox_loss == 'WIoU':
      loss, iou = bbox_wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
      loss_iou = (loss * weight).sum() / target_scores_sum
  else:   # CIoU
      iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
      loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
  ```

#### 4.3.2 Test on SeaDronesSee Leaderboard

Using the structure outlined in the Ultralytics official documentation,  write the following code:

```python
model = YOLO(r'./best.pt')  # Trained model weights
metrics = model.val(split='test', save_json=True)
```

Once executed, use the information from the output log to locate the `predictions.json` file, and upload it to the [SeaDronesSee Leaderboard server](https://macvi.org/leaderboard/airborne/seadronessee/object-detection) to evaluate the model's generalization performance.

## References

[1] Varga, Leon Amadeus, et al. "SeaDronesSee: A Maritime Benchmark for Detecting Humans in Open Water." *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 2022. 
