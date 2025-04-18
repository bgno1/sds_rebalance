[![DOI](https://zenodo.org/badge/864396002.svg)](https://doi.org/10.5281/zenodo.13894509)

# Optimizing Drone-Captured Maritime Rescue Image Object Detection through Dataset Rebalancing under Sample Constraints

## 1. Overview

This repository contains the code supporting the findings in the paper *"Optimizing Drone-Captured Maritime Rescue Image Object Detection through Dataset Rebalancing under Sample Constraints,"* which is currently under peer review. The SeaDronesSee dataset [1], along with its extended version SeaDronesSee v2 [2], is a highly valuable and unique resource for drone-based maritime search and rescue research, providing critical data for advancing object detection models in this field. However, due to the challenging nature of collecting data in such specialized and difficult environments, the datasets inherently face sample constraints and imbalances in the distribution of certain categories.

Our work seeks to address these limitations by proposing a greedy algorithm-based method for re-partitioning the datasets. This approach aims to enhance the generalization and training efficiency of detection models while making the most of the invaluable data contained within the SeaDronesSee series datasets. Ultimately, our goal is to maximize the impact of these valuable resources, ensuring they can be fully leveraged to improve drone-based maritime rescue operations.

## 2. SeaDronesSee series Datasets

The SeaDronesSee v1 and v2 datasets , essential for maritime search and rescue object detection tasks, can be downloaded directly from the official repository: [Download SeaDronesSee v1 Dataset](https://cloud.cs.uni-tuebingen.de/index.php/s/aJQPHLGnke68M52), and [SeaDronesSee v2 Dataset](https://cloud.cs.uni-tuebingen.de/index.php/s/ZZxX65FGnQ8zjBP). For further details please refer to the official SeaDronesSee website: [https://macvi.org/](https://macvi.org/).

## 3. Dataset re-partitioning based on the greedy algorithm

The code has been tested using Python 3.8.

To execute the rebalancing process, run the `greedy_resplit.py` script located in the `greedy_resplit` folder of this repository.

This script requires a dataset version argument—either `sdsv1` or `sdsv2`—which specifies whether to perform rebalancing on the SeaDronesSee v1 or v2 annotation files.

Use the following command in your terminal:

```bash
python greedy_resplit.py --dataset sdsv1
```

or

```bash
python greedy_resplit.py --dataset sdsv2
```



**Input Files**

This script requires the original SeaDronesSee dataset annotation files as inputs:

- `./instances_train.json` – Original training set annotations from the SeaDronesSee v1 or v2 dataset.
- `./instances_val.json` – Original validation set annotations from the SeaDronesSee v1 or v2 dataset.

Ensure these files are downloaded and placed in the correct directory before running the script. You can obtain them from the official [SeaDronesSee v1](https://cloud.cs.uni-tuebingen.de/index.php/s/aJQPHLGnke68M52) or [SeaDronesSee v2](https://cloud.cs.uni-tuebingen.de/index.php/s/ZZxX65FGnQ8zjBP) dataset link.

**Output Files**

After running the script, you will find the rebalanced annotation files in the same directory as the input files. The output filenames depend on the dataset version you specify:

- For SeaDronesSee v1:
  
  - `./instances_train_balanced.json` – Rebalanced training set annotations.
  
  - `./instances_val_balanced.json` – Rebalanced validation set annotations.

- For SeaDronesSee v2:
  
  - `./instances_train_balanced_v2.json` – Rebalanced training set annotations.
  
  - `./instances_val_balanced_v2.json` – Rebalanced validation set annotations.



## 4. Model Validation

### 4.1 Dataset Preparation

- Download the SeaDronesSee v1 and v2 datasets from the following links: [SeaDronesSee v1](https://cloud.cs.uni-tuebingen.de/index.php/s/aJQPHLGnke68M52) and [SeaDronesSee v2](https://cloud.cs.uni-tuebingen.de/index.php/s/ZZxX65FGnQ8zjBP). Then, copy the contents of the `images` folder from each dataset into the corresponding directory in this repository: use  `./dataset/SeaDronesSee` for v1 and `./dataset/SeaDronesSee_v2` for v2.

- Create a `dataset/SeaDronesSee/images/train_val` directory and copy all the images from the `train` and `val` folders of the v1 dataset into this newly created folder. Similarly, for the v2 dataset, create a `dataset/SeaDronesSee_v2/images/train_val` directory and copy all the images from its `train` and `val` folders into that folder.

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
        - labels/         # (for YOLOv8 and YOLO11) Annotations for images
        - test.txt        # (for YOLOv8 and YOLO11) List of test images
        - train.txt       # (for YOLOv8 and YOLO11) List of training images
        - val.txt         # (for YOLOv8 and YOLO11) List of validation images
    - SeaDronesSee_v2/
        - annotations/    # Annotation files in JSON format
        - images/         # Image data
            - test/
            - train/
            - val/
            - train_val/  # Contains all images from both train and val folders
        - labels/         # (for YOLOv8 and YOLO11) Annotations for images
        - test.txt        # (for YOLOv8 and YOLO11) List of test images
        - train.txt       # (for YOLOv8 and YOLO11) List of training images
        - val.txt         # (for YOLOv8 and YOLO11) List of validation images
```

- Run the `./greedy_resplit/copy_images.py` script to organize images into rebalanced training and validation sets based on the dataset version. Use the `--dataset` argument to specify whether to process the SeaDronesSee v1 or v2 dataset:
  
  ```bash
  python ./greedy_resplit/copy_images.py --dataset sdsv1
  ```
  
  or
  
  ```bash
  python ./greedy_resplit/copy_images.py --dataset sdsv2
  ```
  
  This script reads the corresponding rebalanced annotation files and copies images from the `train_val` folder into the appropriate `train` and `val` directories under:
  
  - `./dataset/SeaDronesSee_balanced` for v1
  
  - `./dataset/SeaDronesSee_v2_balanced` for v2

- Copy the `images/test` folder from the official SeaDronesSee datasets into the corresponding balanced dataset directories in this repository:
  
  - For v1, copy it to `./dataset/SeaDronesSee_balanced/images`
  
  - For v2, copy it to `./dataset/SeaDronesSee_v2_balanced/images`

The final directory structure of the dataset should look as follows:

```md
- dataset/
    - SeaDronesSee/           # Original SeaDronesSee dataset
        - annotations/        # Annotation files in JSON format
        - images/             # Image data
            - test/           
            - train/          # original training set
            - val/            # original validation set
        - labels/             # (for YOLOv8 and YOLO11) Annotations for images
        - test.txt            # (for YOLOv8 and YOLO11) List of test images
        - train.txt           # (for YOLOv8 and YOLO11) List of training images
        - val.txt             # (for YOLOv8 and YOLO11) List of validation images
    - SeaDronesSee_balanced/  # Rebalanced version of the SeaDronesSee dataset
        - annotations/        # Annotation files in JSON format
        - images/             # Image data
            - test/
            - train/          # rebalanced training set
            - val/            # rebalanced validation set
        - labels/             # (for YOLOv8 and YOLO11) Annotations for images
        - test.txt            # (for YOLOv8 and YOLO11) List of test images
        - train.txt           # (for YOLOv8 and YOLO11) List of training images
        - val.txt             # (for YOLOv8 and YOLO11) List of validation images
    - SeaDronesSee_v2/           # Original SeaDronesSee v2 dataset
        - annotations/        # Annotation files in JSON format
        - images/             # Image data
            - test/           
            - train/          # original training set
            - val/            # original validation set
        - labels/             # (for YOLOv8 and YOLO11) Annotations for images
        - test.txt            # (for YOLOv8 and YOLO11) List of test images
        - train.txt           # (for YOLOv8 and YOLO11) List of training images
        - val.txt             # (for YOLOv8 and YOLO11) List of validation images
    - SeaDronesSee_v2_balanced/  # Rebalanced version of the SeaDronesSee v2 dataset
        - annotations/        # Annotation files in JSON format
        - images/             # Image data
            - test/
            - train/          # rebalanced training set
            - val/            # rebalanced validation set
        - labels/             # (for YOLOv8 and YOLO11) Annotations for images
        - test.txt            # (for YOLOv8 and YOLO11) List of test images
        - train.txt           # (for YOLOv8 and YOLO11) List of training images
        - val.txt             # (for YOLOv8 and YOLO11) List of validation images
```

The contents of the `SeaDronesSee/images` , `SeaDronesSee_balanced/images`, `SeaDronesSee_v2/images`  and  `SeaDronesSee_v2_balanced/images` folders need to be downloaded and generated through the steps mentioned above. All other files and folders are already included in this repository.

### 4.2 Faster R-CNN and Cascade R-CNN

#### 4.2.1 Training

**Download and prepare the MMDetection project**

- Download the MMDetection project from the [official MMDetection GitHub repository](https://github.com/open-mmlab/mmdetection), and copy the contents of the `mmdetection-main` folder into this repository's `mmdetection` folder. Follow the official guidelines to install the necessary dependencies.

The relevant directory structure will look as follows:

```md
- mmdetection
    - tools/
    - configs/
    ...
    frx.py
    frx_bl.py
    frx_v2.py
    frx_v2_bl.py
    cas.py
    cas_bl.py
    cas_v2.py
    cas_v2_bl.py
```

Among these files, `frx.py`, `frx_bl.py`, `cas.py`, and `cas_bl.py` (as well as those with "v2" in their names) are custom model training scripts provided in our repository to support the experiments in the paper (files with `v2` in the name are for the SeaDronesSee v2 dataset; others are for v1). Specifically:

- `frx.py` and `frx_bl.py` are used to train the Faster R-CNN models on the original and rebalanced SeaDronesSee v1 datasets, respectively.
- `cas.py` and `cas_bl.py` are used to train the Cascade R-CNN models  on the original and rebalanced SeaDronesSee v2 datasets, respectively.
- Files with the suffix `_v2` and `_v2_bl` follow the same naming convention and are used for training on the SeaDronesSee v2 dataset (original and rebalanced versions, respectively).

**Example usage:**

- To train the Faster R-CNN model on the original SeaDronesSee v1 dataset using `frx.py`, navigate to the `mmdetection` directory and execute the following command:

```bash
python ./tools/train.py ./frx.py
```

- To train the Faster R-CNN model on the rebalanced SeaDronesSee v1 dataset using `frx_bl.py`, navigate to the `mmdetection` directory and execute the following command:

```bash
python ./tools/train.py ./frx_bl.py
```

**Note: Regarding Loading Pretrained Weights**

In the files `frx.py`, `frx_bl.py`, `cas.py`, and `cas_bl.py` (as well as those with "v2" in their names), the last line of code that loads the pretrained weights has been commented out, as it requires downloading large `.pth` files. If you wish to train models with higher accuracy using pretrained weights, you can download the corresponding `.pth` files from the [Benchmark and Model Zoo — MMDetection 3.3.0 documentation](https://mmdetection.readthedocs.io/en/latest/model_zoo.html), place them in the `mmdetection` folder, and uncomment the last line in each of the files.

#### 4.2.2 Test on SeaDronesSee Leaderboard

- Use MMDetection’s test script to generate prediction files in PKL format. For example, to generate predictions using the model trained with `frx.py`, run the following command:

```bash
python ./tools/test.py ./frx.py ./work_dirs/frx/epoch_12.pth --out ./test.pkl
```

- Convert the `test.pkl` file generated in the previous step to the JSON format required by the SeaDronesSee Leaderboard using the `mmdetection/to_sds_json.py` script provided in this repository.

- Validate the model’s generalization performance by submitting the JSON file to the SeaDronesSee Leaderboard server at [SeaDronesSee Leaderboard](https://macvi.org/leaderboard/airborne/seadronessee/object-detection).

### 4.3 YOLOv8 and YOLO11

#### 4.3.1 Training

Download the YOLO project from the [official Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics), the following directory structure will be available:

```md
- ultralytics
    - docs/
    ...
    - ultralytics/
        - cfg/
        - data/
        ...
```

Using the Ultralytics official documentation as a reference, create a Python training script in the `ultralytics` directory with the following structure, for training the YOLO11 model on the SeaDronesSee v1 rebalanced dataset:

```python
model = YOLO(r'./yolo11_sds.yaml')  # Define the YOLOv8 model architecture via a YAML configuration file
results = model.train(
    data = 'sds_balanced.yaml',  # Define the dataset structure and location via a YAML configuration file
    workers=12,
    batch=48
)
```

Replace `sds_balanced.yaml` with `sds.yaml` in the code to train on the original SeaDronesSee v1 dataset:

```python
model = YOLO(r'./yolo11_sds.yaml')  # Define the YOLOv8 model architecture via a YAML configuration file
results = model.train(
    data = 'sds.yaml',  # Define the original dataset structure and location via a YAML configuration file
    workers=12,
    batch=48
)
```

In summary, the argument passed to the `YOLO(...)` function can be one of the following:

- `yolov8_sds.yaml`, `yolov8_sds2.yaml`, `yolo11_sds.yaml`, or `yolo11_sds2.yaml`  
  These YAML files define the architecture of different YOLO versions (YOLOv8 or YOLO11) for the corresponding SeaDronesSee v1 or v2 datasets.

The `data` argument passed to `model.train(...)` should match the dataset split and version being used, and can be one of:

- `sds.yaml`, `sds_balanced.yaml`, `sds2.yaml`, or `sds2_balanced.yaml`  
  These files define the dataset structure and annotation paths for the original or rebalanced SeaDronesSee v1 and v2 datasets.

Key details:

- The `workers` and `batch` parameters should be adjusted based on your hardware, particularly the GPU. Setting these values too high on low-spec machines may lead to crashes. Additionally, these parameters impact both training speed and model performance.

- All of the YAML files mentioned above can be found in the `yolo` directory of this repository and can be modified as needed.

In addition, YOLOv8 and YOLO11 uses the CIoU loss function by default. To use the WIoU loss function as mentioned in the paper, the following changes should be made:

- Import `wiou_loss` (the `wiou_loss.py` file can be found in the `yolo` directory of this repository) into `ultralytics/utils/loss.py`.

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

[2] Kiefer, Benjamin, et al. "1st workshop on maritime computer vision (macvi) 2023: Challenge results." *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*. 2023.
