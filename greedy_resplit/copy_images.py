import os
import shutil
import json

# Define file paths
train_json_path = './instances_train_balanced.json'
val_json_path = './instances_val_balanced.json'
source_folder = './train_val'
train_dest_folder = '../dataset/SeaDronesSee_balanced/images/train'
val_dest_folder = '../datasets/SeaDronesSee_balanced/images/val'

# Ensure destination folders exist
os.makedirs(train_dest_folder, exist_ok=True)
os.makedirs(val_dest_folder, exist_ok=True)

# Load and copy training set images
with open(train_json_path, 'r') as f:
    train_data = json.load(f)
    train_images = [img['file_name'] for img in train_data['images']]
    for img_name in train_images:
        src_path = os.path.join(source_folder, img_name)
        dst_path = os.path.join(train_dest_folder, img_name)
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")

# Load and copy validation set images
with open(val_json_path, 'r') as f:
    val_data = json.load(f)
    val_images = [img['file_name'] for img in val_data['images']]
    for img_name in val_images:
        src_path = os.path.join(source_folder, img_name)
        dst_path = os.path.join(val_dest_folder, img_name)
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")
