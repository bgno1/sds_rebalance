import json
from collections import defaultdict


def calculate_ratios(image_annotations, all_images):
    """
    Calculate various statistics and ratios for each category in the dataset.

    Args:
        image_annotations (dict): A dictionary where keys are image IDs and values are lists of annotations.
        all_images (dict): A dictionary where keys are image IDs and values are image metadata.

    Returns:
        dict: A dictionary containing calculated ratios and statistics for each category.
    """
    category_stats = defaultdict(lambda: {'train': 0, 'val': 0, 'train_area': 0, 'val_area': 0,
                                          'train_min_area': float('inf'), 'val_min_area': float('inf'),
                                          'train_max_area': 0, 'val_max_area': 0})

    for img_id, anns in image_annotations.items():
        for ann in anns:
            cat_id = ann['category_id']
            area = ann['area']
            if all_images[img_id]['split'] == 'train':
                category_stats[cat_id]['train'] += 1
                category_stats[cat_id]['train_area'] += area
                category_stats[cat_id]['train_min_area'] = min(category_stats[cat_id]['train_min_area'], area)
                category_stats[cat_id]['train_max_area'] = max(category_stats[cat_id]['train_max_area'], area)
            elif all_images[img_id]['split'] == 'val':
                category_stats[cat_id]['val'] += 1
                category_stats[cat_id]['val_area'] += area
                category_stats[cat_id]['val_min_area'] = min(category_stats[cat_id]['val_min_area'], area)
                category_stats[cat_id]['val_max_area'] = max(category_stats[cat_id]['val_max_area'], area)

    ratios = {}
    for cat_id, stats in category_stats.items():
        train_count = stats['train']
        val_count = stats['val']
        train_area = stats['train_area']
        val_area = stats['val_area']

        train_avg_area = train_area / train_count if train_count > 0 else 0
        val_avg_area = val_area / val_count if val_count > 0 else 0

        count_ratio = train_count / val_count if val_count > 0 else float('inf')
        min_area_ratio = stats['train_min_area'] / stats['val_min_area'] if stats['val_min_area'] > 0 else float('inf')
        max_area_ratio = stats['train_max_area'] / stats['val_max_area'] if stats['val_max_area'] > 0 else float('inf')
        avg_area_ratio = train_avg_area / val_avg_area if val_avg_area > 0 else float('inf')

        ratios[cat_id] = {
            'cat_id': cat_id,
            'train_count': train_count,
            'val_count': val_count,
            'train_min_area': stats['train_min_area'],
            'val_min_area': stats['val_min_area'],
            'train_max_area': stats['train_max_area'],
            'val_max_area': stats['val_max_area'],
            'train_avg_area': train_avg_area,
            'val_avg_area': val_avg_area,
            'count_ratio': count_ratio,
            'min_area_ratio': min_area_ratio,
            'max_area_ratio': max_area_ratio,
            'avg_area_ratio': avg_area_ratio,
        }
    return ratios


def update_category_stats(image_id, split):
    """
    Update category statistics based on the image split (train/val).

    Args:
        image_id (int): The ID of the image being processed.
        split (str): The split type ('train' or 'val').
    """
    anns = image_annotations[image_id]
    for ann in anns:
        cat_id = ann['category_id']
        category_stats[cat_id][split] += ann['area']


# Get a list of images containing a specific category and sort them by area
def get_candidate_images(category_id, exclude_images):
    """
    Retrieve and sort images containing a specific category by the area of annotations.

    Args:
        category_id (int): The ID of the category being processed.
        exclude_images (set): A set of image IDs to exclude from processing.

    Returns:
        list: A sorted list of tuples containing image IDs and total annotation area.
    """
    candidate_images = []
    for img_id, anns in image_annotations.items():
        if img_id not in exclude_images:
            if any(ann['category_id'] == category_id for ann in anns):
                total_area = sum(ann['area'] for ann in anns if ann['category_id'] == category_id)
                candidate_images.append((img_id, total_area))
    return sorted(candidate_images, key=lambda x: x[1], reverse=True)


# Load data
with open('./instances_train.json', 'r') as train_file:
    train_data = json.load(train_file)

with open('./instances_val.json', 'r') as val_file:
    val_data = json.load(val_file)

# Merge datasets
all_images = {img['id']: img for img in train_data['images'] + val_data['images']}
all_annotations = train_data['annotations'] + val_data['annotations']

# Group annotations by image
image_annotations = defaultdict(list)
for ann in all_annotations:
    image_annotations[ann['image_id']].append(ann)

# Initialize category statistics
category_stats = defaultdict(lambda: {'train': 0, 'val': 0})
image_splits = {img_id: None for img_id in all_images}
exclude_images = set()  # Initialize set of assigned images

# Add default 'split' key to all images
for img_id in all_images:
    all_images[img_id]['split'] = None

# Category priority
categories_priority = [6, 4, 5, 1, 2, 3]  # life jacket > swimmer on boat > floater on boat > swimmer > floater > boat

# Define the number of iterations for each category
category_loops = {
    1: 4,  # swimmer
    2: 4,  # floater
    3: 4,  # boat
    4: 5,  # swimmer on boat
    5: 4,  # floater on boat
    6: 6  # life jacket
}

# Greedy re-split
for cat_id in categories_priority:
    print(f"Processing category {cat_id}")
    current_ratios = calculate_ratios(image_annotations, all_images)

    while True:
        # Assign to validation set
        candidate_images = get_candidate_images(cat_id, exclude_images)
        if not candidate_images:
            break
        median_index = len(candidate_images) // 2
        if category_stats[cat_id]['val'] <= category_stats[cat_id]['train']:
            img_id, area = candidate_images[median_index - 1] if median_index > 0 else candidate_images[0]
        else:
            img_id, area = candidate_images[median_index] if median_index < len(candidate_images) - 1 else \
            candidate_images[-1]
        exclude_images.add(img_id)
        image_splits[img_id] = 'val'
        all_images[img_id]['split'] = 'val'
        update_category_stats(img_id, 'val')

        # Assign to training set
        loop_count = category_loops.get(cat_id, 4)  # Default number of iterations per category is 4
        for _ in range(loop_count):
            candidate_images = get_candidate_images(cat_id, exclude_images)
            if not candidate_images:
                break
            median_index = len(candidate_images) // 2
            if category_stats[cat_id]['train'] <= category_stats[cat_id]['val']:
                img_id, area = candidate_images[median_index - 1] if median_index > 0 else candidate_images[0]
            else:
                img_id, area = candidate_images[median_index] if median_index < len(candidate_images) - 1 else \
                candidate_images[-1]
            exclude_images.add(img_id)
            image_splits[img_id] = 'train'
            all_images[img_id]['split'] = 'train'
            update_category_stats(img_id, 'train')

# Assign remaining images
for img_id in image_splits:
    if image_splits[img_id] is None:
        print(f'unassigned image id {img_id}')
        image_splits[img_id] = 'train' if category_stats[cat_id]['train'] <= category_stats[cat_id]['val'] else 'val'
        update_category_stats(img_id, image_splits[img_id])

# Output final results
train_images = [img for img_id, img in all_images.items() if image_splits[img_id] == 'train']
val_images = [img for img_id, img in all_images.items() if image_splits[img_id] == 'val']

print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")

# Save re-split results
with open('./instances_train_balanced.json', 'w') as train_file:
    json.dump({'images': train_images,
               'annotations': [ann for ann in all_annotations if image_splits[ann['image_id']] == 'train']}, train_file)

with open('./instances_val_balanced.json', 'w') as val_file:
    json.dump({'images': val_images,
               'annotations': [ann for ann in all_annotations if image_splits[ann['image_id']] == 'val']}, val_file)
