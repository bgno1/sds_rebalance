import pickle
import json
import torch

def convert_to_serializable(data):
    """Recursively convert data to a JSON serializable format."""
    if isinstance(data, torch.Tensor):
        return data.cpu().tolist()
    elif isinstance(data, dict):
        return {key: convert_to_serializable(val) for key, val in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(val) for val in data]
    return data

def convert_bbox_format(bbox):
    """convert bbox format from (x_min, y_min, x_max, y_max) to (x_min, y_min, width, height)."""
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]

# Load the pickle file
file_path = 'test.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Convert the data into the specified list format
serializable_output_list = []
for item in data:
    # image id
    img_id = item.get('img_id')
    # pred instances: 'labels', 'scores', 'bboxes'
    pred_instances = item.get('pred_instances', {})
    if pred_instances:
        labels = pred_instances.get('labels', [])
        scores = pred_instances.get('scores', [])
        bboxes = pred_instances.get('bboxes', [])

        for label, score, bbox in zip(labels, scores, bboxes):
            bbox = convert_bbox_format(bbox)  # convert bbox from (xmin, ymin, xmax, ymax) to (xmin, ymin, w, h)
            result = {
                "image_id": img_id,
                "category_id": label + 1,  # label 0 corresponds to category_id 1
                "score": score,
                "bbox": bbox
            }
            result = convert_to_serializable(result)
            serializable_output_list.append(result)

with open('test.json', 'w') as file:
    json.dump(serializable_output_list, file)
