## Adding GhostBlock to YOLOv8

To integrate the `GhostBlock` module into the YOLOv8 project, follow these steps:

1. Import `GhostBlock` in `ultralytics/nn/modules/tasks.py`:

2. Add the following `elif` branch code to the `parse_model` function in `ultralytics/nn/modules/tasks.py`:

```
elif m is GhostBlock:
    c1, c2 = ch[f], args[0]
    if c2 != nc:  # if not output
        c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2, *args[1:]]
    args.insert(2, n)  # number of repeats
    n = 1
```

## Adding WIoU Loss to YOLOv8

To integrate the `WIoU` loss function into the YOLOv8 project, follow these steps:

1. Import `wiou_loss` in `ultralytics/utils/loss.py`:

2. Modify the `forward` function in the `BboxLoss` class in `ultralytics/utils/loss.py` by adding the following code:
   
   ```
   if config.bbox_loss == 'WIoU':
       loss, iou = bbox_wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
       loss_iou = (loss * weight).sum() / target_scores_sum
   else:   # CIoU
       iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
       loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
   ```
