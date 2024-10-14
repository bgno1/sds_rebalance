
_base_ = './configs/cascade_rcnn/cascade-rcnn_x101-64x4d_fpn_1x_coco.py'

num_classes = 6

data_root = '../dataset/SeaDronesSee_balanced/'
dataset_type = 'CocoDataset'
metainfo = {
    'classes': ('swimmer', 'floater', 'boat',
                'swimmer on boat', 'floater on boat', 'life jacket')
}

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=num_classes,
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=num_classes,
            ),
            dict(type='Shared2FCBBoxHead',
                 num_classes=num_classes,
            )
        ]
    ),
    rpn_head=dict(
        anchor_generator=dict(
        type='AnchorGenerator',
        scales=[3,4],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64]),
    )
)

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1080, 1920), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1080, 1920), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1080, 1920), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type, 
        data_root=data_root,
        metainfo=metainfo,  
        ann_file='annotations/instances_train.json',
        pipeline=train_pipeline,
        data_prefix=dict(img='images/train/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        pipeline=val_pipeline,
        data_prefix=dict(img='images/val/')))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        pipeline=test_pipeline,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/')))

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val_iscrowd.json', classwise=True)
test_evaluator = dict(ann_file=data_root + 'annotations/instances_test.json')

# downloadable at https://mmdetection.readthedocs.io/en/latest/model_zoo.html
# load_from = './cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth'

