_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_coco_stuff.txt',
    slide_stride=112,
    slide_crop=336
)

# dataset settings
dataset_type = 'COCOStuffDataset'
data_root = '' # Your COCO Stuff dataset path

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))
