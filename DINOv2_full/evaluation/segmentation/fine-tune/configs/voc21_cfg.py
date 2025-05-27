# dataset settings
# dataset_type = 'PascalVOCDataset' # 数据集类型，这将被用来定义数据集
dataset_type = 'PascalVOCDataset'
data_root = '/data/chenyinjie/CYJcode/data/VOCdevkit/VOC2012'

train_pipeline = [
    dict(type='LoadImageFromFile'), # 第1个流程，从文件路径里加载图像
    dict(type='LoadAnnotations'), # 第2个流程，对于当前图像，加载它的标注图像
    dict( # MMSeg 会同时对图像和分割 mask 做相同变换, 保证一一对应？
        type='RandomResize', # 调整输入图像大小(resize)和其标注图像的数据增广流程
        scale=(2048, 518), # 用于设定缩放比例的参考 -> 448
        ratio_range=(0.5, 2.0), # 数据增广的比例范围
        keep_ratio=True), # 调整图像大小时是否保持纵横比
    dict(type='RandomCrop', crop_size=(518, 518), cat_max_ratio=0.75), # 训练时的裁剪大小 448, 448
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'), # 光学上使用一些方法扭曲当前图像和其标注图像的数据增广流程
    dict(type="Pad", size=(518, 518), pad_val=0),
    dict(type='PackSegInputs') # 打包用于语义分割的输入数据
]

train_dataloader = dict(
    batch_size=8, # 16
    num_workers=4,
    persistent_workers=False, # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/train.txt',
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 518), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=(518, 518),
    mean=[123.675, 116.28, 103.53], # [123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375], # [58.395, 57.12, 57.375]
    # mean, std from NACLIP
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)


model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DistileedDINOv2', # DINOv2
        pretrained="/data/chenyinjie/CYJcode/distillation/DistillDINOv2/pretrained/facebook/dinov2-base", # "vit_base_patch14_dinov2.lvd142m"
        distilled_weight="/data/chenyinjie/CYJcode/traindistill/DINOv2_full/distilledweights/distilled_dinov2_weights_70.pth",
        patch_size=14,
        freeze_weights=True,
        out_indices=[8, 9, 10, 11],
        get_intermediates=False),
    decode_head=dict(
        type="BNHead",
        in_channels=[768], # does it need proj?, [768, 768, 768, 768]
        in_index=[-1], # [0, 1, 2, 3]
        input_transform='resize_concat',
        channels=768, # 768 + 768 + 768 + 768 = 3072
        dropout_ratio=0,
        num_classes=21,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(), # train_cfg 当前仅是一个占位符
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode="slide", crop_size=(518, 518), stride=(341, 341)),
    )


default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None # 从文件中加载检查点(checkpoint)
resume = False # 是否从已有的模型恢复


# optimizer
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR', # 调度流程的策略
        eta_min=0.0, # 训练结束时的最小学习率
        power=0.9, # 多项式衰减 (polynomial decay) 的幂
        begin=0, # 开始更新参数的时间步(step)
        end=20000, # 停止更新参数的时间步(step)
        by_epoch=False) # 是否按照 epoch 计算训练时间
]
# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))