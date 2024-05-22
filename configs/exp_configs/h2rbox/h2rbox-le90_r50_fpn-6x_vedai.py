_base_ = [
    '../../_base_/datasets/vedai.py', '../../_base_/schedules/schedule_6x.py',
    '../../_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='H2RBoxDetector',
    crop_size=(1024, 1024),
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='H2RBoxHead',
        num_classes=6,
        in_channels=256,
        angle_version='le90',
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        use_hbbox_loss=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        square_classes=[9, 11],
        crop_size=(1024, 1024),
        loss_bbox_ss=dict(
            type='H2RBoxConsistencyLoss',
            loss_weight=0.4,
            center_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=0.0),
            shape_loss_cfg=dict(type='mmdet.IoULoss', loss_weight=1.0),
            angle_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=1.0))),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# load hbox annotations
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmdet.FixShapeResize', width=1024, height=1024, keep_ratio=True),
    # Horizontal GTBox, (x1,y1,x2,y2)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    # Horizontal GTBox, (x,y,w,h,theta)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.FixShapeResize', width=1024, height=1024, keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_evaluator = dict(
    type='DOTAMetric',
    metric='mAP',
    # iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    iou_thrs=[0.5])

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.005))

# batch_size = (2 GPUs) x (4 samples per GPU) = 8
train_dataloader = dict(dataset=dict(pipeline=train_pipeline), batch_size=4, num_workers=4)
val_dataloader = dict(dataset=dict(pipeline=val_pipeline), batch_size=4, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',
                    interval=4,
                    max_keep_ckpts=6,
                    save_best='auto',
                    rule='greater'))
train_cfg = dict(
    type='EpochBasedTrainLoop', val_interval=4)
work_dir = './work_dirs/vedai/h2rbox/h2rbox-le90_r50_fpn-6x_vedai/'
# work_dir = './work_dirs/shishi/'