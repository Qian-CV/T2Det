_base_ = [
    '../../_base_/datasets/hrsc.py', '../../_base_/schedules/schedule_12x.py',
    '../../_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='T2Detector',
    crop_size=(800, 800),
    view_range=(0.25, 0.75),
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
        type='T2DetHead',
        num_classes=1,
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
        scale_angle=False,
        use_circumiou_loss=False,
        use_standalone_angle=False,
        use_reweighted_loss_bbox=True,
        angle_coder=dict(
            type='PSCCoder',
            angle_version=angle_version,
            dual_freq=False,
            num_step=3,
            thr_mod=0),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=1.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_symmetry_ss=dict(
            type='H2RBoxV2ConsistencyLoss',
            use_snap_loss=True,
            loss_rot=dict(
                type='mmdet.SmoothL1Loss', loss_weight=1.0, beta=0.1),
            loss_flp=dict(
                type='mmdet.SmoothL1Loss', loss_weight=0.05, beta=0.1)),
        loss_scale_ss=dict(type='mmdet.GIoULoss', loss_weight=0.02)),
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
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True),
    # # Horizontal GTBox, (x1,y1,x2,y2)
    # dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    # Horizontal GTBox, (x,y,w,h,theta)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=0.5, angle_range=180),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True),
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
    iou_thrs=[0.5])

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005 / 2,
        betas=(0.9, 0.999),
        weight_decay=0.005))


# batch_size = (2 GPUs) x (4 samples per GPU) = 8
train_dataloader = dict(dataset=dict(pipeline=train_pipeline), batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=8)
test_dataloader  = dict(batch_size=8, num_workers=8)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',
                    interval=4,
                    max_keep_ckpts=6,
                    save_best='auto',
                    rule='greater'))
train_cfg = dict(
    type='EpochBasedTrainLoop', val_interval=4)
work_dir = './work_dirs/hrsc/t2det/t2det-le90_r50_fpn_rr-12x_hrsc/'
# work_dir = './work_dirs/shishi/'