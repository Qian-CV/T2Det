_base_ = ['../../../../rotated_rtmdet/_base_/default_runtime.py', '../../../../rotated_rtmdet/_base_/schedule_6x.py',
          '../../../../_base_/datasets/vedai.py']

# training schedule, hrsc dataset is repeated 3 times, in
# `./_base_/hrsc_rr.py`, so the actual epoch = 3 * 3 * 12 = 9 * 12
max_epochs = 6 * 12
# hrsc dataset use larger learning rate for better performance
base_lr = 0.004 / 8
interval = 4  # 最初是12
angle_version = 'le90'
checkpoint = '/media/ubuntu/nvidia/wlq/part2/mmrotate/tools/data/weight/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth' # noqa
# fp16 = dict(loss_scale='dynamic')
use_ss_branch = False
model = dict(
    type='T2Detector',
    random_transform=use_ss_branch,
    crop_size=(1024, 1024),
    view_range=(0.25, 0.75),
    prob_rot=0.95 * 0.7,
    prob_flp=0.05 * 0.7,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        boxtype2tensor=False,
        batch_augments=None),
    backbone=dict(
        type='mmdet.CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        type='mmdet.CSPNeXtPAFPN',
        in_channels=[192, 384, 768],
        out_channels=192,
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    bbox_head=dict(
        type='T2DetHead',
        num_classes=6,
        in_channels=192,
        stacked_convs=2,
        feat_channels=192,
        angle_version=angle_version,
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        # angle_coder=dict(
        #     type='PSCCoder',
        #     angle_version=angle_version,
        #     dual_freq=False,
        #     num_step=3,
        #     thr_mod=0),
        use_reweighted_loss=False,
        use_ss_branch=use_ss_branch,
        ss_loss_start=0.4,
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        loss_angle=None,
        loss_symmetry_ss=dict(
            type='T2DetConsistencyLoss',
            use_snap_loss=True,
            loss_rot=dict(
                type='mmdet.SmoothL1Loss', loss_weight=0.1, beta=0.1),
            loss_flp=dict(
                type='mmdet.SmoothL1Loss', loss_weight=0.1, beta=0.1)),
        loss_scale_ss=dict(type='mmdet.SmoothL1Loss', loss_weight=0.1, beta=0.1),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        use_hbbox_loss=False,
        scale_angle=False,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.DynamicSoftLabelAssigner',
            iou_calculator=dict(type='RBboxOverlaps2D'),
            topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000),
)


# learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=1.0e-5,
#         by_epoch=False,
#         begin=0,
#         end=1000),
#     dict(
#         # use cosine lr from 54 to 108 epoch
#         type='CosineAnnealingLR',
#         eta_min=base_lr * 0.05,
#         begin=max_epochs // 2,
#         end=max_epochs,
#         T_max=max_epochs // 2,
#         by_epoch=True,
#         convert_to_iter_based=True),
# ]

# batch_size = (1 GPUs) x (8 samples per GPU) = 8
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=8)
test_dataloader = dict(batch_size=8, num_workers=8)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    # accumulative_counts=2,
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',
                    interval=interval,
                    max_keep_ckpts=6,
                    save_best='auto',
                    rule='greater'))
train_cfg = dict(type='EpochBasedTrainLoop', val_interval=interval)
work_dir = './work_dirs/ablation/exp1/t2det_rtmdet_m-6x-vedai_divided-8_bs4_no-scare/'
# work_dir = './work_dirs/shishi/'