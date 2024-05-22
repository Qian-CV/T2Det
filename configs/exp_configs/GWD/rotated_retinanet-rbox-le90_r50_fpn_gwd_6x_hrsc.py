_base_ = '../retinanet/rotated-retinanet-rbox-le90_r50_fpn_6x_hrsc.py'


model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',
                    interval=4,
                    max_keep_ckpts=6,
                    save_best='auto',
                    rule='greater'))
train_cfg = dict(type='EpochBasedTrainLoop', val_interval=4)
work_dir = './work_dirs/hrsc/gwd/rotated_retinanet-rbox-le90_r50_fpn_gwd_6x_hrsc.py/'
# work_dir = './work_dirs/shishi/'