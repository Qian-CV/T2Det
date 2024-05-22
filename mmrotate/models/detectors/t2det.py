# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dist import get_world_size
from mmengine.logging import print_log

from mmrotate.registry import MODELS
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmdet.models.detectors import RTMDet

import copy
import math
from typing import Tuple, Union
from torch import Tensor
from torch.nn.functional import grid_sample
from torchvision import transforms
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.models.utils import unpack_gt_instances
from mmrotate.structures.bbox import RotatedBoxes


@MODELS.register_module()
class T2Detector(RTMDet):
    """Implementation of RTMDet.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of ATSS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of ATSS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
        use_syncbn (bool): Whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 crop_size: Tuple[int, int] = (768, 768),
                 padding: str = 'reflection',
                 view_range: Tuple[float, float] = (0.25, 0.75),
                 sca_range: Tuple[float, float] = (0.5, 1.5),
                 sca_fact: float = 1.0,
                 prob_rot: float = 0.475,
                 prob_flp: float = 0.025,
                 random_transform: bool = False,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.crop_size = crop_size
        self.padding = padding
        self.view_range = view_range
        self.sca_range = sca_range
        self.sca_fact = sca_fact
        self.prob_rot = prob_rot
        self.prob_flp = prob_flp
        self.random_transform = random_transform

        # TODO: Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')

    def rotate_crop(
            self,
            batch_inputs: Tensor,
            rot: float = 0.,
            size: Tuple[int, int] = (768, 768),
            batch_gt_instances: InstanceList = None,
            padding: str = 'reflection') -> Tuple[Tensor, InstanceList]:
        """

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            rot (float): Angle of view rotation. Defaults to 0.
            size (tuple[int]): Crop size from image center.
                Defaults to (768, 768).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            padding (str): Padding method of image black edge.
                Defaults to 'reflection'.

        Returns:
            Processed batch_inputs (Tensor) and batch_gt_instances
            (list[:obj:`InstanceData`])
        """
        device = batch_inputs.device
        n, c, h, w = batch_inputs.shape
        size_h, size_w = size
        crop_h = (h - size_h) // 2
        crop_w = (w - size_w) // 2
        if rot != 0:
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = batch_inputs.new_tensor([[cosa, -sina], [sina, cosa]],
                                         dtype=torch.float)
            x_range = torch.linspace(-1, 1, w, device=device)
            y_range = torch.linspace(-1, 1, h, device=device)
            y, x = torch.meshgrid(y_range, x_range)
            grid = torch.stack([x, y], -1).expand([n, -1, -1, -1])
            grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2)
            # rotate
            batch_inputs = grid_sample(
                batch_inputs, grid, 'bilinear', padding, align_corners=True)
            if batch_gt_instances is not None:
                for i, gt_instances in enumerate(batch_gt_instances):
                    gt_bboxes = get_box_tensor(gt_instances.bboxes)
                    xy, wh, a = gt_bboxes[..., :2], gt_bboxes[
                        ..., 2:4], gt_bboxes[..., [4]]
                    ctr = tf.new_tensor([[w / 2, h / 2]])
                    xy = (xy - ctr).matmul(tf.T) + ctr
                    a = a + rot
                    rot_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                    batch_gt_instances[i].bboxes = RotatedBoxes(rot_gt_bboxes)
        batch_inputs = batch_inputs[..., crop_h:crop_h + size_h,
                                    crop_w:crop_w + size_w]
        if batch_gt_instances is None:
            return batch_inputs
        else:
            for i, gt_instances in enumerate(batch_gt_instances):
                gt_bboxes = get_box_tensor(gt_instances.bboxes)
                xy, wh, a = gt_bboxes[..., :2], gt_bboxes[...,
                                                          2:4], gt_bboxes[...,
                                                                          [4]]
                xy = xy - xy.new_tensor([[crop_w, crop_h]])
                crop_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                batch_gt_instances[i].bboxes = RotatedBoxes(crop_gt_bboxes)

            return batch_inputs, batch_gt_instances

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = unpack_gt_instances(batch_data_samples)
        batch_inputs_origin = batch_inputs

        if self.random_transform:
            # Crop original images and gts
            batch_inputs, batch_gt_instances = self.rotate_crop(
                batch_inputs, 0, self.crop_size, batch_gt_instances, self.padding)
            offset = 1
            for gt_instances in batch_gt_instances:
                gt_instances.bid = torch.arange(
                    0,
                    len(gt_instances.bboxes),
                    1,
                    device=gt_instances.bboxes.device) + offset + 0.2
                offset += len(gt_instances.bboxes)

            p = torch.rand(1)
            # Generate rotated images and gts
            if p < self.prob_rot:  # rot
                rot = math.pi * (
                    torch.rand(1, device=batch_inputs.device) *
                    (self.view_range[1] - self.view_range[0]) + self.view_range[0])
                batch_gt_rot = copy.deepcopy(batch_gt_instances)
                batch_inputs_rot, batch_gt_rot = self.rotate_crop(
                    batch_inputs, rot, self.crop_size, batch_gt_rot, self.padding)
                offset = 1
                for gt_instances in batch_gt_rot:
                    gt_instances.bid = torch.arange(
                        0,
                        len(gt_instances.bboxes),
                        1,
                        device=gt_instances.bboxes.device) + offset + 0.4
                    offset += len(gt_instances.bboxes)
                batch_inputs_all = torch.cat((batch_inputs, batch_inputs_rot))
                batch_gt_instances_all = batch_gt_instances + batch_gt_rot

            # Generate flipped images and gts
            elif p < self.prob_rot + self.prob_flp:  # flp
                batch_inputs_flp = transforms.functional.vflip(batch_inputs)
                batch_gt_flp = copy.deepcopy(batch_gt_instances)
                offset = 1
                for gt_instances in batch_gt_flp:
                    gt_instances.bboxes.flip_(batch_inputs.shape[2:4], 'vertical')
                    gt_instances.bid = torch.arange(
                        0,
                        len(gt_instances.bboxes),
                        1,
                        device=gt_instances.bboxes.device) + offset + 0.6
                    offset += len(gt_instances.bboxes)
                batch_inputs_all = torch.cat((batch_inputs, batch_inputs_flp))
                batch_gt_instances_all = batch_gt_instances + batch_gt_flp

            # Generate scaled images and gts
            else:
                sca = torch.rand(
                    1, device=batch_inputs.device
                ) * (self.sca_range[1] - self.sca_range[0]) + self.sca_range[0]
                size = (self.crop_size[0] / sca).long()
                batch_inputs_sca = transforms.functional.resized_crop(
                    batch_inputs,
                    0,
                    0,
                    size,
                    size,
                    self.crop_size,
                    antialias=False)
                batch_gt_sca = copy.deepcopy(batch_gt_instances)
                offset = 1
                for gt_instances in batch_gt_sca:
                    gt_instances.bboxes.rescale_((sca, sca))
                    gt_instances.bid = torch.arange(
                        0,
                        len(gt_instances.bboxes),
                        1,
                        device=gt_instances.bboxes.device) + offset + 0.8
                    offset += len(gt_instances.bboxes)
                # Concat original/rotated/flipped images and gts
                batch_inputs_all = torch.cat(
                    (batch_inputs, batch_inputs_sca))
                batch_gt_instances_all = batch_gt_instances + batch_gt_sca

            # 排除那些离图像边界太近的标签
            batch_gt_instances_filtered = []
            for gt_instances in batch_gt_instances_all:
                H = self.crop_size[0]
                D = 16
                ignore_mask = torch.logical_or(
                    gt_instances.bboxes.tensor[:, :2].min(1)[0] < D,
                    gt_instances.bboxes.tensor[:, :2].max(1)[0] > H - D)
                gt_instances_filtered = InstanceData()
                gt_instances_filtered.bboxes = RotatedBoxes(
                    gt_instances.bboxes.tensor[~ignore_mask])
                gt_instances_filtered.labels = gt_instances.labels[
                    ~ignore_mask]
                gt_instances_filtered.bid = gt_instances.bid[~ignore_mask]
                batch_gt_instances_filtered.append(gt_instances_filtered)

            batch_data_samples_all = []
            for idx, gt_instances in enumerate(batch_gt_instances_filtered):
                data_sample = DetDataSample()
                data_sample.gt_instances = gt_instances
                data_sample.set_metainfo(batch_img_metas[idx % len(batch_img_metas)])
                data_sample.ignored_instances = batch_gt_instances_ignore[idx % len(batch_img_metas)]
                batch_data_samples_all.append(data_sample)

            feat = self.extract_feat(batch_inputs_all)
            losses = self.bbox_head.loss(feat, batch_data_samples_all)

        else:
            feat = self.extract_feat(batch_inputs_origin)
            losses = self.bbox_head.loss(feat, batch_data_samples)
        return losses

