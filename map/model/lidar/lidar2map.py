import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from ..utils.base import BEV_FPD
from .pointpillar import PointPillarEncoder
from ..camera.lift_splat import LiftSplat
from ..fusion.bevfusion.bevfusion import BEVFusion_lidar
from ..loss.kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from ..loss.kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D


def compute_feature_distill_loss(features_preds, features_targets, low_feats):
    weight = 0.4
    mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
    tree_filter_layers = TreeFilter2D(groups=1, sigma=0.02)
    feature_distill_loss = 0.0
    resize_shape = features_preds[-4].shape[-2:]
    if isinstance(features_preds, list):
        for i in range(len(features_preds)):  # 1/8   1/16   1/32
            feature_target = features_targets[i].detach()
            feature_pred = features_preds[i]

            B, C, H, W = feature_pred.shape
            feature_pred_down = F.interpolate(feature_pred, size=resize_shape, mode="bilinear")
            feature_target_down = F.interpolate(feature_target, size=resize_shape, mode="bilinear")
            low_feats = F.interpolate(low_feats, size=resize_shape, mode="bilinear")

            tree = mst_layers(low_feats)
            fusion_affinity = tree_filter_layers(feature_in=feature_target_down,
                                                 embed_in=low_feats, tree=tree, low_tree=False)

            tree = mst_layers(feature_target_down)
            fusion_affinity = tree_filter_layers(feature_in=fusion_affinity,
                                                 embed_in=feature_target_down, tree=tree, low_tree=False)

            tree = mst_layers(low_feats)
            lidar_affinity = tree_filter_layers(feature_in=feature_pred_down,
                                                embed_in=low_feats, tree=tree, low_tree=False)

            tree = mst_layers(feature_target_down)
            lidar_affinity = tree_filter_layers(feature_in=lidar_affinity,
                                                embed_in=feature_pred_down, tree=tree, low_tree=False)

            feature_distill_loss = feature_distill_loss + F.l1_loss(lidar_affinity, fusion_affinity, reduction='mean') / B

    else:
        feature_target = features_targets.detach()
        feature_pred = features_preds

        B, C, H, W = feature_pred.shape
        feature_pred_down = F.interpolate(feature_pred, size=resize_shape, mode="bilinear", align_corners=False)
        feature_target_down = F.interpolate(feature_target, size=resize_shape, mode="bilinear", align_corners=False)
        low_feats = F.interpolate(low_feats, size=resize_shape, mode="bilinear")

        tree = mst_layers(low_feats)
        fusion_affinity = tree_filter_layers(feature_in=feature_target_down,
                                             embed_in=low_feats, tree=tree)

        tree = mst_layers(feature_target_down)
        fusion_affinity = tree_filter_layers(feature_in=fusion_affinity,
                                             embed_in=feature_target_down, tree=tree, low_tree=False)

        tree = mst_layers(low_feats)
        lidar_affinity = tree_filter_layers(feature_in=feature_pred_down,
                                            embed_in=low_feats, tree=tree)

        tree = mst_layers(feature_target_down)
        lidar_affinity = tree_filter_layers(feature_in=lidar_affinity,
                                            embed_in=feature_pred_down, tree=tree, low_tree=False)

        feature_distill_loss = feature_distill_loss + F.l1_loss(lidar_affinity, fusion_affinity, reduction='mean') / B

    return weight * feature_distill_loss


def compute_logit_distill_loss(logits_preds, logits_targets):
    weight = 1.5
    logit_distill_loss = 0.0
    if isinstance(logits_preds, list):
        for i in range(len(logits_preds)):
            preds_temp = logits_preds[i]
            targets_temp = logits_targets[i]
            logit_distill_loss = logit_distill_loss + weight * F.kl_div(F.log_softmax(preds_temp, dim=1),
                                                                        F.softmax(targets_temp.detach(), dim=1),
                                                                        reduction='none').sum(1).mean()
    else:
        logit_distill_loss = weight * F.kl_div(F.log_softmax(logits_preds, dim=1),
                                               F.softmax(logits_targets.detach(), dim=1),
                                               reduction='none').sum(1).mean()
    return logit_distill_loss


class PosGuidedFeaFusion(nn.Module):
    def __init__(self, cam_channel, lidar_channel):
        super(PosGuidedFeaFusion, self).__init__()
        self.fuse_posconv = nn.Sequential(
            nn.Conv2d(cam_channel + 2, cam_channel,
                      kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cam_channel)
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(cam_channel+lidar_channel, cam_channel,
                      kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cam_channel)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cam_channel, cam_channel,
                      kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(cam_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(cam_channel, cam_channel,
                      kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(cam_channel),
            nn.Sigmoid()
        )

    def forward(self, fea_cam, fea_lidar):
        # add coord for camera
        x_range = torch.linspace(-1, 1, fea_cam.shape[-1], device=fea_cam.device)
        y_range = torch.linspace(-1, 1, fea_cam.shape[-2], device=fea_cam.device)
        y, x = torch.meshgrid(y_range, x_range)

        y = y.expand([fea_cam.shape[0], 1, -1, -1])
        x = x.expand([fea_cam.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        cat_feature = torch.cat((fea_cam, fea_lidar), dim=1)
        fuse_out = self.fuse_conv(cat_feature)

        fuse_out = self.fuse_posconv(torch.cat((fuse_out, coord_feat), dim=1))
        attention_map = self.attention(fuse_out)
        out = fuse_out*attention_map + fea_cam

        return out


class LiDAR2Map(nn.Module):
    def __init__(self, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36):
        super(LiDAR2Map, self).__init__()
        self.camera2bev = LiftSplat(data_conf, instance_seg, embedded_dim, direction_pred, direction_dim)

        self.lidar2bev = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])  # 128
        # self.lidar2bev = BEVFusion_lidar(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])

        self.PGF2M = PosGuidedFeaFusion(128, 128)

        self.lidar_bevfpd = BEV_FPD(inC=128, outC=data_conf['num_channels'], instance_seg=instance_seg,
                                    embedded_dim=embedded_dim, direction_pred=direction_pred,
                                    direction_dim=direction_dim + 1)
        self.fusion_bevfpd = BEV_FPD(inC=128, outC=data_conf['num_channels'], instance_seg=False,
                                     embedded_dim=embedded_dim, direction_pred=False,
                                     direction_dim=direction_dim + 1)

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, flag='training'):
        if flag == 'training':
            camera_feature = self.camera2bev(img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, obtain_bev_feat=True)
            lidar_feature, voxel_feature = self.lidar2bev(lidar_data, lidar_mask)
            fusion_feature = self.PGF2M(camera_feature, lidar_feature)

            semantic, embedding, direction, student_feature = self.lidar_bevfpd(lidar_feature)
            fusion_semantic, fusion_embedding, fusion_direction, teacher_feature = self.fusion_bevfpd(fusion_feature)

            loss_feature_distill = compute_feature_distill_loss(student_feature, teacher_feature, voxel_feature)

            loss_logit_distill = compute_logit_distill_loss(semantic, fusion_semantic)

            return semantic, embedding, direction, loss_feature_distill, loss_logit_distill, fusion_semantic, fusion_embedding, fusion_direction
        else:
            lidar_feature, _ = self.lidar2bev(lidar_data, lidar_mask)
            semantic, embedding, direction, _ = self.lidar_bevfpd(lidar_feature)

            return semantic, embedding, direction

