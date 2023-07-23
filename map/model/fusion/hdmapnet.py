import torch
from torch import nn
from .transfusion import TransFusion
from .homography import bilinear_sampler, IPM
from .utils import plane_grid_2d, get_rot_2d, cam_to_pixel
from .pointpillar import PointPillarEncoder
from .base import CamEncode, BevEncode
from data.utils import gen_dx_bx


class ViewTransformation(nn.Module):
    def __init__(self, fv_size, bv_size, n_views=6):
        super(ViewTransformation, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        self.bv_size = bv_size
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                nn.ReLU()
            )
            self.hw_mat.append(fc_transform)
        self.hw_mat = nn.ModuleList(self.hw_mat)

    def forward(self, feat):
        B, N, C, H, W = feat.shape
        feat = feat.view(B, N, C, H*W)
        outputs = []
        for i in range(N):
            output = self.hw_mat[i](feat[:, i]).view(B, C, self.bv_size[0], self.bv_size[1])
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return outputs


class HDMapNet(nn.Module):
    def __init__(self, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36, lidar=False):
        super(HDMapNet, self).__init__()
        self.camC = 64
        self.downsample = 16

        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        final_H, final_W = nx[1].item(), nx[0].item()

        self.camencode = CamEncode(self.camC)
        fv_size = (data_conf['image_size'][0]//self.downsample, data_conf['image_size'][1]//self.downsample)
        bv_size = (final_H//5, final_W//5)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)

        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4*res_x/final_W]
        ipm_ybound = [-res_x/2, res_x/2, 2*res_x/final_H]
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=self.camC, extrinsic=True)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_sampler = nn.Upsample(scale_factor=5, mode='bilinear', align_corners=True)

        self.dropout_cam = nn.Dropout2d(p=0.2)
        self.dropout_lidar = nn.Dropout2d(p=0.2)

        self.lidar = lidar
        if lidar:
            self.pp = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
            # self.transfusion = TransFusion(num_views=6)
            self.bevencode = BevEncode(inC=self.camC+128, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        else:
            self.bevencode = BevEncode(inC=self.camC, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)

    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = None

        return Ks, RTs, post_RTs

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        img_feat = self.get_cam_feats(img)
        x = self.view_fusion(img_feat)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        topdown = self.up_sampler(topdown)
        if self.lidar:
            lidar_feature = self.pp(lidar_data, lidar_mask)
            # lidar_feature = self.transfusion(lidar_feature, img_feat)
            topdown = torch.cat([topdown, lidar_feature], dim=1)
        return self.bevencode(topdown)


class ResidualBasedFusionBlock(nn.Module):
    def __init__(self, curr_channels, prev_channels):
        super(ResidualBasedFusionBlock, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(curr_channels+prev_channels, curr_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(curr_channels)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(curr_channels, curr_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(curr_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(curr_channels, curr_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(curr_channels),
            nn.Sigmoid()
        )

    def forward(self, curr_feature, prev_feature):
        cat_feature = torch.cat((curr_feature, prev_feature), dim=1)
        fuse_out = self.fuse_conv(cat_feature)
        attention_map = self.attention(fuse_out)
        out = fuse_out*attention_map + curr_feature
        return out


class TemporalHDMapNet(HDMapNet):
    def __init__(self, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36, lidar=False):
        super(TemporalHDMapNet, self).__init__(data_conf, instance_seg, embedded_dim, direction_pred, direction_dim, lidar)
        self.xbound = data_conf['xbound']
        self.ybound = data_conf['ybound']
        self.fusion_layer = ResidualBasedFusionBlock(64, 64)

    def get_cam_feats(self, x):
        """Return B x T x N x H/downsample x W/downsample x C
        """
        B, T, N, C, imH, imW = x.shape
        x = x.view(B*T*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, T, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def temporal_fusion(self, topdown, translation, yaw):
        B, T, C, H, W = topdown.shape

        if T == 1:
            return topdown[:, 0]

        # topdown = torch.split(topdown, 1, dim=1)
        # prev_down = topdown[1].squeeze(1)
        # topdown = topdown[0].squeeze(1)
        # topdown = self.fusion_layer(topdown, prev_down)

        grid = plane_grid_2d(self.xbound, self.ybound).view(1, 1, 2, H*W).repeat(B, T-1, 1, 1)
        rot0 = get_rot_2d(yaw[:, 1:])
        trans0 = translation[:, 1:, :2].view(B, T-1, 2, 1)
        rot1 = get_rot_2d(yaw[:, 0].view(B, 1).repeat(1, T-1))
        trans1 = translation[:, 0, :2].view(B, 1, 2, 1).repeat(1, T-1, 1, 1)
        grid = rot1.transpose(2, 3) @ grid
        grid = grid + trans1
        grid = grid - trans0
        grid = rot0 @ grid
        grid = grid.view(B*(T-1), 2, H, W).permute(0, 2, 3, 1).contiguous()
        grid = cam_to_pixel(grid, self.xbound, self.ybound)
        topdown = topdown.permute(0, 1, 3, 4, 2).contiguous()
        prev_topdown = topdown[:, 1:]
        warped_prev_topdown = bilinear_sampler(prev_topdown.reshape(B*(T-1), H, W, C), grid).view(B, T-1, H, W, C)
        #
        warped_prev_topdown = warped_prev_topdown.squeeze(1).permute(0, 3, 1, 2).contiguous()
        topdown = topdown[:, 0].permute(0, 3, 1, 2).contiguous()
        topdown = self.fusion_layer(topdown, warped_prev_topdown)


        # topdown = torch.cat([topdown[:, 0].unsqueeze(1), warped_prev_topdown], axis=1)
        # topdown = topdown.view(B, T, H, W, C)
        # topdown = topdown.max(1)[0]
        # topdown = topdown.permute(0, 3, 1, 2).contiguous()
        return topdown

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        img_feat = self.get_cam_feats(img)
        B, T, N, C, h, w = img_feat.shape
        img_feat = img_feat.view(B*T, N, C, h, w)
        x = self.view_fusion(img_feat)

        # x = x.view(B*T, N, C, h, w)
        intrins = intrins.view(B*T, N, 3, 3)
        rots = rots.view(B*T, N, 3, 3)
        trans = trans.view(B*T, N, 3)
        post_rots = post_rots.view(B*T, N, 3, 3)
        post_trans = post_trans.view(B*T, N, 3)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)

        topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        topdown = self.up_sampler(topdown)
        _, C, H, W = topdown.shape
        topdown = topdown.view(B, T, C, H, W)
        topdown = self.temporal_fusion(topdown, car_trans, yaw_pitch_roll[..., 0])
        if self.lidar:
            lidar_feature = self.pp(lidar_data, lidar_mask)
            topdown = self.dropout_cam(topdown)
            lidar_feature = self.dropout_lidar(lidar_feature)
            # lidar_feature = self.transfusion(lidar_feature, img_feat)
            topdown = torch.cat([topdown, lidar_feature], dim=1)
        return self.bevencode(topdown)
