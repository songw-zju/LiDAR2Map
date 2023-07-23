from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_neck,
)
from mmdet3d.ops import Voxelization


from .base import Base3DFusionModel


class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords)
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


class BEVSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_transform: Dict[str, Any],
        out_channels: int,
    ) -> None:
        super().__init__()
        self.transform = BEVGridTransform(**grid_transform)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(x, (list, tuple)):
            x = x[0]

        x = self.transform(x)
        # x = self.classifier(x)

        return x


class BEVFusion_lidar(Base3DFusionModel):
    def __init__(
        self, C, xbound, ybound, zbound
    ) -> None:
        super().__init__()
        f = open('model/bevfusion/lidar-centerpoint-bev128.yaml', 'r')
        cfg = yaml.safe_load(f)
        encoders = cfg['model']['encoders']
        decoder = cfg['model']['decoder']
        heads = cfg['model']['heads']

        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound

        self.encoders = nn.ModuleDict()
        self.encoders["lidar"] = nn.ModuleDict(
            {
                "voxelize": Voxelization(**encoders["lidar"]["voxelize"]),
                "backbone": build_backbone(encoders["lidar"]["backbone"]),
            }
        )
        self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )

        self.heads = BEVSegmentationHead(heads['map']['in_channels'], heads['map']['grid_transform'],
                                         heads['map']['out_channels'])

        self.conv_out = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, C, 1),
        )

    def extract_lidar_features(self, x, mask) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, mask)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, mask):
        feats, coords, sizes = [], [], []
        num_points = mask.sum(dim=1)
        for k, res in enumerate(points):
            # res = res[0:int(num_points[k]), :]
            f, c, n = self.encoders["lidar"]["voxelize"](res)
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        sizes = torch.cat(sizes, dim=0)

        if self.voxelize_reduce:
            feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
            feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self, lidar_data, lidar_mask
    ):
        voxel_feature_ori = self.extract_lidar_features(lidar_data, lidar_mask)
        voxel_feature = self.decoder["backbone"](voxel_feature_ori)
        voxel_feature = self.decoder["neck"](voxel_feature)
        voxel_feature = self.heads(voxel_feature)

        return self.conv_out(voxel_feature).transpose(3, 2), voxel_feature_ori.transpose(3, 2)



