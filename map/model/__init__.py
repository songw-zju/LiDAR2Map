from .lidar.lidar2map import LiDAR2Map
from .lidar.pointpillar import PointPillar
from .camera.lift_splat import LiftSplat


def get_model(method, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36):
    if method == 'lidar2map':
        model = LiDAR2Map(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class)
    elif method == 'lift_splat':
        model = LiftSplat(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class)
    elif method == 'HDMapNet_lidar':
        model = PointPillar(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class)
    else:
        raise NotImplementedError

    return model
