import os
import torch
import random
import logging
import argparse
import numpy as np
from time import time
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from model.loss.loss import SimpleLoss, DiscriminativeLoss
from evaluation.iou import get_batch_iou
from evaluation.angle_diff import calc_angle_diff
from model import get_model
from tools.evaluate import onehot_encoding, eval_iou
from tools.utils import inplace_relu, write_log, get_root_logger
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:600'


def train(args):
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "results.log"),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = get_root_logger()

    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
        # 'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        #          'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        # 'Ncams': 6,
        'final_dim': (256, 704),
    }

    parser_name = 'segmentationdata'

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    [train_loader, val_loader], [train_sampler, val_sampler] = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers, args.distributed, parser_name)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model.apply(inplace_relu)
    if args.finetune:
        model.load_state_dict(torch.load(args.modelf, map_location='cpu'), strict=True)
    device = torch.device("cuda", args.local_rank)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    else:
        model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = StepLR(opt, 20, 0.1)
    writer = SummaryWriter(logdir=args.logdir)

    loss_fn = SimpleLoss(args.pos_weight).cuda()
    embedded_loss_fn = DiscriminativeLoss(args.embedding_dim, args.delta_v, args.delta_d).cuda()
    direction_loss_fn = torch.nn.BCELoss(reduction='none')

    model.train()
    counter = 0
    last_idx = len(train_loader) - 1
    for epoch in range(args.nepochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            # val_sampler.set_epoch(epoch)
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans,
                     yaw_pitch_roll, semantic_gt, instance_gt, direction_gt) in enumerate(train_loader):
            t0 = time()
            opt.zero_grad()

            semantic, embedding, direction, feature_distill_loss, logit_distill_loss, fusion_semantic, fusion_embedding, fusion_direction = \
                model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(), post_trans.cuda(), post_rots.cuda(),
                      lidar_data.cuda(), lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), flag='training')

            semantic_gt = semantic_gt.cuda().float()
            instance_gt = instance_gt.cuda()

            seg_loss = 0.0
            fusion_seg_loss = 0.0
            if isinstance(semantic, list):
                for i in range(len(semantic)):
                    pred = semantic[i]
                    seg_loss = seg_loss + loss_fn(pred, semantic_gt)
                    fusion_pred = fusion_semantic[i]
                    fusion_seg_loss = fusion_seg_loss + loss_fn(fusion_pred, semantic_gt)
            else:
                seg_loss = loss_fn(semantic, semantic_gt)
                fusion_seg_loss = loss_fn(fusion_semantic, semantic_gt)

            if args.instance_seg:
                var_loss, dist_loss, reg_loss = embedded_loss_fn(embedding, instance_gt)
            else:
                var_loss = 0
                dist_loss = 0
                reg_loss = 0

            if args.direction_pred:
                direction_gt = direction_gt.cuda()
                lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                direction_loss = direction_loss_fn(torch.softmax(direction, 1), direction_gt)
                direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                angle_diff = calc_angle_diff(direction, direction_gt, args.angle_class)
            else:
                direction_loss = 0
                angle_diff = 0

            final_loss = (seg_loss + fusion_seg_loss + feature_distill_loss + logit_distill_loss) * args.scale_seg + \
                         var_loss * args.scale_var + dist_loss * args.scale_dist + direction_loss * args.scale_direction

            final_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                if isinstance(semantic, list):
                    semantic = semantic[0]
                    fusion_semantic = fusion_semantic[0]
                intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
                iou = intersects / (union + 1e-7)

                fusion_intersects, fusion_union = get_batch_iou(onehot_encoding(fusion_semantic), semantic_gt)
                fusion_iou = fusion_intersects / (fusion_union + 1e-7)
                if fusion_iou[1:].mean() != iou[1:].mean():
                    logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}] "
                                f"Lr: {opt.state_dict()['param_groups'][0]['lr']:>7.4f}  "
                                f"Time: {t1-t0:>7.4f}  "
                                f"Loss: {final_loss.item():>7.4f}  "
                                f"LiDAR Loss: {seg_loss.item():>7.4f}  "
                                f"Fusion Loss: {fusion_seg_loss.item():>7.4f}  "
                                f"Feature Distill Loss: {feature_distill_loss.item():>7.4f}  "
                                f"Logit Distill Loss: {logit_distill_loss.item():>7.4f}  "
                                f"Fusion mIOU: {np.array2string(fusion_iou[1:].numpy().mean(), precision=3, floatmode='fixed')}  "
                                f"Fusion IOU: {np.array2string(fusion_iou[1:].numpy(), precision=3, floatmode='fixed')}  "
                                f"mIOU: {np.array2string(iou[1:].numpy().mean(), precision=3, floatmode='fixed')}  "
                                f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

                    write_log(writer, iou, 'train', counter)
                    writer.add_scalar('train/step_time', t1 - t0, counter)
                    writer.add_scalar('train/learning_rate', opt.state_dict()['param_groups'][0]['lr'], counter)
                    writer.add_scalar('train/seg_loss', seg_loss, counter)
                    writer.add_scalar('train/fusion_seg_loss', fusion_seg_loss, counter)
                    writer.add_scalar('train/feature_distill_loss', feature_distill_loss, counter)
                    writer.add_scalar('train/logit_distill_loss', logit_distill_loss, counter)
                    writer.add_scalar('train/var_loss', var_loss, counter)
                    writer.add_scalar('train/dist_loss', dist_loss, counter)
                    writer.add_scalar('train/reg_loss', reg_loss, counter)
                    writer.add_scalar('train/direction_loss', direction_loss, counter)
                    writer.add_scalar('train/final_loss', final_loss, counter)
                    writer.add_scalar('train/angle_diff', angle_diff, counter)

        iou = eval_iou(model, val_loader, args.logdir, epoch)
        logger.info(f"EVAL[{epoch:>2d}]:    "
                    # f"Fusion mIOU: {np.array2string(fusion_iou[1:].numpy().mean(), precision=3, floatmode='fixed')}    "
                    # f"Fusion IOU: {np.array2string(fusion_iou[1:].numpy(), precision=3, floatmode='fixed')}    "
                    f"mIOU: {np.array2string(iou[1:].numpy().mean(), precision=3, floatmode='fixed')}    "
                    f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

        write_log(writer, iou, 'eval', counter)
        # write_log(writer, fusion_iou, 'eval_fusion', counter)
        model_name = os.path.join(args.logdir, f"model{epoch}.pt")
        if args.distributed:
            torch.save(model.module.state_dict(), model_name)
        else:
            torch.save(model.state_dict(), model_name)
        logger.info(f"{model_name} saved")
        model.train()

        sched.step()


def seed_torch(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    print("We use the seed: {}".format(seed))


if __name__ == '__main__':
    seed_torch()
    parser = argparse.ArgumentParser(description='LiDAR2Map training.')
    # logging config
    parser.add_argument('--logdir', type=str, default='./logs/lidar2map')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='/hdd/ws/data/nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='lidar2map')

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--nworkers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)  # 1e-4
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[256, 704])   # 128 352
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[1.0, 60.0, 0.5])    # 2.0, 90.0, 1.0

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    args = parser.parse_args()
    train(args)
