import argparse
import tqdm
import os
import torch
from PIL import Image
from ..data.dataset import semantic_dataset
from ..data.const import NUM_CLASSES
from ..evaluation.iou import get_batch_iou
from ..model import get_model
import numpy as np
import imageio
import cv2
from shutil import copyfile
import time

def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)

def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def make_contour(img, colour=[0, 0, 0], double_line=False):
    h, w = img.shape[:2]
    out = img.copy()
    # Vertical lines
    out[np.arange(h), np.repeat(0, h)] = colour
    out[np.arange(h), np.repeat(w - 1, h)] = colour

    # Horizontal lines
    out[np.repeat(0, w), np.arange(w)] = colour
    out[np.repeat(h - 1, w), np.arange(w)] = colour

    if double_line:
        out[np.arange(h), np.repeat(1, h)] = colour
        out[np.arange(h), np.repeat(w - 2, h)] = colour

        # Horizontal lines
        out[np.repeat(1, w), np.arange(w)] = colour
        out[np.repeat(h - 2, w), np.arange(w)] = colour
    return out


def convert_color_map(map_lables, bg_color=(169, 169, 169)):
    hd_color_map = [[255, 255, 255], [68, 114, 196], [255, 103, 103], [0, 126, 0]]  # lane pc bound, rgb
    tmp = map_lables[0]
    tmp = tmp.resize(1, 80000)
    colors = np.array([hd_color_map[x] for x in tmp[0]])
    colors = colors.astype(np.uint8)
    color_map = colors.reshape(200, 400, 3)

    # map_lables = torch.nn.functional.one_hot(map_lables[0], 4)[..., 1:]
    # color_map = map_lables * 255
    # color_map = color_map.numpy().astype(np.uint8)
    #
    # # make backgorund
    # bg_mask = color_map.sum(axis=2) == 0
    # color_map[bg_mask] = np.array(bg_color)
    #
    # # make contour
    color_map = make_contour(color_map)

    return color_map


def flip_rotate_image(image):
    pil_img = Image.fromarray(image)
    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    pil_img = pil_img.transpose(Image.ROTATE_90)

    return np.array(pil_img)


def eval_iou(model, val_loader, logdir=None, epoch=0):
    model.eval()
    total_intersects = 0
    total_union = 0

    process_intersects = 0
    process_union = 0
    directory = os.path.join(logdir, "vis")
    if not os.path.isdir(directory):
        os.makedirs(directory)
    nums = 0
    min_area_threshold = 220
    # fusion_total_intersects = 0
    # fusion_total_union = 0
    num_iter = 0
    num_warmup = 5
    pure_inf_time = 0
    fps = 0
    with torch.no_grad():
        for imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, rec in tqdm.tqdm(val_loader):
            # torch.cuda.synchronize()
            # start_time = time.perf_counter()
            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), flag='testing')
            # segmentation = semantic[0]
            # torch.cuda.synchronize()
            # elapsed = time.perf_counter() - start_time
            # if num_iter >= num_warmup:
            #     pure_inf_time += elapsed
            # num_iter = num_iter + 1

            segmentation = onehot_encoding(semantic)
            segmentation = segmentation.squeeze(0)
            oh_pred = segmentation.cpu().numpy()
            processed_mask = []
            processed_mask.append(oh_pred[0])
            for i in range(1, oh_pred.shape[0]):
                single_mask = oh_pred[i].astype('uint8')
                # convert binary_seg_result
                binary_seg_result = np.array(single_mask * 255, dtype=np.uint8)

                # apply image morphology operation to fill in the hold and reduce the small area
                morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

                connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

                labels = connect_components_analysis_ret[1]
                stats = connect_components_analysis_ret[2]
                for index, stat in enumerate(stats):
                    if stat[4] <= min_area_threshold:
                        idx = np.where(labels == index)
                        morphological_ret[idx] = 0
                processed_mask.append(morphological_ret/255)
            processed_mask = np.stack(processed_mask)
            processed_mask = torch.from_numpy(processed_mask)
            processed_mask = processed_mask.unsqueeze(0)
            processed_mask = processed_mask.cuda().float()
            semantic_gt = semantic_gt.cuda().float()

            intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
            total_intersects += intersects
            total_union += union

            iou = intersects / (union + 1e-7)
            miou = iou[1:].mean()

            intersects, union = get_batch_iou(processed_mask, semantic_gt)
            process_intersects += intersects
            process_union += union

            # sample for vis
            # if miou > -1:
                # lidar_top_path = val_loader.dataset.nusc.get_sample_data_path(rec['data']['LIDAR_TOP'][0])
                # cam_front_path = val_loader.dataset.nusc.get_sample_data_path(rec['data']['CAM_FRONT'][0])
                # cam_front_left_path = val_loader.dataset.nusc.get_sample_data_path(rec['data']['CAM_FRONT_LEFT'][0])
                # cam_front_right_path = val_loader.dataset.nusc.get_sample_data_path(rec['data']['CAM_FRONT_RIGHT'][0])
                # cam_back_path = val_loader.dataset.nusc.get_sample_data_path(rec['data']['CAM_BACK'][0])
                # cam_back_left_path = val_loader.dataset.nusc.get_sample_data_path(rec['data']['CAM_BACK_LEFT'][0])
                # cam_back_right_path = val_loader.dataset.nusc.get_sample_data_path(rec['data']['CAM_BACK_RIGHT'][0])
                # base_path = lidar_top_path.split('/')[-1].replace('__LIDAR_TOP__', '_').split('.')[0]

                # nums = nums + 1
                # # base_path = os.path.join(directory, base_path)
                # base_path = directory
                # if not os.path.exists(base_path):
                #     os.mkdir(base_path)

                # semantic_gt_path = os.path.join(base_path, "SEMANTIC_GT_Sample_"+str(i)+".png")
                # sample_semantic_gt = semantic_gt[0, :, :, :]
                # semantic_mask_gt = sample_semantic_gt.max(0)[1]
                # semantic_mask_gt = semantic_mask_gt.unsqueeze(0)
                # color_map_labels_gt = convert_color_map(semantic_mask_gt, bg_color=(255, 255, 255))
                # imageio.imwrite(semantic_gt_path, flip_rotate_image(color_map_labels_gt))

                # semantic_pred_path = os.path.join(base_path, "SEMANTIC_PRED_180_Sample_" + str(nums) + ".png")
                # sample_semantic_pred = processed_mask[0, :, :, :]
                # semantic_mask_pred = sample_semantic_pred.max(0)[1]
                # semantic_mask_pred = semantic_mask_pred.unsqueeze(0)
                # color_map_labels_pred = convert_color_map(semantic_mask_pred, bg_color=(255, 255, 255))
                # imageio.imwrite(semantic_pred_path, flip_rotate_image(color_map_labels_pred))

                # np.save(base_path + '/pc.npy', lidar_data.numpy())
                # copyfile(cam_front_path, base_path + '/cam_front.jpg')
                # copyfile(cam_front_left_path, base_path + '/cam_front_left.jpg')
                # copyfile(cam_front_right_path, base_path + '/cam_front_right.jpg')
                # copyfile(cam_back_path, base_path + '/cam_back.jpg')
                # copyfile(cam_back_left_path, base_path + '/cam_back_left.jpg')
                # copyfile(cam_back_right_path, base_path + '/cam_back_right.jpg')


            # fusion_intersects, fusion_union = get_batch_iou(onehot_encoding(fusion_semantic), semantic_gt)
            # fusion_total_intersects += fusion_intersects
            # fusion_total_union += fusion_union , fusion_total_intersects / (fusion_total_union + 1e-7)
    # fps = (num_iter - num_warmup) / pure_inf_time
    return total_intersects / (total_union + 1e-7), process_intersects / (process_union + 1e-7), fps


def main(args):
    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 6,
        'final_dim': (128, 352),
    }
    if 'temporal' in args.model:
        parser_name = 'temporalsegmentationdata'
    else:
        parser_name = 'segmentationdata'
    [train_loader, val_loader], [train_sampler, val_sampler] = semantic_dataset(args.version, args.dataroot, data_conf,
                                                                                args.bsz, args.nworkers,
                                                                                args.distributed, parser_name)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model.load_state_dict(torch.load(args.modelf, map_location='cuda:0'), strict=False)
    model.cuda()
    iou, process_iou, fps = eval_iou(model, val_loader, logdir=args.logdir)
    miou = iou[1:].mean()
    process_miou = process_iou[1:].mean()
    print(iou)
    print(miou)
    print(process_iou)
    print(process_miou)
    print(fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./ablation')
    parser.add_argument('--distributed', action='store_true')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='/home/ws/data/nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='lift_splat')

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default='./ablation/model1.pt')

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

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
    main(args)
