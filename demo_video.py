# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------

import os
import sys
import numpy as np
import argparse
import pprint
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, \
    vis_detections_filtered_objects_PIL, vis_detections_filtered_objects  # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--videodir', dest='videodir',
                        help='video path',
                        default="/home/walter/Videos/boardgame.mp4", type=str)
    parser.add_argument('--output', dest='output',
                        help='whether save the detected video',
                        default=False, type=bool)
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save results',
                        default="images_det")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=8, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=89999, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=True)
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--thresh_hand',
                        type=float, default=0.5,
                        required=False)
    parser.add_argument('--thresh_obj', default=0.5,
                        type=float,
                        required=False)

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    # 第一维为batch，H W C
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == '__main__':
    args = parse_args()
    # print('Called with args:')
    # print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print(args.cuda)
    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)

    # load model
    model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
    if not os.path.exists(model_dir):
        raise Exception('There is no input directory for loading network from ' + model_dir)
    load_name = os.path.join(model_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']

    # initialize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        print("using cuda")
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    with torch.no_grad():
        if args.cuda > 0:
            cfg.CUDA = True

        if args.cuda > 0:
            fasterRCNN.cuda()

        fasterRCNN.eval()

        start = time.time()
        max_per_image = 100
        thresh_hand = args.thresh_hand
        thresh_obj = args.thresh_obj
        vis = args.vis

        webcam_num = args.webcam_num
        # print(f'thresh_hand = {thresh_hand}')
        # print(f'thnres_obj = {thresh_obj}')

        """
        camera配置，或者给定图像路径
        """
        # 逐帧读取给变量im传给net做检测  im = cv2格式
        vc = cv2.VideoCapture(args.videodir)
        if not vc.isOpened():
            raise ValueError("文件不存在")
        #video_FourCC = int(vc.get(cv2.CAP_PROP_FOURCC)) 会有编码格式错误
        video_fps = vc.get(cv2.CAP_PROP_FPS)
        video_size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
        if args.output:
            print(type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter("/home/walter/Videos/detected_video/test.mp4", video_FourCC, video_fps/2, video_size)


        success =True
        c = 0
        while(success):
            total_tic = time.time()
            success, frame = vc.read()
            if not success:
                break
            #print(c)
            c+=1
            #每两帧检测一下
            if c % 2 == 0:
                im = frame
                num_images = 1

                #缩放后图像， 缩放因子
                blobs, im_scales = _get_image_blob(im)
                assert len(im_scales) == 1, "Only single-image batch implemented"
                im_blob = blobs
                im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

                im_data_pt = torch.from_numpy(im_blob)
                #调整为batch, C, H, W
                im_data_pt = im_data_pt.permute(0, 3, 1, 2)
                im_info_pt = torch.from_numpy(im_info_np)

                with torch.no_grad():
                    im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                    im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                    gt_boxes.resize_(1, 1, 5).zero_()
                    num_boxes.resize_(1).zero_()
                    box_info.resize_(1, 1, 5).zero_()

                    # pdb.set_trace()
                det_tic = time.time()

                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

                #shape(1,300,3)
                scores = cls_prob.data
                boxes = rois.data[:, :, 1:5] #(1,300,4)
                #rois.data[:,:,0]代表什么?

                # extact predicted params
                #(1,300,5)
                contact_vector = loss_list[0][0]  # hand contact state info
                #(1,300,3)
                offset_vector = loss_list[1][0].detach()  # offset vector (factored into a unit vector and a magnitude)
                #(1,300,1)
                lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

                # get hand contact
                _, contact_indices = torch.max(contact_vector, 2)
                contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

                # get hand side
                lr = torch.sigmoid(lr_vector) > 0.5
                lr = lr.squeeze(0).float()

                if cfg.TEST.BBOX_REG:#???
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        # Optionally normalize targets by a precomputed mean and stdev
                        if args.class_agnostic:#???
                            if args.cuda > 0:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            else:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                            box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            if args.cuda > 0:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            else:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                            box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                pred_boxes /= im_scales[0]

                scores = scores.squeeze()
                pred_boxes = pred_boxes.squeeze()
                det_toc = time.time()
                detect_time = det_toc - det_tic
                misc_tic = time.time()
                if vis:
                    im2show = np.copy(im)
                obj_dets, hand_dets = None, None
                for j in range(1, len(pascal_classes)):
                    # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                    if pascal_classes[j] == 'hand':
                        inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
                    elif pascal_classes[j] == 'targetobject':
                        inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)

                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if args.class_agnostic:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds],
                                            offset_vector.squeeze(0)[inds], lr[inds]), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        if pascal_classes[j] == 'targetobject':
                            obj_dets = cls_dets.cpu().numpy()
                        if pascal_classes[j] == 'hand':
                            hand_dets = cls_dets.cpu().numpy()

                if vis:
                    # visualization
                    cvimg = vis_detections_filtered_objects_PIL(frame, im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)

                misc_toc = time.time()
                nms_time = misc_toc - misc_tic

                print(nms_time+detect_time)

                #cv2.imwrite("detectedvideo/detected{}.png".format(c),cvimg)
                cv2.imshow('detection', cvimg)
                if args.output:
                    out.write(cvimg)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


                # if webcam_num == -1:
                #     sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                #                     .format(num_images + 1, len(imglist), detect_time, nms_time))
                #     sys.stdout.flush()
                #
                # if vis and webcam_num == -1:
                #
                #     folder_name = args.save_dir
                #     os.makedirs(folder_name, exist_ok=True)
                #     result_path = os.path.join(folder_name, imglist[num_images][:-4] + "_det.png")
                #     im2show.save(result_path)
                # else:
                #     im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
                #     cv2.imshow("frame", im2showRGB)
                #     total_toc = time.time()
                #     total_time = total_toc - total_tic
                #     frame_rate = 1 / total_time
                #     print('Frame rate:', frame_rate)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break

            # if webcam_num >= 0:
            #     cap.release()
            #     cv2.destroyAllWindows()
        vc.release()