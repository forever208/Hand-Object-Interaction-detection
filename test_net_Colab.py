from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_filtered_objects_PIL
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/resnet101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models", type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
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
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')

    parser.add_argument('--model_name',
                        help='directory to load models', default='handobj_100K',
                        required=False, type=str)
    parser.add_argument('--save_name',
                        help='folder to save eval results',
                        required=True)
    parser.add_argument('--thresh_hand',
                        type=float, default=0.1,
                        required=False)
    parser.add_argument('--thresh_obj', default=0.1,
                        type=float,
                        required=False)

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    # load local cfg file
    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    # automatically choose GPU or not
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfg.USE_GPU_NMS = True if torch.cuda.is_available() else False

    # Load training data from local xml files
    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)
    print('\n{:d} roidb entries\n'.format(len(roidb)))

    # model weights path
    input_dir = args.load_dir + "/" + args.net + "_" + args.model_name + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print(f'\n ---------> model path: {load_name}\n')

    # initialize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    # load model weights
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')

    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1).to(device)
    im_info = torch.FloatTensor(1).to(device)
    num_boxes = torch.LongTensor(1).to(device)
    gt_boxes = torch.FloatTensor(1).to(device)
    box_info = torch.FloatTensor(1).to(device)

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])


    """
    start test
    """
    fasterRCNN.to(device)
    start = time.time()
    max_per_image = 100
    vis = args.vis

    print(f'\n---------> det score thres_hand = {args.thresh_hand}')
    print(f'---------> det score thres_obj = {args.thresh_obj}\n')

    save_name = args.save_name
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]    # 3 rows, num_images columns

    # get batch data
    output_dir = get_output_dir(imdb, save_name)    # output/res101/voc_2007_test/hand0bj_100K/
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))  # 2D array, (1, 5)

    for i in range(num_images):
        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])    # 4D tensor (1, 3, h, w)
            im_info.resize_(data[1].size()).copy_(data[1])    # 2D tensor [[h, w, scale_factor]]
            gt_boxes.resize_(data[2].size()).copy_(data[2])    # 2D tensor [[x1, y1, x2, y2, cls], [], ...]
            num_boxes.resize_(data[3].size()).copy_(data[3])    # 1D tensor [num_boxes]
            box_info.resize_(data[4].size()).copy_(data[4])    # link gt label, 2D tensor [[contactstate, handside, magnitude, unitdx, unitdy], [], ...]]

        # get predictions from Faster R-CNN
        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

        class_scores = cls_prob.data    # class predictions, 3D tensor (batch, 128, num_classes), each row is processed by softmax: [0.1, 0.1, 0.8]
        boxes = rois.data[:, :, 1:5]    # roi coordinates, 3D tensor (batch, 128, 4), each row: [x1, y1, x2, y2]
        hand_contacts = loss_list[0][0]    # contactstate class predictions, 3D tensor (batch, 128, 5), cls cls_scores are before softmax
        hand_vector = loss_list[1][0].detach()    # link predictions, 3D tensor (batch, 128, 3), each row is [magnitude, dx, dy]
        lr_vector = loss_list[2][0].detach()    # handside predictions, 2D tensor (batch, 128), cls cls_scores are before softmax

        # contact predictions
        maxs, indices = torch.max(hand_contacts, 2)    # max: 2D tensor (batch, 128), index: 2D tensor (batch, 128)
        indices = indices.squeeze(0).unsqueeze(-1).float()    # (1, 128) ==> (128, 1)

        """might be problematic, each row of nc_prob is [1], I guess the author meant sigmoid """
        nc_prob = F.softmax(hand_contacts[:, :, 0].squeeze(0).unsqueeze(-1).float().detach(), dim=1)    # no contact, 2D tensor (128, 1)

        # hand side predictions
        lr = torch.sigmoid(lr_vector) > 0.5    # mask matrix
        lr = lr.squeeze(0).float()    # 1D tensor, (128)

        # Apply bbox regression based on final output
        if cfg.TEST.BBOX_REG:
            box_deltas = bbox_pred.data    # bbox delta prediction, 3D tensor (batch, 128, 4*num_total_classes)

            # normalize bbox coordinates by a pre-computed mean and stdev
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(device) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device)
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(device) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device)
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))    # 3D tensor (1, 128, 4*num_total_classes)

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)    # final bbox prediction, 3D tensor (1, 128, 4*num_total_classes)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

        # Simply repeat the boxes, once for each class
        else:
            pred_boxes = np.tile(boxes, (1, class_scores.shape[1]))

        pred_boxes /= data[1][0][2].item()    # back to original image scale

        class_scores = class_scores.squeeze()    # (1, 128, num_classes) ==> (128, num_classes)
        pred_boxes = pred_boxes.squeeze()    # (1, 128, 4*num_total_classes) ==> (128, 4*num_classes)
        det_toc = time.time()
        detect_time = det_toc - det_tic

        misc_tic = time.time()
        if args.vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)

        # nms for each class
        for j in range(1, imdb.num_classes):
            if pascal_classes[j] == 'hand':
                inds = torch.nonzero(class_scores[:, j] > args.thresh_hand).view(-1)    # 1D tensor (2*num)
            elif pascal_classes[j] == 'targetobject':
                inds = torch.nonzero(class_scores[:, j] > args.thresh_obj).view(-1)
            else:
                inds = torch.nonzero(class_scores[:, j] > args.thresh_obj).view(-1)

            # if there is det
            if inds.numel() > 0:
                cls_scores = class_scores[:, j][inds]    # only retain class_score whose probability > threshold, 1D tensor (num)
                _, order = torch.sort(cls_scores, 0, True)    # sort from high to low, order: 1D tensor
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j*4 : (j+1)*4]    # only retain bbox whose cls probability > threshold, 2D tensor (num, 4)

                # 2D tensor (num, 11)
                cls_dets = torch.cat((cls_boxes,    # bbox, 2D tensor (num, 4)
                                      cls_scores.unsqueeze(1),    # class, 2D tensor (num, 1)
                                      indices[inds, :],    # contact state, 2D tensor (num, 1)
                                      hand_vector.squeeze(0)[inds, :],    # link, 2D tensor (num, 3)
                                      lr[inds, :],    # hand side, 2D tensor (num, 1)
                                      nc_prob[inds, :]),    # no contact, 2D tensor (num, 1)
                                     1)
                cls_dets = cls_dets[order]

                # apply NMS
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]    # 2D tensor (keep_num, 11)

                if args.vis:
                    im2show = vis_detections_filtered_objects_PIL(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.1)

                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, 4] for j in range(1, imdb.num_classes)])    # class score of each image, 1D array
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, 4] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        if i % 100 == 0:
            print('im_detect: {:d}/{:d}  detection_time: {:.3f}s  NMS_time: {:.3f}s' .format(i + 1, num_images, detect_time, nms_time))

    # save detection results into file: detections.pkl
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # entrance of AP calculation  --> goes to pascal_voc.py
    print('\nEvaluating detections......................................\n')
    imdb.evaluate_detections(all_boxes, output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))
