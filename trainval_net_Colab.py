# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------

import os
import numpy as np
import argparse
import pprint
import time

import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient
from model.faster_rcnn.resnet import resnet


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=10, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models',
                        default="models", type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=2, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large image scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=3, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # load the haft-trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)

    # log and display
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
    # save model and log
    parser.add_argument('--model_name',
                        help='directory to save trained models',
                        default='handobj_100K', type=str)
    parser.add_argument('--log_name',
                        help='directory to save training logs',
                        default='handobj_100K', type=str)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)
        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':
    # command arguments
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.dataset == 'pascal_voc':
        args.imdb_name = 'voc_2007_trainval'
        args.imdbval_name = 'voc_2007_test'
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    else:
        raise Exception("we currently only support pascal_voc dataset")

    # configuration
    args.cfg_file = 'cfgs/{}_ls.yml'.format(args.net) if args.large_scale else 'cfgs/{}.yml'.format(args.net)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # automatically choose GPU or not
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfg.USE_GPU_NMS = True if torch.cuda.is_available() else False

    # Load training data from local xml files
    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    print('{:d} number of training images'.format(len(roidb)))

    # build up dataloader pipeline
    sampler_batch = sampler(train_size, args.batch_size)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)

    # output path of the trained model
    output_dir = args.save_dir + "/" + args.net + "_" + args.model_name + "/" + args.dataset
    print(f'\n---------> model output_dir = {output_dir}\n')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initialize the tensor holder.
    im_data = torch.FloatTensor(1).to(device)
    im_info = torch.FloatTensor(1).to(device)
    num_boxes = torch.LongTensor(1).to(device)
    gt_boxes = torch.FloatTensor(1).to(device)
    box_info = torch.FloatTensor(1).to(device)    # ground truth link info between hand-object

    # initialize the network.
    if args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        raise Exception("network is not defined")
    fasterRCNN.create_architecture()

    # set optimizer
    lr = args.lr
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception("optimizer is not defined")

    # load the half-trained model
    if args.resume:
        load_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    if args.use_tfboard:
        args.log_name = args.model_name
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(f"logs/log_{args.log_name}")
        print(f'\n---------> log_dir = logs/log_{args.log_name}\n')

    """
    start predictions
    """
    iters_per_epoch = int(train_size / args.batch_size)
    fasterRCNN.to(device)
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # set as train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        # lr=0.1*lr for every 3 epochs
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        # load a batch of images
        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])    # 4D tensor (1, 3, h, w)
                im_info.resize_(data[1].size()).copy_(data[1])    # 2D tensor [[h, w, scale_factor]]
                gt_boxes.resize_(data[2].size()).copy_(data[2])    # 2D tensor [[x1, y1, x2, y2, cls], [], ...]
                num_boxes.resize_(data[3].size()).copy_(data[3])    # 1D tensor [num_boxes]
                box_info.resize_(data[4].size()).copy_(data[4])    # link gt label, 2D tensor [[contactstate, handside, magnitude, unitdx, unitdy], [], ...]]

            # get predictions
            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

            # compute loss
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            for score_loss in loss_list:
                if type(score_loss[1]) is not int:
                    loss += score_loss[1].mean()    # auxiliary loss terms from auxiliary layers
            loss_temp += loss.item()

            # back propagation and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # display interval training info
            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    loss_hand_state = 0 if type(loss_list[0][1]) is int else loss_list[0][1].mean().item()
                    loss_hand_dydx = 0 if type(loss_list[1][1]) is int else loss_list[1][1].mean().item()
                    loss_hand_lr = 0 if type(loss_list[2][1]) is int else loss_list[2][1].mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    loss_hand_state = 0 if type(loss_list[0][1]) is int else loss_list[0][1].item()
                    loss_hand_dydx = 0 if type(loss_list[1][1]) is int else loss_list[1][1].item()
                    loss_hand_lr = 0 if type(loss_list[2][1]) is int else loss_list[2][1].item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                print("\t\t\tcontact_state_loss: %.4f, dydx_loss: %.4f, lr_loss: %.4f" % (
                loss_hand_state, loss_hand_dydx, loss_hand_lr))

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box,
                        'loss_hand_state': loss_hand_state,
                        'loss_hand_dydx': loss_hand_dydx,
                        'loss_hand_lr': loss_hand_lr
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

        # save model locally after each epoch
        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        state = {
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }
        torch.save(state, save_name)
        print('save model: {}'.format(save_name))

        # save model to Google drive
        Google_drive_path = '/content/drive/MyDrive/HOI_detection/trained_model'
        if os.path.exists(Google_drive_path):
            save_name_gdrive = os.path.join(Google_drive_path, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
            torch.save(state, save_name_gdrive)
            print('save model to Google drive: {}'.format(save_name_gdrive))

    if args.use_tfboard:
        logger.close()
