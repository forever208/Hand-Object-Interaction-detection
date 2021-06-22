# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import numpy.random as npr
# from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob


def get_minibatch(roidb, num_classes):
    """
    Given a roidb, read the image and subtract pixel mean and resize to 600 (for both training and test)
    :param roidb: annotation list [{}] for one image, the {} contains all labels
    :param num_classes: 3
    :return blobs, a dict contains infos of an image,
            {'data': 4D array (1, 3, h, w),
             'gt_boxes': 2D array [[x1, y1, x2, y2, cls], [], ...],
             'im_info':2D array [[h, w, scale_factor]],
             'img_id':xx,
             'box_info': 2D array [[contactstate, handside, magnitude, unitdx, unitdy], [], ...]}
    """

    num_images = len(roidb)    # 1
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)    # always be [0]
    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), 'num_images ({}) must divide BATCH_SIZE ({})'.format(num_images, cfg.TRAIN.BATCH_SIZE)

    # load the image from local path, subtract pixel mean and resize the image
    # im_blob: an image 4D array(1, 3, h, w),  im_scales: a float number
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    blobs = {'data': im_blob}

    # only support batch_size = 1 ???????????????????????????????????????
    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes (for pascal voc)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]

    # gt boxes: 2D array [[x1, y1, x2, y2, cls], [], ...]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    # handinfo: 2D array [[contactstate, handside, magnitude, unitdx, unitdy], [], ...]
    handinfo = np.empty((len(gt_inds), 5), dtype=np.float32)
    handinfo[:, 0] = roidb[0]['contactstate'][gt_inds]
    handinfo[:, 1] = roidb[0]['handside'][gt_inds]
    handinfo[:, 2] = roidb[0]['magnitude'][gt_inds]
    handinfo[:, 3] = roidb[0]['unitdx'][gt_inds]
    handinfo[:, 4] = roidb[0]['unitdy'][gt_inds]

    blobs['img_id'] = roidb[0]['img_id']
    blobs['box_info'] = handinfo

    return blobs


def _get_image_blob(roidb, scale_inds):
    """
    load the image from local path, subtract pixel mean and resize the image
    :param roidb: annotation list [{}] for one image, the {} contains all labels
    :param scale_inds: [0]
    :return blob: an image 4D array (1, 3, h, w)
            im_scales: a float number
    """
    num_images = len(roidb)    # 1
    processed_ims = []
    im_scales = []

    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        # subtract pixel mean and rescale the image by factor = 600/shortest side
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]    # 600
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
