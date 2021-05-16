from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of anchors
    """

    def __init__(self, feat_stride, scales, ratios):
        """
        @param feat_stride: 16
        @param scales: [8, 16, 32]
        @param ratios: [0.5, 1, 2]
        """
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), ratios=np.array(ratios))).float()    # (9, 4) tensor
        self._num_anchors = self._anchors.size(0)


    def forward(self, input):
        """
        for each (H, W) location i
            generate 9 anchor boxes centered on cell i
            finetune the for the 9 anchors at cell i bbox by predicted bbox deltas
        clip predicted boxes to image
        remove predicted boxes with either height or width < threshold
        sort all (proposal, score) pairs by score from highest to lowest
        take top pre_nms_topN proposals before NMS
        apply NMS with threshold 0.7 to remaining proposals
        take after_nms_topN proposals after NMS
        return the top proposals (-> RoIs top, scores top)

        @param input: a tuple (rpn_cls_prob,    rpn_bbox_pred,   im_info,  cfg_key)
                              ((batch,18,h,w), (batch,36,h,w), (batch, 2), 'train/test')
        @return: rois (batch, 300, 5), maximum 300 proposals, each row is [batch_ind, x1, y1, x2, y2]
        """

        scores = input[0][:, self._num_anchors:, :, :]    # take the positive (object) scores
        bbox_deltas = input[1]
        im_info = input[2]
        cfg_key = input[3]

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N    # 6000 for train
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N    # 300 for test
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH    # 0.7
        min_size      = cfg[cfg_key].RPN_MIN_SIZE    # 16
        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()    # (feat_h*feat_w, 4)

        A = self._num_anchors    # 9
        K = shifts.size(0)    # feat_height * feat_width

        self._anchors = self._anchors.type_as(scores)
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)    # (h*w, 9, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)    # (batch, h*w*9, 4)

        # make cls_score the same order with the anchors:
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposal bboxes
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)
        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        # initialise the proposals by creating a zero tensor
        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            keep_idx_i = nms(proposals_single, scores_single.squeeze(1), nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        return output    # (batch, 300, 5) maximum 300 proposals, each row is [batch_ind, x1, y1, x2, y2]


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass


    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep
