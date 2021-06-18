from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------
# Reorganized and modified by Mang Ning
# --------------------------------------------------------


import torch
import torch.nn as nn
import numpy as np
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms

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

        # generate default 9 anchors, 2D (9, 4)tensor, for each point in feature map
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)    # 9


    def forward(self, input):
        """
        for each (H, W) location i
            generate 9 anchor boxes centered on cell i
            finetune the for the 9 anchors at cell i bbox by predicted bbox deltas
        H = feat_h = h/16
        W = feat_w = w/16

        @param input: a tuple (rpn_cls_prob,    rpn_bbox_pred,   im_info,  cfg_key) whose shape is
                              ((batch,18,H,W), (batch,36,H,W),  (batch,2), 'train/test')
        @return: rois (batch, 2000, 5), 2000 training proposals, each row is [batch_ind, x1, y1, x2, y2]
        """

        # take the positive (object) cls_scores
        scores = input[0][:, self._num_anchors:, :, :]    # (batch, 9, H, W)
        bbox_deltas = input[1]    # (batch, 36, H, W)
        im_info = input[2]    # (batch, 2)
        cfg_key = input[3]    # 'train/test'

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N    # 6000 for train
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N    # 300 for test
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH    # 0.7
        min_size      = cfg[cfg_key].RPN_MIN_SIZE    # 16
        batch_size = bbox_deltas.size(0)    # batch

        # compute the shift value for H*W cells
        feat_height, feat_width = scores.size(2), scores.size(3)    # H, W
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()    # (H*W, 4)

        # copy and shift the 9 anchors for H*W cells
        # copy the H*W*9 anchors for batch images
        A = self._num_anchors    # 9
        K = shifts.size(0)    # H * W
        self._anchors = self._anchors.type_as(scores)
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)    # (H*W, 9, 4) anchors for 1 image
        anchors = anchors.view(1, K*A, 4).expand(batch_size, K*A, 4)    # (batch, H*W*9, 4) anchors for batch images

        # make bbox_deltas the same order with the anchors:
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()    # (batch, 36, H, W) --> (batch, H, W, 36)
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)    # (batch, H, W, 36) --> (batch, H*W*9, 4)

        # Same story for the cls_scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()    # (batch, 9, H, W) --> (batch, H, W, 9)
        scores = scores.view(batch_size, -1)    # (batch, H, W, 9) --> (batch, H*W*9)

        # Finetune [x1, y1, x2, y2] of anchors according to the predicted bbox_delta
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)    # (batch, H*W*9, 4)

        # 2. clip predicted boxes to the image, make sure [x1, y1, x2, y2] are within the image [h, w]
        proposals = clip_boxes(proposals, im_info, batch_size)    # (batch, H*W*9, 4)
        scores_keep = scores
        proposals_keep = proposals

        # 3. remove predicted bboxes whose height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        _, order = torch.sort(scores_keep, 1, True)    # high score to low score

        # initialise the proposals by zero tensor
        output = scores.new(batch_size, post_nms_topN, 5).zero_()

        # for each image
        for i in range(batch_size):
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]
            order_single = order[i]

            # 5. take top pre_nms_topN proposals before NMS (e.g. 6000)
            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]
            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply NMS (e.g. threshold = 0.7)
            keep_idx_i = nms(proposals_single, scores_single.squeeze(1), nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            # 7. take after_nms_topN proposals after NMS (e.g. 300 for test, 2000 for train)
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]

            # 8. return the top proposals (-> RoIs top)
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # 9. padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        return output    # (batch, 2000, 5) 2000 training proposals, each row is [batch_ind, x1, y1, x2, y2]


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
