from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss


class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din    # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES    # [8, 16, 32], scale factors of down sampling
        self.anchor_ratios = cfg.ANCHOR_RATIOS    # [0.5, 1, 2]
        self.feat_stride = cfg.FEAT_STRIDE[0]    # 16

        # 3*3 conv layer (feature maps --> 3*3 conv)
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # bg/fg classification layer (reduce channels to 18 by 1*1 conv, predict the bg/fg class)
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2    # 18 = 2(bg/fg) * 9(anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # bbox regression layer (reduce channels to 36 by 1*1 conv, predict the bbox delta)
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4    # 36 = 4(coords) * 9(anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # proposal finetune layer (generate roi bboxes, finetune them by predicted bbox delta)
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # proposal labels layer (produces anchor labels for classification and bbox regression)
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0


    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x


    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        """
        @param base_feat: 4D tensor, (batch, 1024, h/16, w/16)
        @param im_info: 2D tensor, [[height, width, scale_factor (1.3)]]
        @param gt_boxes: 3D tensor [[[conf, x, y, w, h]]]
        @param num_boxes: 1D tensor [num_boxes]
        @return:
        """
        batch_size = base_feat.size(0)

        # apply ReLU after 3*3 conv
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)

        # get rpn classification score (reshape --> softmax --> reshape is caused by caffe)
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)    # 4D tensor, (batch, 18, h/16, w/16)

        # get rpn offsets to the pre-defined anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)    # 4D tensor, (batch, 36, h/16, w/16)

        # get the 300 proposals for each test image, finetune the proposlas by bbox delta
        # rois has shape (batch, 300, 5) maximum 300 proposals, each coloumn is [batch_ind, x1, y1, x2, y2]
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key))

        # generating training labels and compute the rpn loss
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        if self.training:
            assert gt_boxes is not None

            # get gt labels for proposals
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box
