import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.extension_layers import extension_layers
from model.roi_layers import ROIAlign, ROIPool
# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta


class _fasterRCNN(nn.Module):
    """
    Define the RPN & ROI_Pooling in this father class (_fasterRCNN)
    Define the backbone and head in the child class (resnet)
    combine the above two by call _fasterRCNN.create_architecture()
    """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # RPN layer (also compute the cls loss and bbox loss for RPN layer)
        self.RCNN_rpn = _RPN(self.dout_base_model)

        # RCNN gt labels layer (produces gt labels for final cls and bbox regression)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # ROIPooling or ROIAlign layer
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

        # new layer
        self.extension_layer = extension_layers.extension_layer()


    def forward(self, im_data, im_info, gt_boxes, num_boxes, box_info):
        """
        get call when fasterRCNN(im_data, im_info, gt_boxes, num_boxes), after fasterRCNN.create_architecture
        @param im_data: 4D tensor, (batch, 3, h, w)
        @param im_info: 2D tensor, [[height, width, scale_factor (1.3)]]
        @param gt_boxes: 3D tensor (batch, num_boxes, 5), each row is [cls, x1, y1, x2, y2]
        @param num_boxes: 1D tensor [num_boxes]
        @param box_info: 3D tensor (batch, num_boxes, 5), each row is [contactstate, handside, magnitude, unitdx, unitdy]
        @return:
        """
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        box_info = box_info.data

        # 1. img --> backbone (resnet) --> feature maps
        base_feat = self.RCNN_base(im_data)    # RCNN_base() is defined in the child class (resnet)

        # 2. feature map --> RPN --> roi bboxes (also compute the loss of RPN proposals)
        # rois: 3D tensor (batch, 2000, 5), each row is a bbox [batch_ind, x1, y1, x2, y2]
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # Select best 128 proposals for training, generate the 128 gt labels for final RCNN output
        if self.training:    # self.training is a class attribute in nn.module
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes, box_info)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, box_info = roi_data
            rois_label_retain = Variable(rois_label.long())
            box_info = Variable(box_info)
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label_retain = None
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        # 128 proposal bboxes, 3D tensor (batch, 128, 5)
        rois = Variable(rois)
        # expand the size of each bbox by 0.3*2 times
        rois_padded = Variable(self.enlarge_bbox(im_info, rois, 0.3))

        # 3.        rois --> roi pooling --> pooled features 4D tensor (128, 1024, 7, 7)
        # 3  padded rois --> roi pooling --> pooled features 4D tensor (128, 1024, 7, 7)
        # loss batch dimension in this step
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            pooled_feat_padded = self.RCNN_roi_align(base_feat, rois_padded.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
            pooled_feat_padded = self.RCNN_roi_pool(base_feat, rois_padded.view(-1, 5))
        else:
            raise Exception("rpn pooling mode is not defined")

        # 4.        pooled features --> downsample to 2D tensor (128, 2048) --> get bbox predictions
        # 4. padded pooled features --> downsample to 2D tensor (128, 2048)
        pooled_feat = self._head_to_tail(pooled_feat)    # _head_to_tail() is defined in the child class (resnet)
        pooled_feat_padded = self._head_to_tail(pooled_feat_padded)

        # 5. 2D feature tensor (128, 2048) --> get bbox predictions
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)    # RCNN_bbox_pred() is defined in the child class (resnet)

        # select the corresponding columns according to roi labels
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # 5. 2D feature tensor (128, 2048) --> get class predictions
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)
        # object_feat = pooled_feat[rois_label==1,:]
        # result = self.lineartrial(object_feat)

        # 5. loss function
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        loss_list = []

        if self.training:
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)    # classification loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)    # bbox regression L1 loss
            # prediction and loss of auxiliary layer
            loss_list = self.extension_layer(pooled_feat, pooled_feat_padded, rois_label_retain, box_info)
        else:
            loss_list = self.extension_layer(pooled_feat, pooled_feat_padded, None, box_info)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, loss_list


    def enlarge_bbox(self, im_info, rois, ratio=0.5):
        """
        double the size of each bbox
        :param im_info: 2D tensor, (batch, 3), each row is [height, width, scale_factor (1.3)]
        :param rois: 3D tensor (batch, 128, 5), each row is a selected best proposal [batch_ind, x1, y1, x2, y2]
        :return:
        """
        rois_width, rois_height = (rois[:, :, 3] - rois[:, :, 1]), (rois[:, :, 4] - rois[:, :, 2])
        rois_padded = rois.clone()
        rois_padded[:, :, 1] = rois_padded[:, :, 1] - ratio * rois_width    # x1 - 0.5*width
        rois_padded[:, :, 2] = rois_padded[:, :, 2] - ratio * rois_height    # y1 - 0.5*height
        rois_padded[:, :, 1][rois_padded[:, :, 1] < 0] = 0    # reset to 0 if x1 exceed the boundary
        rois_padded[:, :, 2][rois_padded[:, :, 2] < 0] = 0    # reset to 0 if y1 exceed the boundary

        rois_padded[:, :, 3] = rois_padded[:, :, 3] + ratio * rois_width    # x2 + 0.5*width
        rois_padded[:, :, 4] = rois_padded[:, :, 4] + ratio * rois_height    # y2 + 0.5*height

        # # fix the bug when batch > 1
        # for i in range(rois_padded.size(0)):
        #     rois_padded[i, :, 3][rois_padded[i, :, 3] > im_info[i, 1]] = im_info[i, 1]    # reset x2 if it exceed the boundary
        #     rois_padded[i, :, 4][rois_padded[i, :, 4] > im_info[i, 0]] = im_info[i, 0]    # reset y2 if it exceed the boundary

        rois_padded[:, :, 3][rois_padded[:, :, 3] > im_info[:, 0]] = im_info[:, 0]
        rois_padded[:, :, 4][rois_padded[:, :, 4] > im_info[:, 1]] = im_info[:, 1]
        return rois_padded


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)


    def create_architecture(self):
        """
        Merge resnet backbone to fasterRCNN, initialise the entire fasterRCNN
        """
        self._init_modules()    # _init_modules() is defined in the child class (resnet)
        self._init_weights()
