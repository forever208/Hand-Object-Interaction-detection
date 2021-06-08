import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import pickle
import datetime
from model.utils.config import cfg


class extension_layer(nn.Module):
    def __init__(self):
        super(extension_layer, self).__init__()
        self.init_layers_weights()    # define the the layer and weights initialisation


    def forward(self, input, input_padded, roi_labels, box_info):
        """
        compute both predictions and loss for 3 branches (contact_state, link, hand_side)
        :param input: pooled_feat, 2D tensor (128*batch_size, 2048)
        :param input_padded: padded_pooled_feat, 2D tensor (128*batch_size, 2048)
        :param roi_labels: object class labels, 2D tensor (batch, 128)
        :param box_info: 3D tensor (batch, num_boxes, 5), each row is [contactstate, handside, magnitude, unitdx, unitdy]
        :return:
        """

        if self.training:
            batch_size = roi_labels.size(0)
            num_proposals = cfg.TRAIN.BATCH_SIZE
            input = input.view(batch_size, num_proposals, -1)
            input_padded = input_padded.view(batch_size, num_proposals, -1)
        else:
            input = input.unsqueeze(0)
            input_padded = input_padded.unsqueeze(0)

        # output the predictions and loss
        # loss_list: [(predictions, loss), (predictions, loss), (predictions, loss)]
        loss_list = [self.hand_contactstate_part(input_padded, roi_labels, box_info), \
                     self.hand_dxdymagnitude_part(input_padded, roi_labels, box_info), \
                     self.hand_handside_part(input, roi_labels, box_info)]

        return loss_list


    def init_layers_weights(self):
        """
        define the the layer and do weights initialisation
        """
        # contact_state branch (5 outputs, portable, no contact, self-contact, stationary, other-person-contact)
        self.hand_contact_state_layer = nn.Sequential(nn.Linear(2048, 32), \
                                                      nn.ReLU(), \
                                                      nn.Dropout(p=0.5), \
                                                      nn.Linear(32, 5))

        # link branch (3 outputs, dx, dy, magnitude)
        self.hand_dydx_layer = torch.nn.Linear(2048, 3)

        # hand side branch (1 output, left/right) But the author said the output is R2...
        self.hand_lr_layer = torch.nn.Linear(2048, 1)

        # loss function for each branch
        self.hand_contactstate_loss = nn.CrossEntropyLoss()
        self.hand_dxdymagnitude_loss = nn.MSELoss()
        self.hand_handside_loss = nn.BCEWithLogitsLoss()

        # initialise the weights
        self._init_weights()


    def hand_contactstate_part(self, input, roi_labels, box_info):
        """
        compute the prediction and loss for contact state
        :param input: padded_pooled_feat, 3D tensor (batch, 128, 2048)
        :param roi_labels: object class labels, 2D tensor (batch, 128)
        :param box_info: 3D tensor (batch, num_boxes, 5), each row is [contactstate, handside, magnitude, unitdx, unitdy]
        :return:
        """
        contactstate_pred = self.hand_contact_state_layer(input)
        contactstate_loss = 0

        if self.training:
            for i in range(input.size(0)):    # for each batch
                gt_labels = box_info[i, :, 0]    # contact_state label
                index = roi_labels[i] == 2    # get a True/False array
                if index.sum() > 0:    # if there is a hand, add up the loss
                    contactstate_loss_sub = 0.1 * self.hand_contactstate_loss(contactstate_pred[i][index],
                                                                              gt_labels[index].long())

                    if not contactstate_loss:
                        contactstate_loss = contactstate_loss_sub
                    else:
                        contactstate_loss += contactstate_loss_sub

        return contactstate_pred, contactstate_loss


    def hand_dxdymagnitude_part(self, input, roi_labels, box_info):
        """
        compute the prediction and loss for link
        :param input: padded_pooled_feat, 3D tensor (batch, num_boxes, 2048)
        :param roi_labels: object class labels, 2D tensor (batch, num_boxes)
        :param box_info: 3D tensor (batch, num_boxes, 5), each row is [contactstate, handside, magnitude, unitdx, unitdy]
        :return:
        """
        dxdymagnitude_pred = self.hand_dydx_layer(input)

        dxdymagnitude_pred_sub = 0.1 * F.normalize(dxdymagnitude_pred[:, :, 1:], p=2, dim=2)

        dxdymagnitude_pred_norm = torch.cat([dxdymagnitude_pred[:, :, 0].unsqueeze(-1), dxdymagnitude_pred_sub], dim=2)
        dxdymagnitude_loss = 0

        if self.training:
            # if 1:
            for i in range(input.size(0)):
                gt_labels = box_info[i, :, 2:5]  # [magnitude, dx, dy] label
                index = box_info[i, :, 0] > 0  # in-contact

                if index.sum() > 0:
                    dxdymagnitude_loss_sub = 0.1 * self.hand_dxdymagnitude_loss(dxdymagnitude_pred_norm[i][index],
                                                                                gt_labels[index])

                    if not dxdymagnitude_loss:
                        dxdymagnitude_loss = dxdymagnitude_loss_sub
                    else:
                        dxdymagnitude_loss += dxdymagnitude_loss_sub

        return dxdymagnitude_pred_norm, dxdymagnitude_loss


    def hand_handside_part(self, input, roi_labels, box_info):
        handside_pred = self.hand_lr_layer(input)
        handside_loss = 0

        if self.training:
            for i in range(input.size(0)):
                gt_labels = box_info[i, :, 1]  # hand side label
                index = roi_labels[i] == 2  # if class is hand
                if index.sum() > 0:
                    handside_loss_sub = 0.1 * self.hand_handside_loss(handside_pred[i][index],
                                                                      gt_labels[index].unsqueeze(-1))

                    if not handside_loss:
                        handside_loss = handside_loss_sub
                    else:
                        handside_loss += handside_loss_sub

        return handside_pred, handside_loss


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initializer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.hand_contact_state_layer[0], 0, 0.01)
        normal_init(self.hand_contact_state_layer[3], 0, 0.01)
        normal_init(self.hand_dydx_layer, 0, 0.01)
        normal_init(self.hand_lr_layer, 0, 0.01)
