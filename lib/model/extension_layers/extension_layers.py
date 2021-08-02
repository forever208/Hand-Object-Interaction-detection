
import torch
from torch import nn
import torch.nn.functional as F
import pickle
import datetime
from model.utils.config import cfg


class extension_layer(nn.Module):
    def __init__(self):
        super(extension_layer, self).__init__()
        self.init_layers_weights()    # define layers, loss and weights initialisation
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    def forward(self, input, input_padded, roi_labels, box_info):
        """
        compute both predictions and loss for 3 branches (contact_state, link, hand_side)
        :param input: pooled_feat, 2D tensor (128*batch_size, 2048)
        :param input_padded: padded_pooled_feat, 2D tensor (128*batch_size, 2048)
        :param roi_labels: object class labels, 2D tensor (batch, 128)
        :param box_info: contact gt labels, 3D tensor (batch, num_boxes, 5), each row is [contactstate, handside, magnitude, unitdx, unitdy]
        :return:
            loss_list: [(contact predictions, loss), (link predictions, loss), (handside predictions, loss)]
        """

        if self.training:
            batch_size = roi_labels.size(0)
            num_proposals = cfg.TRAIN.BATCH_SIZE
            input = input.view(batch_size, num_proposals, -1)    # ==> (batch, 128, 2048)
            input_padded = input_padded.view(batch_size, num_proposals, -1)    # ==> (batch, 128, 2048)
        else:
            input = input.unsqueeze(0)
            input_padded = input_padded.unsqueeze(0)

        # predictions and loss
        loss_list = [self.hand_contactstate_part(input_padded, roi_labels, box_info), \
                     self.hand_dxdymagnitude_part(input_padded, roi_labels, box_info), \
                     self.hand_handside_part(input, roi_labels, box_info)]

        return loss_list


    def init_layers_weights(self):
        """
        define the layers and loss function, do weights initialisation
        """

        # contact_state branch (5 outputs, portable, no contact, self-contact, stationary, other-person-contact)
        self.hand_contact_state_layer = nn.Sequential(nn.Linear(2048, 32), \
                                                      nn.ReLU(), \
                                                      nn.Dropout(p=0.2), \
                                                      nn.Linear(32, 5))

        # link branch (3 outputs, [magnitude, dx, dy])
        self.hand_dydx_layer = torch.nn.Linear(2048, 3)

        # hand side branch (1 output, [left/right] ,But the author said the output is R2...)
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
        :param roi_labels: object class labels (bg, target, hand), 2D tensor (batch, 128)
        :param box_info: contact gt labels, 3D tensor (batch, num_boxes, 5), each row is [contactstate, handside, magnitude, unitdx, unitdy]
        :return:
            contactstate_pred: contactstate class predictions, 3D tensor (batch, 128, 5), cls cls_scores are before softmax
            contactstate_loss: 1D tensor
        """
        contactstate_pred = self.hand_contact_state_layer(input)
        contactstate_loss = torch.zeros(1, dtype=torch.float).to(self.device)

        if self.training:
            for i in range(input.size(0)):    # for each batch
                gt_labels = box_info[i, :, 0]    # contact_state label
                index = roi_labels[i] == 2    # mask matrix, True indicates a hand
                if index.sum() > 0:    # if there is a hand, sum up the loss
                    contactstate_loss_sub = 0.1 * self.hand_contactstate_loss(contactstate_pred[i][index],
                                                                              gt_labels[index].long())
                    contactstate_loss += contactstate_loss_sub

            contactstate_loss = contactstate_loss / input.size(0)

        return contactstate_pred, contactstate_loss


    def hand_dxdymagnitude_part(self, input, roi_labels, box_info):
        """
        compute the prediction and loss for link (contact state centered)
        :param input: padded_pooled_feat, 3D tensor (batch, num_boxes, 2048)
        :param roi_labels: object class labels, 2D tensor (batch, num_boxes)
        :param box_info: contact gt labels, 3D tensor (batch, num_boxes, 5), each row is [contactstate, handside, magnitude, unitdx, unitdy]
        :return:
            dxdymagnitude_pred_norm: link predictions, 3D tensor (batch, 128, 3), each row is [magnitude, dx, dy]
            dxdymagnitude_loss: 1D tensor
        """
        dxdymagnitude_pred = self.hand_dydx_layer(input)    # (batch, 128, 3), each row is [magnitude, dx, dy]
        dxdymagnitude_pred_sub = 0.1 * F.normalize(dxdymagnitude_pred[:, :, 1:], p=2, dim=2)    # dx = dx / sqrt(dx.^2+dy.^2)
        dxdymagnitude_pred_norm = torch.cat([dxdymagnitude_pred[:, :, 0].unsqueeze(-1), dxdymagnitude_pred_sub], dim=2)    # (batch, 128, 3)
        dxdymagnitude_loss = torch.zeros(1, dtype=torch.float).to(self.device)

        # compute the MSEloss between predictions and gt labels
        if self.training:
            for i in range(input.size(0)):
                gt_labels = box_info[i, :, 2:5]    # [magnitude, dx, dy] label
                index = box_info[i, :, 0] > 0    # mask matrix, True indicates there is a gt contactstate
                if index.sum() > 0:    # index.sum() = 5 if there are 5 gt contactstate
                    dxdymagnitude_loss_sub = 0.1 * self.hand_dxdymagnitude_loss(dxdymagnitude_pred_norm[i][index],
                                                                                gt_labels[index])
                    dxdymagnitude_loss += dxdymagnitude_loss_sub

            dxdymagnitude_loss = dxdymagnitude_loss / input.size(0)

        return dxdymagnitude_pred_norm, dxdymagnitude_loss


    def hand_handside_part(self, input, roi_labels, box_info):
        """
        compute the prediction and loss for handside
        :param input: pooled_feat, 3D tensor (batch, 128, 2048)
        :param roi_labels: object class labels (bg, target, hand), 2D tensor (batch, 128)
        :param box_info: contact gt labels, 3D tensor (batch, num_boxes, 5), each row is [contactstate, handside, magnitude, unitdx, unitdy]
        :return:
            handside_pred: handside predictions, 2D tensor (batch, 128), cls cls_scores are before softmax
            handside_loss: 1D tensor
        """
        handside_pred = self.hand_lr_layer(input)    # (batch, 128)
        handside_loss = torch.zeros(1, dtype=torch.float).to(self.device)

        # compute the BCEWithLogitsLoss between predictions and gt labels
        if self.training:
            for i in range(input.size(0)):
                gt_labels = box_info[i, :, 1]    # get handside label
                index = roi_labels[i] == 2    # mask matrix, True indicates there is a gt hand
                if index.sum() > 0:
                    handside_loss_sub = 0.1 * self.hand_handside_loss(handside_pred[i][index],
                                                                      gt_labels[index].unsqueeze(-1))
                    handside_loss += handside_loss_sub

            handside_loss = handside_loss / input.size(0)

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
