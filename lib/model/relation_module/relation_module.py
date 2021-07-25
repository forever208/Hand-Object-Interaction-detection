import torch
import torch.nn as nn
import numpy as np



class RelationModule(nn.Module):

    def __init__(self, n_relations=16, appear_feature_dim=2048, key_feature_dim=128, geo_feature_dim=128):
        super(RelationModule, self).__init__()
        self.num_relations = n_relations
        self.dim_g = geo_feature_dim

        # define 32 relation modules
        self.relation = nn.ModuleList()
        for N in range(self.num_relations):
            self.relation.append(RelationUnit(appear_feature_dim, key_feature_dim, geo_feature_dim))


    def forward(self, app_feature, bbox_coordinates):
        """
        build up 32 relation modules
        :param app_feature: appearance feature of 128 proposals, 2D tensor (128*batch, 2048)
        :param bbox_coordinates: coordinate of 128 proposals, 2D tensor (batch, 128, 5)
        :return: 
        """
        position_embedding = self.PositionalEmbedding(bbox_coordinates)

        for N in range(self.num_relations):
            if (N == 0):
                concat = self.relation[N](app_feature, position_embedding)
            else:
                concat = torch.cat((concat, self.relation[N](app_feature, position_embedding)), -1)  # concat along last channel
        # return concat + app_feature
        return concat

    def PositionalEmbedding(self, bbox_coor, dim_g=128, wave_len=1000):
        bbox_coor = bbox_coor.squeeze(0)  # (batch, 128, 5) ==> (128, 5)
        bbox_coor = bbox_coor[:, 1:]  # (128, 5) == > (128, 4), remove the first column
        x_min, y_min, x_max, y_max = torch.chunk(bbox_coor, 4, dim=1)  # (128, 4) ==> (128, 1)

        cx = (x_min + x_max) * 0.5  # (128, 1)
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min)
        h = (y_max - y_min)
        w = torch.clamp(w, min=1e-4)
        h = torch.clamp(h, min=1e-4)

        delta_x = cx - cx.view(1, -1)  # (128, 128), each row is the delta_x between the box and the all boxes
        delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
        delta_x = torch.log(delta_x)

        delta_y = cy - cy.view(1, -1)  # (128, 128), each row is the delta_y between the box and the all boxes
        delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
        delta_y = torch.log(delta_y)

        delta_w = torch.log(w / w.view(1, -1))  # (128, 128), each row is the delta_w between the box and the all boxes
        delta_h = torch.log(h / h.view(1, -1))  # (128, 128), each row is the delta_h between the box and the all boxes
        size = delta_h.size()

        delta_x = delta_x.view(size[0], size[1], 1)  # (128, 128) ==> (128, 128, 1)
        delta_y = delta_y.view(size[0], size[1], 1)
        delta_w = delta_w.view(size[0], size[1], 1)
        delta_h = delta_h.view(size[0], size[1], 1)

        position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # (128, 128, 4)

        feat_range = torch.arange(dim_g / 8).cuda()  # [0,1,2,3,...,7]
        dim_mat = feat_range / (dim_g/8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))  # [1.0000, 0.4217, 0.1778, 0.0750, 0.0316, 0.0133, 0.0056, 0.0024]

        dim_mat = dim_mat.view(1, 1, 1, -1)  # (1, 1, 1, 8)
        position_mat = position_mat.view(size[0], size[1], 4, -1)  # (128, 128, 4) ==> (128, 128, 4, 1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat  # (128,128,4,1) * (1,1,1,8) ==> (128,128,4,8)
        mul_mat = mul_mat.view(size[0], size[1], -1)  # (128, 128, 32)
        sin_mat = torch.sin(mul_mat)  # (128, 128, 32)
        cos_mat = torch.cos(mul_mat)  # (128, 128, 32)
        embedding = torch.cat((sin_mat, cos_mat), -1)  # (128, 128, 64)

        return embedding



class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=2048, key_feature_dim=128, geo_feature_dim=128):
        super(RelationUnit, self).__init__()

        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        self.WG = nn.Linear(geo_feature_dim, 1, bias=True)  # (128, 128, 64) ==> (128, 128, 1)
        self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)  # (128, 2048) ==> (128, 64)
        self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, app_feature, position_embedding):
        """
        :param app_feature: appearance feature of 128 proposals, 2D tensor (128*batch, 2048)
        :param position_embedding: positional embedding of 128 proposals, 3D tensor (128, 128, 64)
        :return:
        """
        N, _ = app_feature.size()

        # similarity measurement
        w_q = self.WQ(app_feature)  # (128, 2048) ==> (128, 64)
        w_k = self.WK(app_feature)  # (128, 2048) ==> (128, 64)
        w_k = w_k.transpose(0, 1)
        scaled_dot = torch.mm(w_q, w_k)  # (128, 128), each element is a appearance score between obj_n and obj_m
        scaled_dot = scaled_dot / np.sqrt(self.dim_k)  # (128, 128)

        # positional embedding
        w_g = self.relu(self.WG(position_embedding))  # (128, 128, 64) ==> (128, 128, 1)
        w_g = w_g.view(N, N)  # (128, 128), each element is a position score between obj_n and obj_m
        w_a = scaled_dot.view(N, N)
        w_mn = w_a + w_g  # merge appearance feature and geo feature
        w_mn = torch.nn.Softmax(dim=1)(w_mn)  # (128, 128), each element is a relation score between obj_n and obj_m

        w_v = self.WV(app_feature)  # (128, 2048) ==> (128, 64)
        output = torch.mm(w_mn, w_v)

        return output


