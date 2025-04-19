# coding: utf-8
r"""
################################################
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from collections import defaultdict
import json

class FC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.constant_(self.layer.bias, 0.0)

    def forward(self, input):
        return self.layer(input)

class MLP(nn.Module):
    def __init__(self, dims, act='sigmoid', dropout=0):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(FC(dims[i - 1], dims[i]))
        self.act = getattr(F, act)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, input):
        curr_input = input
        for i in range(len(self.layers) ):
            hidden = self.layers[i](curr_input)
            hidden = self.act(hidden)
            if self.dropout:
                hidden = self.dropout(hidden)
            curr_input = hidden
        return curr_input

class AnchorConv(nn.Module):
    def __init__(self, dim, n_anchor, dim_anchor):
        super(AnchorConv, self).__init__()
        self.dim = dim
        self.n_anchor = n_anchor #
        self.dim_anchor = dim_anchor
        self.anchors = Anchors(self.n_anchor, self.dim_anchor)

        self.recv = MLP([self.dim_anchor, self.dim])
        self.send = MLP([self.dim, self.dim_anchor])

        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim_anchor)
        self.norm3 = nn.LayerNorm(self.dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.dim, num_heads=1, batch_first=True)
        self.linear = MLP([self.dim, self.dim])

    def forward(self,input):
        input = self.norm1(input)
        s = self.send(input)
        r, anchor = self.anchors(s)
        r_ = self.norm2(r)
        a = self.recv(r_)
        a = self.linear(a)
        return a, r, anchor


class Anchors(nn.Module):
    def __init__(self, n_anchor, dim_anchor):
        super(Anchors, self).__init__()
        self.anchor_emb = nn.Parameter(torch.empty(n_anchor, dim_anchor))
        nn.init.xavier_uniform_(self.anchor_emb)
        self.attention = nn.MultiheadAttention(embed_dim=dim_anchor, num_heads=1, batch_first=True)

    def forward(self, input): ##num_nodes, dim_anchor
        input, _ = self.attention(input, self.anchor_emb, self.anchor_emb)
        return input, self.anchor_emb

class AGG_LRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(AGG_LRec, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.reg_weight = config['reg_weight']
        self.dropout = config['dropout']
        self.lambda_reg = config['lambda_reg']
        self.temperature = config['temperature']
        self.n_anchor = config['n_anchor']
        self.dim_anchor = config['dim_anchor']
        self.vt_loss = config['vt_loss']
        self.n_layers_v = config['n_layers_v']
        self.n_layers_t = config['n_layers_t']
        self.cluster_loss = config["cluster_loss"]
        self.cluster_beta = config['cluster_beta']
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.user_embedding_v = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_embedding_t = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.user_embedding_v.weight)
        nn.init.xavier_uniform_(self.user_embedding_t.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(self.dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(self.dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        self.inter = self.find_inter(self.image_original_adj, self.text_original_adj)
        self.ii_adj = self.add_edge(self.inter)
        self.norm_adj = self.get_adj_mat(self.ii_adj.tolil())
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        self.masked_adj = None
        self.edge_indices, self.edge_values = self.get_edge_info()

        if self.v_feat is not None:
            self.image_trs = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.embedding_dim),
                nn.Softmax()
            )
        if self.t_feat is not None:
            self.text_trs = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.embedding_dim),
                nn.Softmax()
            )
        self.softmax = nn.Softmax(dim=-1)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        # self.tau = 0.5

        self.anchor_conv_v = AnchorConv(self.embedding_dim,  self.n_anchor, self.dim_anchor)
        self.anchor_conv_t = AnchorConv(self.embedding_dim, self.n_anchor, self.dim_anchor)

        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.n_users + self.n_items, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)
    # def pre_epoch_processing(self):
    #     pass
    def find_inter(self, image_adj, text_adj):
        inter_file = os.path.join(self.dataset_path, 'inter.json')
        if os.path.exists(inter_file):
            with open(inter_file) as f:
                inter = json.load(f)
        else:
            j = 0
            inter = defaultdict(list)
            img_sim = []
            txt_sim = []
            for i in range(0, len(image_adj._indices()[0])):
                img_id = image_adj._indices()[0][i]
                txt_id = text_adj._indices()[0][i]
                assert img_id == txt_id
                id = img_id.item()
                img_sim.append(image_adj._indices()[1][j].item())
                txt_sim.append(text_adj._indices()[1][j].item())

                if len(img_sim) == 10 and len(txt_sim) == 10:
                    it_inter = list(set(img_sim) & set(txt_sim))
                    inter[id] = [v for v in it_inter if v != id]
                    img_sim = []
                    txt_sim = []

                j += 1

            with open(inter_file, "w") as f:
                json.dump(inter, f)

        return inter

    def add_edge(self, inter):
        sim_rows = []
        sim_cols = []
        for id, vs in inter.items():
            if len(vs) == 0:
                continue
            for v in vs:
                sim_rows.append(int(id))
                sim_cols.append(v)

        sim_rows = torch.tensor(sim_rows)
        sim_cols = torch.tensor(sim_cols)
        sim_values = [1] * len(sim_rows)

        item_adj = sp.coo_matrix((sim_values, (sim_rows, sim_cols)), shape=(self.n_items, self.n_items), dtype=int)
        return item_adj

    def pre_epoch_processing(self):
        # pass
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users+self.n_items, self.n_users+self.n_items)))
        self.masked_adj = torch.sparse.FloatTensor(keep_indices, keep_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.4)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def _normalize_adj_mm(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values
    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        edges[1] += self.n_users
        all_indices = torch.cat((edges, torch.flip(edges, [0])), 1)
        adj_rows = self.ii_adj.row + self.n_users
        adj_cols = self.ii_adj.col + self.n_users
        all_indices = torch.cat((all_indices, torch.tensor(np.vstack((adj_rows, adj_cols)))), dim=1)
        # edge normalized values
        values = self._normalize_adj_mm(all_indices, torch.Size((self.n_users+self.n_items, self.n_users+self.n_items)))
        return all_indices, values

    def get_adj_mat(self, item_adj):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat[self.n_users:, self.n_users:] = item_adj
        adj_mat = adj_mat.todok()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            d_inv_sqrt_last = np.power(rowsum, -0.4).flatten()
            d_inv_sqrt_last[np.isinf(d_inv_sqrt_last)] = 0.
            d_mat_inv_sqrt_last = sp.diags(d_inv_sqrt_last)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv_sqrt_last)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def GCN(self, user, item, num_layers, adj):
        ego_embeddings = torch.cat([user, item], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(num_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)  # GCN
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        return all_embeddings

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        # anchor
        image_item_an ,r_i, image_anchor_embeds = self.anchor_conv_v(image_feats)
        text_item_an ,r_t, text_anchor_embeds = self.anchor_conv_t(text_feats)

        # User-Item View
        user_embeds = self.user_embedding.weight
        item_embeds = self.item_id_embedding.weight
        content_embeds = self.GCN(user_embeds , item_embeds, self.n_ui_layers,adj)

        image_item_embeds = torch.multiply(item_embeds, image_item_an)
        text_item_embeds = torch.multiply(item_embeds, text_item_an)

        # Item-Item View
        for i in range(self.n_layers_v):
            image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        for i in range(self.n_layers_t):
             text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # Behavior-Aware Fuser
        att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
            dim=1) * text_embeds
        sep_image_embeds = image_embeds - common_embeds
        sep_text_embeds = text_embeds - common_embeds
        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
        side_embeds = (sep_image_embeds + sep_text_embeds + common_embeds) / 3

        all_embeds = content_embeds + side_embeds
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds, image_item_an, text_item_an, r_i ,image_anchor_embeds, r_t ,text_anchor_embeds


        return all_embeddings_users, all_embeddings_items
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        return mf_loss, regularizer

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1) #余弦相似度
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def align_vt(self, embed1, embed2):
        emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
        emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)

        vt_loss = (torch.abs(emb1_var - emb2_var) + torch.abs(emb1_mean - emb2_mean)).mean()

        return vt_loss

    def kmeans_clustering(self, item_embeds, anchor_embeds):
        cosine_similarities = F.cosine_similarity(item_embeds.unsqueeze(1), anchor_embeds.unsqueeze(0), dim=2)
        distances = 1 - cosine_similarities
        cluster_assignments = torch.argmin(distances, dim=1)

        positive_loss = torch.mean(distances[range(item_embeds.size(0)), cluster_assignments])
        negative_distances = torch.max(distances, dim=1)[0]
        negative_loss = torch.mean(negative_distances)

        total_loss = positive_loss - self.cluster_beta * negative_loss
        return total_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds, image_embeds, text_embeds, image_item_embeds ,image_anchor_embeds, text_item_embeds ,text_anchor_embeds= self.forward(self.masked_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        batch_mf_loss, batch_mf_l2 = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items],   self.temperature) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users],  self.temperature)

        vt_loss = self.align_vt(image_embeds, text_embeds)

        cluster_loss = self.kmeans_clustering(image_item_embeds, image_anchor_embeds) + self.kmeans_clustering(text_item_embeds, text_anchor_embeds)

        l2_reg = batch_mf_l2
        for param in [self.gate_image_prefer[0].weight, self.gate_text_prefer[0].weight, self.image_trs[0].weight,
                      self.text_trs[0].weight]:
            l2_reg += torch.norm(param, 2) ** 2
        for param in self.anchor_conv_v.parameters():
            l2_reg += torch.norm(param, 2) ** 2
        for param in self.anchor_conv_t.parameters():
            l2_reg += torch.norm(param, 2) ** 2
        return batch_mf_loss + self.cl_loss * cl_loss + self.vt_loss * vt_loss + self.cluster_loss * cluster_loss + self.lambda_reg * l2_reg

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]
        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
