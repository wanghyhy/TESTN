import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
from layers import SpatialEvoConv, MLPClassifier, TimeDiscriminator

class TESTN(nn.Layer):
    def __init__(self, in_dim, h_dim, num_rels, num_neighbor, time_list, dropout=0, boundaries=None):
        super(TESTN, self).__init__()
        self.w_relation = self.create_parameter(shape=[num_rels, h_dim], default_initializer=nn.initializer.XavierUniform())
        self.boundaries = boundaries
        self.dist_embed = nn.Embedding(len(self.boundaries) + 1, h_dim)
        self.simnet = nn.Bilinear(h_dim, h_dim, 1)

        self.embedding = nn.Embedding(in_dim, h_dim) 
        self.gnn = SpatialEvoConv(h_dim, h_dim, h_dim, num_neighbor, time_list, self.boundaries, self.dist_embed, dropout, activation=F.relu, hop1_fc=False, merge='sum')

    def calc_score(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = paddle.sum(s * r * o, axis=1)
        return score
    
    def calc_tri_score(self, embedding, pairs):
        pair_embed = embedding[pairs[:,0]] * embedding[pairs[:,1]]
        pair_embed_ = embedding[pairs[:,2]] * embedding[pairs[:,1]]
        distance = paddle.nn.functional.pairwise_distance(pair_embed, pair_embed_)
        return distance
    
    def calc_pair_score(self, embedding, pairs):
        pair_embed = embedding[pairs[:,0]] * embedding[pairs[:,1]]
        pair_embed_ = embedding[pairs[:,2]] * embedding[pairs[:,1]]
        score = self.simnet(pair_embed,pair_embed_)
        return score.squeeze(-1)
    
    def filter_o(self, triplets_to_filter, target_s, target_r, target_o, train_ids):
        target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
        filtered_o = []
        if (target_s, target_r, target_o) in triplets_to_filter:
            triplets_to_filter.remove((target_s, target_r, target_o))
        for o in train_ids:
            if (target_s, target_r, o) not in triplets_to_filter:
                filtered_o.append(o)
        return paddle.to_tensor(filtered_o)

    @paddle.no_grad()
    def rank_score_filtered(self, embedding, test_triplets, train_triplets, valid_triplets):
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]
        triplets_to_filter = paddle.concat([train_triplets, valid_triplets, test_triplets]).tolist()
        train_ids = paddle.unique(paddle.concat([train_triplets[:,0], train_triplets[:,2]])).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        ranks = []

        for idx in range(test_size):
            target_s = s[idx]
            target_r = r[idx]
            target_o = o[idx]

            filtered_o = self.filter_o(triplets_to_filter, target_s, target_r, target_o, train_ids)
            if len((filtered_o == target_o).nonzero()) == 0:
                continue
            target_o_idx = int((filtered_o == target_o).nonzero())
            emb_s = embedding[target_s]
            emb_r = self.w_relation[target_r]
            emb_o = embedding[filtered_o]
            emb_triplet = emb_s * emb_r * emb_o
            scores = F.sigmoid(paddle.sum(emb_triplet, axis=1))
            indices = paddle.argsort(scores, descending=True)
            rank = int((indices == target_o_idx).nonzero())
            ranks.append(rank)

        return np.array(ranks)

    def forward(self, g, h):
        h = self.embedding(h.squeeze())
        h = self.gnn.forward(g, h)
        return h

    def get_loss(self, g, embed, triplets, labels,tri_pos,tri_neg, dir_pair_data, pair_label,ssl_weight=2.0):
        predict_tloss = 0
        tloss3=0
        tpair_loss=0
        loss=0
        for idx in range(len(embed)):

            pair_score = self.calc_pair_score(embed[idx], dir_pair_data[idx])
            pair_loss = ssl_weight* F.binary_cross_entropy_with_logits(pair_score, pair_label[idx],reduction='sum')

            loss += pair_loss
            tpair_loss += pair_loss

            score = self.calc_score(embed[idx], triplets[idx])
            predict_loss = F.binary_cross_entropy_with_logits(score, labels[idx],reduction='sum')
            loss += predict_loss
            predict_tloss += predict_loss

        return loss,predict_tloss,loss,tpair_loss