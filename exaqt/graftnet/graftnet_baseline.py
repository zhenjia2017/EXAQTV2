import json

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from util import get_config, load_dict, cal_accuracy, load_date, sparse_bmm,use_cuda

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor

class GRAFTNET(nn.Module):
    def __init__(self, pretrained_entity_embedding_file, pretrained_word_embedding_file, pretrained_relation_embedding_file,
                     num_layer, num_relation, num_entity,
                     num_word, entity_dim, word_dim, pagerank_lambda, fact_scale,
                     lstm_dropout, linear_dropout):

        """
        num_relation: number of relation including self-connection
        """
        super(GRAFTNET, self).__init__()
        self.num_layer = num_layer
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.num_word = num_word
        self.entity_dim = entity_dim
        self.word_dim = word_dim
        self.pagerank_lambda = pagerank_lambda
        self.fact_scale = fact_scale


        print("\n\nself.num_relation:", self.num_relation)
        print("\n\nself.num_word:", self.num_word)
        # initialize entity embedding
        self.entity_embedding = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=word_dim,
                                             padding_idx=num_entity)
        if pretrained_entity_embedding_file is not None:
            self.entity_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pretrained_entity_embedding_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.entity_embedding.weight.requires_grad = False

        self.entity_linear = nn.Linear(in_features=word_dim, out_features=entity_dim)

        # initialize relation embedding

        self.relation_embedding = nn.Embedding(num_embeddings=num_relation + 1, embedding_dim=word_dim,
                                               padding_idx=num_relation)
        if pretrained_relation_embedding_file is not None:
            self.relation_embedding.weight = nn.Parameter(
                torch.from_numpy(
                    np.pad(np.load(pretrained_relation_embedding_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))

        self.relation_linear = nn.Linear(in_features=word_dim, out_features=entity_dim)

        # initialize text embeddings

        self.word_embedding = nn.Embedding(num_embeddings=num_word + 1, embedding_dim=entity_dim, padding_idx=num_word)
        if pretrained_word_embedding_file is not None:
            self.word_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pretrained_word_embedding_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.word_embedding.weight.requires_grad = False

        # create LSTMs
        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim,
                                        batch_first=True,
                                        bidirectional=False)

        # self.query_rel_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, batch_first=True,
        #                                  bidirectional=False)  # not used
        #
        self.k = 3
        for i in range(self.num_layer):
            self.add_module('q2e_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('e2q_linear' + str(i), nn.Linear(in_features=self.k * entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=self.k * entity_dim, out_features=entity_dim))

            self.add_module('kb_head_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('kb_tail_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('kb_self_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))

        # dropout
        self.lstm_drop = nn.Dropout(p=lstm_dropout)
        self.linear_drop = nn.Dropout(p=linear_dropout)
        # non-linear activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # scoring function
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        # loss
        self.softmax_d1 = nn.Softmax(dim=1)
        self.bce_loss_logits = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        """
        :local_entity: global_id of each entity                     (batch_size, max_local_entity)
        :q2e_adj_mat: adjacency matrices (dense)                    (batch_size, max_local_entity, 1)
        :kb_adj_mat: adjacency matrices (sparse)                    (batch_size, max_fact, max_local_entity), (batch_size, max_local_entity, max_fact)
        :kb_fact_rel:                                               (batch_size, max_fact)
        :query_text: a list of words in the query                   (batch_size, max_query_word)
        :answer_dist: an distribution over local_entity             (batch_size, max_local_entity)
        """
        # local_entity, q2e_adj_mat, kb_adj_mat, kb_fact_rel, kb_fact_rels_emb, query_text, query_type, query_signal, query_date, answer_dist = batch
        local_entity, q2e_adj_mat, kb_adj_mat, kb_fact_rel, query_text, answer_dist = batch

        batch_size, max_local_entity = local_entity.shape

        _, max_fact = kb_fact_rel.shape
        _, max_query_word = query_text.shape

        # numpy to tensor
        local_entity = use_cuda(Variable(torch.from_numpy(local_entity).type('torch.LongTensor'), requires_grad=False))
        local_entity_mask = use_cuda((local_entity != self.num_entity).type('torch.FloatTensor'))

        kb_fact_rel = use_cuda(Variable(torch.from_numpy(kb_fact_rel).type('torch.LongTensor'), requires_grad=False))

        query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor'), requires_grad=False))
        answer_dist = use_cuda(Variable(torch.from_numpy(answer_dist).type('torch.FloatTensor'), requires_grad=False))

        query_mask = use_cuda((query_text != self.num_word).type('torch.FloatTensor'))


        # encode query
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim


        query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                                  self.init_hidden(1, batch_size,
                                                                                   self.entity_dim))  # 1, batch_size, entity_dim

        query_node_emb = query_node_emb.squeeze(dim=0).unsqueeze(dim=1)


        # build kb_adj_matrix from sparse matrix
        (e2f_batch, e2f_f, e2f_e, e2f_val), (f2e_batch, f2e_e, f2e_f, f2e_val) = kb_adj_mat
        entity2fact_index = torch.LongTensor([e2f_batch, e2f_f, e2f_e])
        entity2fact_val = torch.FloatTensor(e2f_val)
        entity2fact_mat = use_cuda(torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size(
            [batch_size, max_fact, max_local_entity])))  # batch_size, max_fact, max_local_entity

        fact2entity_index = torch.LongTensor([f2e_batch, f2e_e, f2e_f])
        fact2entity_val = torch.FloatTensor(f2e_val)
        fact2entity_mat = use_cuda(torch.sparse.FloatTensor(fact2entity_index, fact2entity_val,
                                                            torch.Size([batch_size, max_local_entity, max_fact])))

        # load fact embedding
        local_fact_emb = self.relation_embedding(kb_fact_rel)  # batch_size, max_fact, word_dim
        local_fact_emb = self.relation_linear(local_fact_emb)  # batch_size, max_fact, entity_dim

        # attention fact2question
        div = float(np.sqrt(self.entity_dim))

        local_fact_emb_trans = local_fact_emb.transpose(1, 2)
        fact2query_sim = torch.bmm(query_hidden_emb, local_fact_emb_trans) / div  # batch_size, max_query_word, max_fact
        fact2query_sim = self.softmax_d1(fact2query_sim + (
                    1 - query_mask.unsqueeze(dim=2)) * VERY_NEG_NUMBER)  # batch_size, max_query_word, max_fact
        fact2query_att = torch.sum(fact2query_sim.unsqueeze(dim=3) * query_hidden_emb.unsqueeze(dim=2),
                                   dim=1)  # batch_size, max_fact, entity_dim
        W = torch.sum(fact2query_att * local_fact_emb, dim=2) / div  # batch_size, max_fact
        W_max = torch.max(W, dim=1, keepdim=True)[0]  # batch_size, 1
        W_tilde = torch.exp(W - W_max)  # batch_size, max_fact
        e2f_softmax = sparse_bmm(entity2fact_mat.transpose(1, 2), W_tilde.unsqueeze(dim=2)).squeeze(
            dim=2)  # batch_size, max_local_entity
        e2f_softmax = torch.clamp(e2f_softmax, min=VERY_SMALL_NUMBER)

        pagerank_f = use_cuda(
            Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor'), requires_grad=True).squeeze(
                dim=2))  # batch_size, max_local_entity
        q2e_adj_mat = use_cuda(Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor'),
                                        requires_grad=False))  # batch_size, max_local_entity, 1
        assert pagerank_f.requires_grad == True
        # print("\n\n initial pagerank_f", pagerank_f)
        # load entity embedding
        local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim


        local_entity_emb = self.entity_linear(local_entity_emb)  # batch_size, max_local_entity, entity_dim

        # 1-hop GCN
        for i in range(self.num_layer):
            # get linear transformation functions for each layer
            q2e_linear = getattr(self, 'q2e_linear' + str(i))
            e2q_linear = getattr(self, 'e2q_linear' + str(i))
            e2e_linear = getattr(self, 'e2e_linear' + str(i))

            kb_self_linear = getattr(self, 'kb_self_linear' + str(i))
            kb_head_linear = getattr(self, 'kb_head_linear' + str(i))
            kb_tail_linear = getattr(self, 'kb_tail_linear' + str(i))
                        # start propagation
            next_local_entity_emb = local_entity_emb
            # STEP 1: propagate from question and facts to entities
            # question -> entity
            q2e_emb = q2e_linear(self.linear_drop(query_node_emb)).expand(batch_size, max_local_entity,
                                                                          self.entity_dim)  # batch_size, max_local_entity, entity_dim

            next_local_entity_emb = torch.cat((next_local_entity_emb, q2e_emb),
                                              dim=2)  # batch_size, max_local_entity, entity_dim * 2

            # fact -> entity

            e2f_emb = self.relu(kb_self_linear(local_fact_emb) + sparse_bmm(entity2fact_mat, kb_head_linear(
                    self.linear_drop(local_entity_emb))))  # batch_size, max_fact, entity_dim
            e2f_softmax_normalized = W_tilde.unsqueeze(dim=2) * sparse_bmm(entity2fact_mat,
                                                                               (pagerank_f / e2f_softmax).unsqueeze(
                                                                                   dim=2))  # batch_size, max_fact, 1

            e2f_emb = e2f_emb * e2f_softmax_normalized  # batch_size, max_fact, entity_dim
            f2e_emb = self.relu(kb_self_linear(local_entity_emb) + sparse_bmm(fact2entity_mat, kb_tail_linear(
                self.linear_drop(e2f_emb))))

            pagerank_f = self.pagerank_lambda * sparse_bmm(fact2entity_mat, e2f_softmax_normalized).squeeze(dim=2) + (
                    1 - self.pagerank_lambda) * pagerank_f  # batch_size, max_local_entity

            # STEP 2: combine embeddings from fact
            next_local_entity_emb = torch.cat((next_local_entity_emb, self.fact_scale * f2e_emb),
                                              dim=2)  # batch_size, max_local_entity, entity_dim * 3

            # STEP 3: propagate from entities to update question and facts

            # entity -> query
            query_node_emb = torch.bmm(pagerank_f.unsqueeze(dim=1), e2q_linear(self.linear_drop(next_local_entity_emb)))
            # update entity
            local_entity_emb = self.relu(
                e2e_linear(self.linear_drop(next_local_entity_emb)))  # batch_size, max_local_entity, entity_dim

        # calculate loss and make prediction
        score = self.score_func(self.linear_drop(local_entity_emb)).squeeze(dim=2)  # batch_size, max_local_entity
        loss = self.bce_loss_logits(score, answer_dist)

        score = score + (1 - local_entity_mask) * VERY_NEG_NUMBER
        pred_dist = self.sigmoid(score) * local_entity_mask
        pred = torch.max(score, dim=1)[1]

        return loss, pred, pred_dist

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))))

