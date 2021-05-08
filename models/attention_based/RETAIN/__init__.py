# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import pickle as pickle
import random
from sklearn.metrics import roc_auc_score

# RETAIN
# Cite: Choi, E., Bahadori, M. T., Kulas, J. A., Schuetz, A., Stewart, W. F., & Sun, J. (2016).
# Retain: An interpretable predictive model for healthcare using reverse time attention mechanism.
# arXiv preprint arXiv:1608.05745.
#
# Code adapted from https://github.com/easyfan327/Pytorch-RETAIN/


class RetainNN(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        """
        num_embeddings(int): size of the dictionary of embeddings
        embedding_dim(int) the size of each embedding vector
        """
        # self.emb_layer = nn.Embedding(num_embeddings=params["num_embeddings"], embedding_dim=params["embedding_dim"])
        self.emb_layer = nn.Linear(
            in_features=params["num_embeddings"], out_features=params["embedding_dim"]
        )
        self.dropout = nn.Dropout(params["dropout_p"])
        self.variable_level_rnn = nn.GRU(
            params["var_rnn_hidden_size"], params["var_rnn_output_size"]
        )
        self.visit_level_rnn = nn.GRU(
            params["visit_rnn_hidden_size"], params["visit_rnn_output_size"]
        )
        self.variable_level_attention = nn.Linear(
            params["var_rnn_output_size"], params["var_attn_output_size"]
        )
        self.visit_level_attention = nn.Linear(
            params["visit_rnn_output_size"], params["visit_attn_output_size"]
        )
        self.output_dropout = nn.Dropout(params["output_dropout_p"])
        self.output_layer = nn.Linear(
            params["embedding_output_size"], params["num_class"]
        )

        self.var_hidden_size = params["var_rnn_hidden_size"]

        self.visit_hidden_size = params["visit_rnn_hidden_size"]

        self.n_samples = params["batch_size"]
        self.reverse_rnn_feeding = params["reverse_rnn_feeding"]

    def forward(self, input, var_rnn_hidden, visit_rnn_hidden):
        """
        :param input:
        :param var_rnn_hidden:
        :param visit_rnn_hidden:
        :return:
        """
        # emb_layer: input(*): LongTensor of arbitrary shape containing the indices to extract
        # emb_layer: output(*,H): where * is the input shape and H = embedding_dim
        # print("size of input:")
        # print(input.shape)
        v = self.emb_layer(input)
        # print("size of v:")
        # print(v.shape)
        v = self.dropout(v)

        # GRU:
        # input of shape (seq_len, batch, input_size)
        # seq_len: visit_seq_len
        # batch: batch_size
        # input_size: embedding dimension
        #
        # h_0 of shape (num_layers*num_directions, batch, hidden_size)
        # num_layers(1)*num_directions(1)
        # batch: batch_size
        # hidden_size:
        if self.reverse_rnn_feeding:
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(
                torch.flip(v, [0]), visit_rnn_hidden
            )
            alpha = self.visit_level_attention(torch.flip(visit_rnn_output, [0]))
        else:
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(
                v, visit_rnn_hidden
            )
            alpha = self.visit_level_attention(visit_rnn_output)
        visit_attn_w = F.softmax(alpha, dim=0)

        if self.reverse_rnn_feeding:
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(
                torch.flip(v, [0]), var_rnn_hidden
            )
            beta = self.variable_level_attention(torch.flip(var_rnn_output, [0]))
        else:
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(v, var_rnn_hidden)
            beta = self.variable_level_attention(var_rnn_output)
        var_attn_w = torch.tanh(beta)

        # print("beta attn:")
        # print(var_attn_w.shape)
        # '*' = hadamard product (element-wise product)
        attn_w = visit_attn_w * var_attn_w
        c = torch.sum(attn_w * v, dim=0)
        # print("context:")
        # print(c.shape)

        c = self.output_dropout(c)
        # print("context:")
        # print(c.shape)
        output = self.output_layer(c)
        # print("output:")
        # print(output.shape)
        output = F.softmax(output, dim=1)
        # print("output:")
        # print(output.shape)

        return output, var_rnn_hidden, visit_rnn_hidden

    def init_hidden(self, current_batch_size):
        return (
            torch.zeros(current_batch_size, self.var_hidden_size)
            .unsqueeze(0)
            .to(device),
            torch.zeros(current_batch_size, self.visit_hidden_size)
            .unsqueeze(0)
            .to(device),
        )
