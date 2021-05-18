import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from models.attention_based.helpers.train_darnn.constants import device


# RETAIN
# Cite: Choi, E., Bahadori, M. T., Kulas, J. A., Schuetz, A., Stewart, W. F., & Sun, J. (2016).
# Retain: An interpretable predictive model for healthcare using reverse time attention mechanism.
# arXiv preprint arXiv:1608.05745.
#
# Code adapted from https://github.com/easyfan327/Pytorch-RETAIN/


class RetainNN(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        var_rnn_hidden_size: int,
        var_rnn_output_size: int,
        visit_rnn_hidden_size: int,
        visit_rnn_output_size: int,
        visit_attn_output_size: int,
        var_attn_output_size: int,
        output_dropout_p: float,
        visit_level_dropout: float,
        variable_level_dropout: float,
        embedding_output_size: int,
        num_class: int,
        dropout_p: float = 0.01,
        batch_size: int = 2,
        reverse_rnn_feeding: bool = True,
        gru_lstm: bool = True,
    ):
        super().__init__()
        """
        num_embeddings(int): size of the dictionary of embeddings
        embedding_dim(int) the size of each embedding vector
        """
        # self.emb_layer = nn.Embedding(num_embeddings=params["num_embeddings"], embedding_dim=params["embedding_dim"])
        self.emb_layer = nn.Linear(
            in_features=num_embeddings, out_features=embedding_dim
        )
        self.dropout = nn.Dropout(dropout_p)

        if gru_lstm:
            self.variable_level_rnn = nn.LSTM(
                input_size=var_rnn_hidden_size,
                hidden_size=var_rnn_output_size,
                batch_first=True,
            )
            self.visit_level_rnn = nn.LSTM(
                input_size=visit_rnn_hidden_size,
                hidden_size=visit_rnn_output_size,
                batch_first=True,
            )
        else:
            self.variable_level_rnn = nn.GRU(
                input_size=var_rnn_hidden_size,
                hidden_size=var_rnn_output_size,
                batch_first=True,
            )
            self.visit_level_rnn = nn.GRU(
                input_size=visit_rnn_hidden_size,
                hidden_size=visit_rnn_output_size,
                batch_first=True,
            )
        self.variable_level_attention = nn.Linear(
            var_rnn_output_size, var_attn_output_size
        )
        self.visit_level_attention = nn.Linear(
            visit_rnn_output_size, visit_attn_output_size
        )
        self.output_dropout = nn.Dropout(output_dropout_p)
        self.output_layer = nn.Linear(embedding_output_size, num_class)

        self.var_hidden_size = var_rnn_hidden_size
        self.variable_level_dropout = nn.Dropout(variable_level_dropout)

        self.visit_hidden_size = visit_rnn_hidden_size
        self.visit_level_dropout = nn.Dropout(visit_level_dropout)

        self.n_samples = batch_size
        self.reverse_rnn_feeding = reverse_rnn_feeding

    def forward(self, input_data):
        """
        :param input_data:
        :return:
        """
        # emb_layer: input(*): LongTensor of arbitrary shape containing the indices to extract
        # emb_layer: output(*,H): where * is the input shape and H = embedding_dim
        # print("size of input:")
        # print(input_data.shape)
        v = self.emb_layer(input_data)
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
        # print("Visit Level started")
        var_rnn_hidden, visit_rnn_hidden = self.init_hidden(input_data.size(0))
        if self.reverse_rnn_feeding:
            self.visit_level_rnn.flatten_parameters()
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(
                torch.flip(v, [0]), (visit_rnn_hidden, visit_rnn_hidden)
            )
            alpha = self.visit_level_attention(torch.flip(visit_rnn_output, [0]))
        else:
            self.visit_level_rnn.flatten_parameters()
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(
                v, (visit_rnn_hidden, visit_rnn_hidden)
            )
            alpha = self.visit_level_attention(visit_rnn_output)
        visit_attn_w = self.visit_level_dropout(F.softmax(alpha, dim=0))

        # print("Variable Level started")
        if self.reverse_rnn_feeding:
            self.variable_level_rnn.flatten_parameters()
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(
                torch.flip(v, [0]), (var_rnn_hidden, var_rnn_hidden)
            )
            beta = self.variable_level_attention(torch.flip(var_rnn_output, [0]))
        else:
            self.variable_level_rnn.flatten_parameters()
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(
                v, (var_rnn_hidden, var_rnn_hidden)
            )
            beta = self.variable_level_attention(var_rnn_output)
        var_attn_w = self.variable_level_dropout(torch.tanh(beta))

        # print("beta attn:")
        # '*' = hadamard product (element-wise product)
        attn_w = visit_attn_w * var_attn_w
        c = torch.sum(attn_w * v, dim=1)
        # print("context:")
        # print(c.shape)

        c = self.output_dropout(c)
        # print("context:")
        output = self.output_layer(c)
        # print("output:")
        # print(output.shape)
        # print("output:")
        # print(output.shape)

        return output, alpha, var_attn_w

    def init_hidden(self, current_batch_size):
        return (
            torch.zeros(current_batch_size, self.var_hidden_size)
            .unsqueeze(0)
            .to(device),
            torch.zeros(current_batch_size, self.visit_hidden_size)
            .unsqueeze(0)
            .to(device),
        )
