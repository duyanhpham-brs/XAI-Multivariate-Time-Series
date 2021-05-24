from typing import Tuple
import torch
from torch import nn
from torch.autograd import Variable
from models.attention_based.helpers.train_darnn.constants import device


# STAM - Spatiotemporal Attention for Multivariate Time Series Prediction and Interpretation
# Cite: Gangopadhyay, T., Tan, S. Y., Jiang, Z., Meng, R., & Sarkar, S. (2020).
# Spatiotemporal attention for multivariate time series prediction and interpretation.
# arXiv preprint arXiv:2008.04882.
#
# Code adapted from https://github.com/arleigh418/Paper-Implementation-DSTP-RNN-For-Stock-Prediction-Based-On-DA-RNN/blob/master/DSTP_RNN.py
#
# Converted from univariate time series regression to multivariate time series classification


def init_hidden(x, hidden_size: int, num_layers: int) -> torch.autograd.Variable:
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return Variable(torch.zeros(num_layers, x.size(0), hidden_size)).to(x.device)


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        time_length: int,
        hidden_size: int,
        batch_size: int,
        spat_dropout: float,
        temp_dropout: float,
        spatial_emb_size: int = 100,
        gru_lstm: bool = True,
    ):
        """
        input size: number of underlying factors
        hidden_size: dimension of the hidden stats
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gru_lstm = gru_lstm
        self.time_length = time_length
        self.spatial_emb_size = spatial_emb_size
        # Softmax fix
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.s_dropout = nn.Dropout(spat_dropout)
        self.t_dropout = nn.Dropout(temp_dropout)
        # print(input_size, hidden_size)
        if gru_lstm:
            self.temporal_emb_converter = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
            )
        else:
            self.temporal_emb_converter = nn.GRU(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
            )

        self.spatial_emb_converter = []
        for _ in range(self.input_size):
            self.spatial_emb_converter.append(
                nn.Linear(
                    in_features=self.time_length, out_features=self.spatial_emb_size
                )
            )

    def forward(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("Encoder started")
        # embedding hidden, cell size
        input_data = input_data.to(device)
        hidden_emb = init_hidden(
            input_data[:, :, 0], self.hidden_size, 2
        )  # 2 * batch_size * hidden_size
        cell_emb = init_hidden(input_data[:, :, 0], self.hidden_size, 2)

        # Build spatial embeddings
        spatial_emb = Variable(
            torch.zeros(input_data.size(0), self.input_size, self.spatial_emb_size)
        ).to(device)
        for size in range(self.input_size):
            spatial_emb[:, size, :] = self.s_dropout(self.spatial_emb_converter[size](
                input_data[:, size, :]
            ).to(device))

        # Build temporal embeddings
        temp_emb = []
        for i in range(self.time_length):
            self.temporal_emb_converter.flatten_parameters()
            _, generic_states = self.temporal_emb_converter(
                input_data[:, :, i].unsqueeze(2), (hidden_emb, cell_emb)
            )
            cell_emb = self.t_dropout(generic_states[1])
            hidden_emb = self.t_dropout(generic_states[0])
            temp_emb.append(hidden_emb[1].unsqueeze(0))

            # print(input_encoded)

        # print(input_weighted.size())

        return spatial_emb, temp_emb


class Decoder(nn.Module):
    def __init__(
        self,
        time_length,
        encoder_hidden_size,
        spat_attn_dropout: float,
        temp_attn_dropout: float,
        out_dropout: float,
        input_size: int,
        batch_size: int,
        out_feats: int = 1,
        spatial_emb_size: int = 100,
        gru_lstm: bool = True,
    ):
        super().__init__()
        self.time_length = time_length
        self.encoder_hidden_size = encoder_hidden_size
        self.spatial_emb_size = spatial_emb_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.s_attn_dropout = nn.Dropout(spat_attn_dropout)
        self.t_attn_dropout = nn.Dropout(temp_attn_dropout)
        self.o_dropout = nn.Dropout(out_dropout)

        if gru_lstm:
            self.spatial_rnn = nn.LSTM(
                input_size=encoder_hidden_size * 2 + 1,
                hidden_size=encoder_hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.temporal_rnn = nn.LSTM(
                input_size=encoder_hidden_size * 2 + 1,
                hidden_size=encoder_hidden_size,
                num_layers=1,
                batch_first=True,
            )
        else:
            self.spatial_rnn = nn.GRU(
                input_size=encoder_hidden_size * 2 + 1,
                hidden_size=encoder_hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.temporal_rnn = nn.GRU(
                input_size=encoder_hidden_size * 2 + 1,
                hidden_size=encoder_hidden_size,
                num_layers=1,
                batch_first=True,
            )

        self.spatial_attn_linear = nn.Linear(
            in_features=self.encoder_hidden_size + self.spatial_emb_size, out_features=1
        )
        self.spatial_attn_wrapper = nn.Linear(
            in_features=self.spatial_emb_size, out_features=1
        )

        self.temporal_attn_linear = nn.Linear(
            in_features=self.encoder_hidden_size * 2, out_features=1
        )
        self.temporal_attn_wrapper = nn.Linear(
            in_features=self.encoder_hidden_size, out_features=1
        )

        self.fc = nn.Linear(encoder_hidden_size * 2 + self.input_size, out_feats)

        fc_final_out_feats = out_feats
        self.fc_final = nn.Linear(self.time_length * out_feats, fc_final_out_feats)

        self.fc_final.weight.data.normal_()

    def forward(
        self,
        input_data: torch.Tensor,
        spatial_emb: torch.Tensor,
        temp_emb: torch.Tensor,
    ) -> torch.Tensor:
        input_data = input_data.to(device)
        input_weighted = Variable(
            torch.zeros(1, input_data.size(0), self.encoder_hidden_size * 2)
        ).to(device)
        input_encoded = Variable(
            torch.zeros(1, input_data.size(0), self.encoder_hidden_size * 2)
        ).to(device)

        spat_hidden = init_hidden(input_data[:, :, 0], self.encoder_hidden_size, 1)
        spat_cell = init_hidden(input_data[:, :, 0], self.encoder_hidden_size, 1)

        temp_hidden = init_hidden(input_data[:, :, 0], self.encoder_hidden_size, 1)
        temp_cell = init_hidden(input_data[:, :, 0], self.encoder_hidden_size, 1)

        spatial_weighted_input = 0
        x1_all = []
        for size in range(self.input_size):
            # print(spatial_emb[:,size].unsqueeze(1).size(), spat_hidden.repeat(spatial_emb.size(1), 1, 1).permute(1, 0, 2).size())
            x1 = torch.cat(
                (
                    spat_hidden.permute(1, 0, 2),
                    spatial_emb[:, size].unsqueeze(1),
                ),
                dim=2,
            )
            # print(x1.size())
            x1 = self.spatial_attn_linear(x1)
            x1_all.append(x1)
            spatial_attn_weights = self.s_attn_dropout(self.softmax(self.relu(x1)))
            # print(spatial_attn_weights.size(), spatial_emb[:,size].size())
            spatial_weighted_input += torch.mul(
                spatial_attn_weights, spatial_emb[:, size].unsqueeze(1)
            ).view(input_data.size(0), -1)
        # print(spatial_weighted_input.size())
        # print(spatial_weighted_input.size())
        spatial_context = self.relu(self.spatial_attn_wrapper(spatial_weighted_input))
        # print(spatial_context.size(), input_weighted.size())
        spatial_concat = torch.cat(
            (
                spatial_context.T.unsqueeze(2),
                input_weighted,
            ),
            dim=2,
        )

        # print(f"Step {i + 1} / {self.time_length}")
        # Calculate spatial embeddings

        # print(spatial_emb.size())
        # Calculate spatial attention weights

        # print(spatial_concat.size())
        self.spatial_rnn.flatten_parameters()
        _, generic_states1 = self.spatial_rnn(
            spatial_concat.permute(1, 0, 2), (spat_hidden, spat_cell)
        )
        spat_cell = generic_states1[1]
        spat_hidden = generic_states1[0]

        # print(spat_hidden.size())

        # embedding hidden, cell size

        # print(temp_emb[i].size(), temp_hidden.size())
        temporal_weighted_input = 0
        x2_all = []
        for i in range(self.time_length):
            x2 = torch.cat(
                (
                    temp_emb[i],
                    temp_hidden.repeat(temp_emb[i].size(0), 1, 1),
                ),
                dim=2,
            )
            # print(x2.size())
            x2 = self.temporal_attn_linear(x2)
            # print(x2.size())
            x2_all.append(x2)
            temporal_attn_weights = self.t_attn_dropout(self.softmax(
                self.relu(x2)
            ))
            # print(
            #     torch.mul(
            #         temporal_attn_weights.repeat(1, 1, input_data.size(1)).permute(
            #             0, 2, 1
            #         ),
            #         input_data[:, :, i].unsqueeze(2),
            #     ).size()
            # )
            # print(temporal_attn_weights.size(), temp_emb[i].size())
            temporal_weighted_input += torch.mul(
                temporal_attn_weights,
                temp_emb[i],
            ).reshape(input_data.size(0), -1)
        # print(temporal_weighted_input.size())

        temporal_context = self.relu(
            self.temporal_attn_wrapper(temporal_weighted_input)
        )
        # print(temporal_context.size())
        temporal_concat = torch.cat(
            (
                temporal_context.T.unsqueeze(2),
                input_weighted,
            ),
            dim=2,
        )
        # print(temporal_concat.size())
        self.temporal_rnn.flatten_parameters()
        _, generic_states2 = self.temporal_rnn(
            temporal_concat.permute(1, 0, 2), (temp_hidden, temp_cell)
        )
        temp_cell = generic_states2[1]
        temp_hidden = generic_states2[0]
        # print(temp_hidden.size())

        input_encoded = torch.cat((spat_hidden, temp_hidden), dim=2)

        input_weighted = torch.cat((spat_cell, temp_cell), dim=2)

        # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
        # print(
        #     hidden.repeat(8, 1, 1).permute(1, 0, 2).size(),
        #     cell.repeat(8, 1, 1).permute(1, 0, 2).size(),
        #     input_encoded.size(),
        # )
        # print(input_data.permute(0,2,1).size(), input_encoded.view(-1, 1, self.encoder_hidden_size * 2).repeat(1, self.input_size, 1).size())
        context = torch.bmm(
            input_data.permute(0,2,1), input_encoded.view(-1, 1, self.encoder_hidden_size * 2).repeat(1, self.input_size, 1)
        )
        # print(context.size())
        
        y_tilde = self.o_dropout(self.fc(
            torch.cat(
                (
                    context,
                    input_data.permute(0,2,1),
                ),
                dim=2,
            )
        ))
        
        # print(y_tilde.size())
        return self.fc_final(y_tilde.view(input_data.size(0), -1)), x1_all, x2_all
