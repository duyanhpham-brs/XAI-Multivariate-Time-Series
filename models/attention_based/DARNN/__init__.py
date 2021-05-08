from typing import Tuple
import torch
from torch import nn
from torch.autograd import Variable
from models.attention_based.helpers.train_darnn.constants import device


# Dual-stage Attention-based Recurrent Neural Network
# Cite: Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., & Cottrell, G. (2017).
# A dual-stage attention-based recurrent neural network for time series prediction.
# arXiv preprint arXiv:1704.02971.
# 
# Code adapted from https://github.com/AIStream-Peelout/flow-forecast/tree/master/flood_forecast/da_rnn
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
        gru_lstm: bool = True,
        num_layers: int = 1,
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
        self.num_layers = num_layers
        # Softmax fix
        self.softmax = nn.Softmax(dim=1)
        # print(input_size, hidden_size)
        if gru_lstm:
            self.lstm_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            self.gru_layer = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        self.attn_linear = nn.Linear(in_features=self.hidden_size * 2 + time_length, out_features=1)

    def forward(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_data: (batch_size, T - 1, input_size)
        # print(input_data.size())
        input_weighted = Variable(torch.zeros(self.batch_size, self.input_size, input_data.size(1))).to(
            device
        )
        input_encoded = Variable(torch.zeros(self.batch_size, self.input_size, self.hidden_size)).to(
            device
        )

        # hidden, cell: initial states with dimension hidden_size
        hidden = init_hidden(
            input_data, self.hidden_size, self.num_layers
        )  # 1 * batch_size * hidden_size
        cell = init_hidden(input_data, self.hidden_size, self.num_layers)

        # print(
        #     hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2).size(),
        #     cell.repeat(self.input_size, 1, 1).permute(1, 0, 2).size(),
        #     input_data.size(),
        # )

        # Eqn. 8: concatenate the hidden states with each predictor
        x = torch.cat(
            (
                hidden.repeat(input_data.size(1), 1, 1).permute(1, 0, 2),
                cell.repeat(input_data.size(1), 1, 1).permute(1, 0, 2),
                input_data,
            ),
            dim=2,
        )  # batch_size * input_size * (2*hidden_size + T - 1)
        # print(x.size())
        # Eqn. 8: Get attention weights
        x = self.attn_linear(
            x.view(-1, self.hidden_size * 2 + input_data.size(-1))
        )  # (batch_size * input_size) * 1
        # Eqn. 9: Softmax the attention weights
        # Had to replace functional with generic Softmax
        # (batch_size, input_size)
        attn_weights = self.softmax(x.view(-1, self.batch_size))
        # Eqn. 10: LSTM
        # (batch_size, input_size)

        # print(attn_weights.T.unsqueeze(2).size(), input_data.size())
        weighted_input = torch.mul(attn_weights.T.unsqueeze(2), input_data)
        # print(weighted_input.permute(0, 2, 1).size(), hidden.size(), cell.size())
        # Fix the warning about non-contiguous memory
        # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
        if self.gru_lstm:
            self.lstm_layer.flatten_parameters()
            _, generic_states = self.lstm_layer(
                weighted_input.permute(0, 2, 1), (hidden, cell)
            )
            cell = generic_states[1]
            hidden = generic_states[0]
        else:
            self.gru_layer.flatten_parameters()
            __, generic_states = self.gru_layer(weighted_input.permute(0, 2, 1), hidden)
            hidden = generic_states[0].unsqueeze(0)

            # Save output
        input_weighted = weighted_input
        input_encoded = hidden

        return input_weighted, input_encoded


class Decoder(nn.Module):
    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        input_size: int,
        time_length: int,
        out_feats=1,
        gru_lstm: bool = True,
        num_layers: int = 1,
    ):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.attn_layer = nn.Sequential(
            nn.Linear(
                2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size
            ),
            nn.Tanh(),
            nn.Linear(encoder_hidden_size, 1),
        )
        # Softmax fix
        self.softmax = nn.Softmax(dim=1)
        self.gru_lstm = gru_lstm
        if gru_lstm:
            self.lstm_layer = nn.LSTM(
                input_size=out_feats,
                hidden_size=decoder_hidden_size,
                num_layers=num_layers,
            )
        else:
            self.gru_layer = nn.GRU(
                input_size=out_feats,
                hidden_size=decoder_hidden_size,
                num_layers=num_layers,
            )

        self.fc = nn.Linear(encoder_hidden_size + time_length, out_feats)

        fc_final_out_feats = out_feats
        self.fc_final = nn.Linear(
            decoder_hidden_size + encoder_hidden_size, fc_final_out_feats
        )

        self.fc.weight.data.normal_()

    def forward(
        self, input_encoded: torch.Tensor, input_data: torch.Tensor
    ) -> torch.Tensor:

        hidden = init_hidden(input_encoded, self.decoder_hidden_size, self.num_layers)
        cell = init_hidden(input_encoded, self.decoder_hidden_size, self.num_layers)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
        # print(
        #     hidden.repeat(8, 1, 1).permute(1, 0, 2).size(),
        #     cell.repeat(8, 1, 1).permute(1, 0, 2).size(),
        #     input_encoded.size(),
        # )
        x = torch.cat(
            (
                hidden.repeat(input_encoded.size(1), 1, 1).permute(1, 0, 2),
                cell.repeat(input_encoded.size(1), 1, 1).permute(1, 0, 2),
                input_encoded,
            ),
            dim=2,
        )
        # print(x.size())
        # Eqn. 12 & 13: softmax on the computed attention weights
        # Had to replace functional with generic Softmax
        x = self.softmax(
            self.attn_layer(
                x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
            ).view(-1, self.input_size - 1)
        )  # (batch_size, T - 1)

        # Eqn. 14: compute context vector
        # print(
        #     x.unsqueeze(1).size(),
        #     input_encoded.size(),
        #     torch.bmm(x.unsqueeze(1), input_encoded).size(),
        # )
        context = torch.bmm(x.unsqueeze(1), input_encoded)

        # Eqn. 15
        # (batch_size, out_size)
        # print(
        #     context.size(),
        #     input_data.size(),
        #     context.repeat(input_encoded.size(1), input_data.size(1), 1).size(),
        # )
        y_tilde = self.fc(
            torch.cat((context.repeat(input_encoded.size(1), input_data.size(1), 1), input_data), dim=2)
        )
        # Eqn. 16: LSTM
        # print(y_tilde.size())
        if self.gru_lstm:
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(
                y_tilde.view(y_tilde.size(1), y_tilde.size(0), y_tilde.size(2)),
                (
                    hidden.repeat(1, y_tilde.size(0), 1),
                    cell.repeat(1, y_tilde.size(0), 1),
                ),
            )
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size
        else:
            self.gru_layer.flatten_parameters()
            __, generic_states = self.gru_layer(
                y_tilde.view(y_tilde.size(1), y_tilde.size(0), y_tilde.size(2)),
                hidden.repeat(1, y_tilde.size(0), 1),
            )
            hidden = generic_states[0].unsqueeze(0)

        # print(
        #     torch.cat(
        #         (hidden[0].unsqueeze(0), context.repeat(1, hidden[0].size(0), 1)), dim=2
        #     ).size()
        # )
        return self.fc_final(
            torch.cat(
                (hidden[0].unsqueeze(0), context.repeat(1, hidden[0].size(0), 1)), dim=2
            ).view(-1, self.decoder_hidden_size + self.encoder_hidden_size)
        )
