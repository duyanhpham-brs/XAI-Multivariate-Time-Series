from typing import Tuple
import torch
from torch import nn
from torch.autograd import Variable
from models.attention_based.helpers.train_darnn.constants import device


# GeoMAN
# Cite: Liang, Y., Ke, S., Zhang, J., Yi, X., & Zheng, Y. (2018, July).
# Geoman: Multi-level attention networks for geo-sensory time series prediction.
# In IJCAI (pp. 3428-3434).
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
        local_attn_dropout: float,
        global_attn_dropout: float,
        gru_lstm: bool = True,
        num_layers: int = 1,
        local_attn: bool = True,
        global_attn: bool = True,
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
        self.local_attn = local_attn
        self.global_attn = global_attn
        self.local_attn_dropout = nn.Dropout(local_attn_dropout)
        self.global_attn_dropout = nn.Dropout(global_attn_dropout)
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

        self.local_attn_linear = nn.Sequential(
            nn.Linear(
                in_features=self.hidden_size * 2 + time_length,
                out_features=self.hidden_size,
            ),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )

        self.global_attn_linear = nn.Sequential(
            nn.Linear(
                in_features=self.hidden_size * 2 + 2 * time_length,
                out_features=self.hidden_size,
            ),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_data: (batch_size, T - 1, input_size)
        # print(input_data.size())
        input_weighted = Variable(
            torch.zeros(self.batch_size, self.input_size, input_data.size(1))
        ).to(device)
        input_encoded = Variable(
            torch.zeros(self.batch_size, self.input_size, self.hidden_size)
        ).to(device)

        # hidden, cell: initial states with dimension hidden_size
        hidden1 = init_hidden(
            input_data, self.hidden_size, self.num_layers
        )  # 1 * batch_size * hidden_size
        cell1 = init_hidden(input_data, self.hidden_size, self.num_layers)

        hidden2 = init_hidden(
            input_data, self.hidden_size, self.num_layers
        )  # 1 * batch_size * hidden_size
        cell2 = init_hidden(input_data, self.hidden_size, self.num_layers)

        # print(
        #     hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2).size(),
        #     cell.repeat(self.input_size, 1, 1).permute(1, 0, 2).size(),
        #     input_data.size(),
        # )
        # print("Encoder started")
        # Eqn. 8: concatenate the hidden states with each predictor
        if self.local_attn:
            local_total_attn = []
            for i in range(input_data.size(1)):
                x = torch.cat(
                    (
                        hidden1.permute(1, 0, 2),
                        cell1.permute(1, 0, 2),
                        input_data[:, i, :].unsqueeze(1),
                    ),
                    dim=2,
                )  # batch_size * input_size * (2*hidden_size + T - 1)
                # print(x.size())
                # Eqn. 8: Get attention weights
                x = self.local_attn_linear(
                    x.view(-1, self.hidden_size * 2 + input_data.size(-1))
                )  # (batch_size * input_size) * 1
                # Eqn. 9: Softmax the attention weights
                # Had to replace functional with generic Softmax
                # (batch_size, input_size)
                local_attn_weights = self.local_attn_dropout(self.softmax(x.view(-1, self.batch_size)))
                local_total_attn.append(local_attn_weights)
            local_total_attn = torch.cat(local_total_attn, dim=0)
            local_weighted_input = torch.mul(
                local_total_attn.T.unsqueeze(2), input_data
            )

        if self.global_attn:
            x = torch.cat(
                (
                    hidden2.repeat(input_data.size(1), 1, 1).permute(1, 0, 2),
                    cell2.repeat(input_data.size(1), 1, 1).permute(1, 0, 2),
                    input_data,
                    input_data,
                ),
                dim=2,
            )  # batch_size * input_size * (2*hidden_size + T - 1)
            # print(x.size())
            # Eqn. 8: Get attention weights
            x = self.global_attn_linear(
                x.view(-1, self.hidden_size * 2 + 2 * input_data.size(-1))
            )  # (batch_size * input_size) * 1
            # Eqn. 9: Softmax the attention weights
            # Had to replace functional with generic Softmax
            # (batch_size, input_size)
            global_attn_weights = self.global_attn_dropout(self.softmax(x.view(-1, self.batch_size)))
            global_weighted_input = torch.mul(
                global_attn_weights.T.unsqueeze(2), input_data
            )
        # Eqn. 10: LSTM
        # (batch_size, input_size)
        # print(attn_weights.T.unsqueeze(2).size(), input_data.size())
        if self.local_attn and self.global_attn:
            weighted_input = torch.cat(
                (local_weighted_input, global_weighted_input), dim=0
            )
            hidden = torch.cat((hidden1, hidden2), dim=1)
            cell = torch.cat((cell1, cell2), dim=1)
        elif self.local_attn:
            weighted_input = local_weighted_input
            hidden = hidden1
            cell = cell1
        elif self.global_attn:
            weighted_input = global_weighted_input
            hidden = hidden2
            cell = cell2
        else:
            raise RuntimeError

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
        temp_attn_dropout: float,
        output_dropout: float,
        out_feats=1,
        gru_lstm: bool = True,
        num_layers: int = 1,
        local_attn: bool = True,
        global_attn: bool = True,
    ):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.local_attn = local_attn
        self.global_attn = global_attn
        self.temp_attn_dropout = nn.Dropout(temp_attn_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

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

        if local_attn and global_attn:
            self.fc = nn.Linear(encoder_hidden_size * 2 + time_length, out_feats)

            fc_final_out_feats = out_feats
            self.fc_final = nn.Linear(
                decoder_hidden_size + encoder_hidden_size * 2, fc_final_out_feats
            )
        else:
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
        # print("Decoder started")
        # print(
        #     hidden.repeat(input_encoded.size(1), 1, 1).permute(1, 0, 2).size(),
        #     cell.repeat(input_encoded.size(1), 1, 1).permute(1, 0, 2).size(),
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
        x = self.temp_attn_dropout(self.softmax(
            self.attn_layer(
                x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
            ).view(-1, 1)
        ))  # (batch_size, T - 1)

        # Eqn. 14: compute context vector
        # print(
        #     x.unsqueeze(1).size(),
        #     input_encoded.view(-1, 1, self.decoder_hidden_size).size(),
        #     # torch.bmm(x.unsqueeze(1), input_encoded).size(),
        # )
        context = torch.bmm(
            x.unsqueeze(1), input_encoded.view(-1, 1, self.decoder_hidden_size)
        )

        # Eqn. 15
        # (batch_size, out_size)
        # print(
        #     context.size(),
        #     input_data.size(),
        #     context.repeat(1, input_data.size(1), 1).size(),
        # )
        y_tilde = self.output_dropout(self.fc(
            torch.cat(
                (
                    context.repeat(1, input_data.size(1), 1).view(
                        input_data.size(0), input_data.size(1), -1
                    ),
                    input_data,
                ),
                dim=2,
            )
        ))
        # print(y_tilde.size())
        # Eqn. 16: LSTM
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
            cell = lstm_output[1]
        else:
            self.gru_layer.flatten_parameters()
            __, generic_states = self.gru_layer(
                y_tilde.view(y_tilde.size(1), y_tilde.size(0), y_tilde.size(2)),
                hidden.repeat(1, y_tilde.size(0), 1),
            )
            hidden = generic_states[0].unsqueeze(0)

        # print(
        #     hidden[0].unsqueeze(0).size(),
        #     context.view(context.size(1), hidden[0].size(0), -1).size(),
        # )
        # print(
        #     "Final size: ",
        #     self.fc_final(
        #         torch.cat(
        #             (
        #                 hidden[0].unsqueeze(0),
        #                 context.view(context.size(1), hidden[0].size(0), -1),
        #             ),
        #             dim=2,
        #         ).view(-1, self.decoder_hidden_size + self.encoder_hidden_size * 2)
        #     ).size(),
        # )
        if self.local_attn and self.global_attn:
            return self.fc_final(
                torch.cat(
                    (
                        hidden[0].unsqueeze(0),
                        context.view(context.size(1), hidden[0].size(0), -1),
                    ),
                    dim=2,
                ).view(-1, self.decoder_hidden_size + self.encoder_hidden_size * 2)
            )
        else:
            return self.fc_final(
                torch.cat(
                    (
                        hidden[0].unsqueeze(0),
                        context.view(context.size(1), hidden[0].size(0), -1),
                    ),
                    dim=2,
                ).view(-1, self.decoder_hidden_size + self.encoder_hidden_size)
            )
