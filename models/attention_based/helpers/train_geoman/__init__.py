# pylint: disable=not-callable
# to fix torch.tensor lint error
import typing
from typing import Tuple
import json
import os
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from models.attention_based.helpers.train_darnn.constants import device
from models.attention_based.GEOMAN import Encoder, Decoder
from models.attention_based.helpers.train_darnn.custom_types import (
    GeoMan,
    TrainData,
    TrainConfig,
    TestConfig,
    TestData,
)
from models.attention_based.helpers.train_darnn.utils import numpy_to_tvar

# GeoMAN
# Cite: Liang, Y., Ke, S., Zhang, J., Yi, X., & Zheng, Y. (2018, July). 
# Geoman: Multi-level attention networks for geo-sensory time series prediction. 
# In IJCAI (pp. 3428-3434).
def GeoMAN(
    train_data: TrainData,
    test_data: TestData,
    n_targs: int,
    local_attn_dropout: float = 0.3,
    global_attn_dropout: float = 0.3,
    temp_attn_dropout: float = 0.3,
    output_dropout: float = 0.3,
    encoder_hidden_size=64,
    decoder_hidden_size=64,
    learning_rate=0.0001,
    batch_size=2,
    param_output_path="",
    save_path: str = None,
    num_layers: int = 1,
    gru_lstm: bool = True,
    local_attn: bool = True,
    global_attn: bool = True
) -> Tuple[dict, GeoMan]:
    """
    n_targs: The number of target columns (not steps)
    """
    print("Using device: " + str(device))
    train_cfg = TrainConfig(
        train_data.feats.shape[0], batch_size, nn.CrossEntropyLoss()
    )
    test_cfg = TestConfig(test_data.feats.shape[0], batch_size, nn.CrossEntropyLoss())

    enc_kwargs = {
        "input_size": train_data.feats.shape[1],
        "time_length": train_data.feats.shape[-1],
        "hidden_size": encoder_hidden_size,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "gru_lstm": gru_lstm,
        "local_attn": local_attn,
        "global_attn": global_attn,
        "local_attn_dropout": local_attn_dropout,
        "global_attn_dropout": global_attn_dropout
    }
    encoder = Encoder(**enc_kwargs).to(device)
    with open(os.path.join(param_output_path, "enc_kwargs.json"), "w+") as fi:
        json.dump(enc_kwargs, fi, indent=4)

    dec_kwargs = {
        "encoder_hidden_size": encoder_hidden_size,
        "decoder_hidden_size": decoder_hidden_size,
        "temp_attn_dropout": temp_attn_dropout,
        "output_dropout": output_dropout,
        "input_size": train_data.feats.shape[1],
        "time_length": train_data.feats.shape[-1],
        "out_feats": n_targs,
        "num_layers": num_layers,
        "gru_lstm": gru_lstm,
        "local_attn": local_attn,
        "global_attn": global_attn
    }
    decoder = Decoder(**dec_kwargs).to(device)
    with open(os.path.join(param_output_path, "dec_kwargs.json"), "w+") as fi:
        json.dump(dec_kwargs, fi, indent=4)
    if save_path:
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        print("Resuming training from " + os.path.join(save_path, "encoder.pth"))
        encoder.load_state_dict(
            torch.load(os.path.join(save_path, "encoder.pth"), map_location=device)
        )
        decoder.load_state_dict(
            torch.load(os.path.join(save_path, "decoder.pth"), map_location=device)
        )

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad], lr=learning_rate
    )
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad], lr=learning_rate
    )
    geoman_net = GeoMan(encoder, decoder, encoder_optimizer, decoder_optimizer)
    return train_cfg, test_cfg, geoman_net


def train(
    net: GeoMan,
    train_data: TrainData,
    test_data: TestData,
    test_cfg: TestConfig,
    t_cfg: TrainConfig,
    n_epochs=20,
):

    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1.0 / t_cfg.batch_size))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    n_iter = 0

    for e_i in range(n_epochs):
        net.encoder.train()
        net.decoder.train()
        perm_idx = np.random.permutation(t_cfg.train_size)
        batch = 0
        print("--------------------Training-------------------------")
        print(f"Batch {batch+1} / {t_cfg.train_size // t_cfg.batch_size}")
        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            batch_idx = perm_idx[t_i : (t_i + t_cfg.batch_size)]
            feats, y_target = prep_train_data(batch_idx, train_data)
            batch += 1
            if len(feats) > 0 and len(y_target) > 0:
                if batch % ((t_cfg.train_size // t_cfg.batch_size) // 4) == 0:
                    print(f"Batch {batch} / {t_cfg.train_size // t_cfg.batch_size}")
                # print(feats.shape, y_target.shape)
                loss = train_iteration(net, t_cfg.loss_func, feats, y_target)
                iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
                n_iter += 1
                adjust_learning_rate(net, n_iter)

        epoch_losses[e_i] = np.mean(
            iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)]
        )

        if e_i % 1 == 0:
            net.encoder.eval()
            net.decoder.eval()
            print("--------------------Calculating Test Acc-------------------------")
            y_test_pred = predict(
                net,
                train_data,
                test_data,
                t_cfg.train_size,
                test_cfg.test_size,
                t_cfg.batch_size,
                on_train=False,
            )
            output = [
                int(
                    torch.argmax(
                        torch.log(
                            nn.functional.softmax(torch.tensor(y_test_pred), dim=1)
                            + 1e-9
                        )[i]
                    ).numpy()
                )
                for i in range(
                    len(
                        torch.log(
                            nn.functional.softmax(torch.tensor(y_test_pred), dim=1)
                            + 1e-9
                        )
                    )
                )
            ]
            acc = (
                sum(test_data.targs.reshape(-1) == np.array(output))
                / len(test_data.targs.reshape(-1))
                * 100
            )
            print(
                output,
                torch.tensor(test_data.targs.reshape(-1)),
            )

            print(
                torch.log(
                    nn.functional.softmax(torch.tensor(y_test_pred), dim=1) + 1e-9
                ).size(),
                torch.tensor(test_data.targs.reshape(-1)).size(),
            )
            val_loss = nn.functional.nll_loss(
                torch.log(
                    nn.functional.softmax(torch.tensor(y_test_pred), dim=1) + 1e-9
                ),
                torch.tensor(test_data.targs.reshape(-1)),
            ).numpy()

            net.encoder.eval()
            net.decoder.eval()
            print("--------------------Calculating Train Acc-------------------------")
            y_train_pred = predict(
                net,
                train_data,
                test_data,
                t_cfg.train_size,
                test_cfg.test_size,
                t_cfg.batch_size,
                on_train=True,
            )
            train_output = [
                int(
                    torch.argmax(
                        torch.log(
                            nn.functional.softmax(torch.tensor(y_train_pred), dim=1)
                            + 1e-9
                        )[i]
                    ).numpy()
                )
                for i in range(
                    len(
                        torch.log(
                            nn.functional.softmax(torch.tensor(y_train_pred), dim=1)
                            + 1e-9
                        )
                    )
                )
            ]
            train_acc = (
                sum(train_data.targs.reshape(-1) == np.array(train_output))
                / len(train_data.targs.reshape(-1))
                * 100
            )

            print(
                f"Epoch {e_i} with Train loss = {epoch_losses[e_i]}, Val loss = {val_loss}, Val acc = {acc}%, Train acc = {train_acc}%"
            )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(dir_path, "checkpoint")):
        os.makedirs(os.path.join(dir_path, "checkpoint"))
    print(os.path.join(dir_path, "checkpoint", "encoder.pth"))
    torch.save(
        net.encoder.state_dict(), os.path.join(dir_path, "checkpoint", "encoder.pth")
    )
    torch.save(
        net.decoder.state_dict(), os.path.join(dir_path, "checkpoint", "decoder.pth")
    )

    return [iter_losses, epoch_losses], net


def prep_train_data(batch_idx: np.ndarray, train_data: TrainData) -> Tuple:
    feats = np.zeros(
        (len(batch_idx), train_data.feats.shape[1], train_data.feats.shape[2])
    )
    y_target = np.zeros((len(batch_idx), 1))

    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + 1)
        feats[b_i, :, :] = train_data.feats[b_slc, :]
        y_target[b_i, :] = train_data.targs[b_slc]

    return feats, y_target


def adjust_learning_rate(net: GeoMan, n_iter: int) -> None:
    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts
    # https://www.jeremyjordan.me/nn-learning-rate/
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(
            net.enc_opt.param_groups, net.dec_opt.param_groups
        ):
            enc_params["lr"] = enc_params["lr"] * 0.9
            dec_params["lr"] = dec_params["lr"] * 0.9


def train_iteration(t_net: GeoMan, loss_func: typing.Callable, X, y_target):
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()
    _, input_encoded = t_net.encoder(numpy_to_tvar(X))
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(X))

    y_true = numpy_to_tvar(y_target)
    # print(y_true.size(), y_pred.size())
    loss = loss_func(y_pred, y_true.view(-1).long())
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss.item()


def predict(
    t_net: GeoMan,
    t_dat: TrainData,
    test: TestData,
    train_size: int,
    test_size: int,
    batch_size: int,
    on_train=False,
):
    # print(t_dat.targs.shape)
    out_size = len(np.unique(t_dat.targs))
    if on_train:
        y_pred = np.zeros((train_size, out_size))
    else:
        y_pred = np.zeros((test_size, out_size))

    n_iter = 0
    print("Batch 1")
    for y_i in range(0, len(y_pred), batch_size):
        n_iter += 1
        if on_train:
            if n_iter % ((train_size // batch_size) // 4) == 0:
                print(f"Batch {n_iter} / {train_size // batch_size}")
        else:
            if n_iter % ((test_size // batch_size) // 4) == 0:
                print(f"Batch {n_iter} / {test_size // batch_size}")
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        # print(batch_idx, b_len)
        X = np.zeros((b_len, t_dat.feats.shape[1], t_dat.feats.shape[2]))
        y_target = np.zeros((b_len, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + 1)
                X[b_i] = t_dat.feats[idx]
                y_target[b_i, :] = t_dat.targs[idx]
            else:
                idx = range(b_idx, b_idx + 1)
                # print(b_i, idx, X.shape, test.feats.shape)
                X[b_i] = test.feats[idx]
                y_target[b_i, :] = test.targs[idx]

        y_target = numpy_to_tvar(y_target)
        _, input_encoded = t_net.encoder(numpy_to_tvar(X))
        # print(input_encoded.size(), y_target.size())
        y_pred[y_slc] = (
            t_net.decoder(input_encoded, numpy_to_tvar(X)).cpu().data.numpy()
        )

    return y_pred
