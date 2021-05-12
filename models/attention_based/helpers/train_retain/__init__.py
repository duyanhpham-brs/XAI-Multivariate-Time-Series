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
from models.attention_based.RETAIN import RetainNN
from models.attention_based.helpers.train_darnn.custom_types import (
    RetainNet,
    TrainData,
    TrainConfig,
    TestConfig,
    TestData,
)
from models.attention_based.helpers.train_darnn.utils import numpy_to_tvar

# RETAIN
# Cite: Choi, E., Bahadori, M. T., Kulas, J. A., Schuetz, A., Stewart, W. F., & Sun, J. (2016).
# Retain: An interpretable predictive model for healthcare using reverse time attention mechanism.
# arXiv preprint arXiv:1608.05745.
#
# Code adapted from https://github.com/easyfan327/Pytorch-RETAIN/


def retain_nn(
    train_data: TrainData,
    test_data: TestData,
    n_targs: int,
    embedding_dim: int,
    var_rnn_hidden_size: int,
    var_rnn_output_size: int,
    visit_rnn_hidden_size: int,
    visit_rnn_output_size: int,
    visit_attn_output_size: int,
    var_attn_output_size: int,
    embedding_output_size: int,
    encoder_hidden_size=64,
    decoder_hidden_size=64,
    learning_rate=0.0001,
    batch_size=2,
    param_output_path="",
    save_path: str = None,
    gru_lstm: bool = True,
    dropout_p: float = 0.01,
    output_dropout_p: float = 0.01,
    reverse_rnn_feeding: bool = True,
) -> Tuple[dict, RetainNet]:
    """
    n_targs: The number of target columns (not steps)
    """
    print("Using device: " + str(device))
    train_cfg = TrainConfig(
        train_data.feats.shape[0], batch_size, nn.CrossEntropyLoss()
    )
    test_cfg = TestConfig(test_data.feats.shape[0], batch_size, nn.CrossEntropyLoss())

    model_kwargs = {
        "num_embeddings": train_data.feats.shape[2],
        "embedding_dim": embedding_dim,
        "dropout_p": dropout_p,
        "var_rnn_hidden_size": var_rnn_hidden_size,
        "var_rnn_output_size": var_rnn_output_size,
        "var_attn_output_size": var_attn_output_size,
        "visit_rnn_hidden_size": visit_rnn_hidden_size,
        "visit_attn_output_size": visit_attn_output_size,
        "visit_rnn_output_size": visit_rnn_output_size,
        "output_dropout_p": output_dropout_p,
        "embedding_output_size": embedding_output_size,
        "num_class": n_targs,
        "batch_size": batch_size,
        "reverse_rnn_feeding": reverse_rnn_feeding,
    }
    model = RetainNN(**model_kwargs).to(device)

    if save_path:
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        print("Resuming training from " + os.path.join(save_path, "retain.pth"))
        model.load_state_dict(
            torch.load(os.path.join(save_path, "retain.pth"), map_location=device)
        )

    model_optimizer = optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )

    retainNet = RetainNet(model, model_optimizer)

    return train_cfg, test_cfg, retainNet


def train(
    net,
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
        net.model.train()
        perm_idx = np.random.permutation(t_cfg.train_size)
        batch = 0
        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            batch_idx = perm_idx[t_i : (t_i + t_cfg.batch_size)]
            feats, y_target = prep_train_data(batch_idx, train_data)
            batch += 1
            if len(feats) > 0 and len(y_target) > 0:
                print(f"Batch {batch} / {t_cfg.train_size // t_cfg.batch_size}")
                loss = train_iteration(net, t_cfg.loss_func, feats, y_target)
                iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
                n_iter += 1
                adjust_learning_rate(net, n_iter)

        epoch_losses[e_i] = np.mean(
            iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)]
        )

        if e_i % 1 == 0:
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
    print(os.path.join(dir_path, "checkpoint", "retain.pth"))
    torch.save(
        net.model.state_dict(), os.path.join(dir_path, "checkpoint", "retain.pth")
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


def adjust_learning_rate(net: RetainNet, n_iter: int) -> None:
    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts
    # https://www.jeremyjordan.me/nn-learning-rate/
    if n_iter % 10000 == 0 and n_iter > 0:
        for model_params in net.model_opt.param_groups:
            model_params["lr"] = model_params["lr"] * 0.9


def train_iteration(t_net: RetainNet, loss_func: typing.Callable, X, y_target):
    t_net.model_opt.zero_grad()
    var_rnn_hidden_init, visit_rnn_hidden_init = t_net.model.init_hidden(X.shape[0])
    y_pred, var_rnn_hidden_init, visit_rnn_hidden_init = t_net.model(
        numpy_to_tvar(X), var_rnn_hidden_init, visit_rnn_hidden_init
    )

    y_true = numpy_to_tvar(y_target)
    loss = loss_func(y_pred, y_true.view(-1).long())
    loss.backward()

    t_net.model_opt.step()

    return loss.item()


def predict(
    t_net: RetainNet,
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
    for y_i in range(0, len(y_pred), batch_size):
        n_iter += 1
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
        # print(input_encoded.size(), y_target.size())
        var_rnn_hidden_init, visit_rnn_hidden_init = t_net.model.init_hidden(batch_size)
        y_pred[y_slc] = (
            t_net.model(numpy_to_tvar(X), var_rnn_hidden_init, visit_rnn_hidden_init)[0]
            .cpu()
            .data.numpy()
        )

    return y_pred
