import collections
import typing

import numpy as np


class TrainConfig(typing.NamedTuple):
    train_size: int
    batch_size: int
    loss_func: typing.Callable


class TestConfig(typing.NamedTuple):
    test_size: int
    batch_size: int
    loss_func: typing.Callable


class TrainData(typing.NamedTuple):
    feats: np.ndarray
    targs: np.ndarray


class TestData(typing.NamedTuple):
    feats: np.ndarray
    targs: np.ndarray


DaRnnNet = collections.namedtuple(
    "DaRnnNet", ["encoder", "decoder", "enc_opt", "dec_opt"]
)

DtspRnnNet = collections.namedtuple(
    "DtspRnnNet", ["encoder", "decoder", "enc_opt", "dec_opt"]
)

RetainNet = collections.namedtuple("RetainNet", ["model", "model_opt"])

GeoMan = collections.namedtuple("GeoMan", ["encoder", "decoder", "enc_opt", "dec_opt"])

STAM = collections.namedtuple("STAM", ["encoder", "decoder", "enc_opt", "dec_opt"])