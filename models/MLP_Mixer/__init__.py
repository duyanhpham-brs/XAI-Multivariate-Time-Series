from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout),
    )


def MLPMixer(
    *,
    image_size,
    channels,
    patch_size,
    dim,
    depth,
    num_classes,
    expansion_factor=4,
    dropout=0.0
):
    """The implementation of MLPMixer

    Based on the paper:

        Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., ... & Dosovitskiy, A.
        (2021). Mlp-mixer: An all-mlp architecture for vision. arXiv preprint arXiv:2105.01601.

    This implementation is modified to only support Multivariate Time Series
    Classification data from the original implementation as follows:

        @misc{tolstikhin2021mlpmixer,
            title   = {MLP-Mixer: An all-MLP Architecture for Vision},
            author  = {Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
            year    = {2021},
            eprint  = {2105.01601},
            archivePrefix = {arXiv},
            primaryClass = {cs.CV}
        }

    """
    assert (image_size % patch_size) == 0, "image must be divisible by patch size"
    num_patches = image_size // patch_size
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        Rearrange("b c (h p) w -> b h (p w c)", p=patch_size),
        nn.Linear((patch_size) * channels, dim),
        *[
            nn.Sequential(
                PreNormResidual(
                    dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)
                ),
                PreNormResidual(
                    dim, FeedForward(dim, expansion_factor, dropout, chan_last)
                ),
            )
            for _ in range(depth)
        ],
        nn.LayerNorm(dim),
        Reduce("b n c -> b c", "mean"),
        nn.Linear(dim, num_classes)
    )