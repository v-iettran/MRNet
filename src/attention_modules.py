# """CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018).

# Owner: Caolan

# Provides channel + spatial attention blocks that get inserted into the
# convolutional stages of a backbone (see Project Pipeline.md, section 3.2).
# This is the "spatial attention" advanced technique required by the rubric.

# NOTE: unimplemented scaffold. Subclass ``torch.nn.Module`` where indicated.
# """
# from __future__ import annotations


# # TODO(Caolan): subclass torch.nn.Module.
# class ChannelAttention:
#     """Channel-attention sub-module of CBAM.

#     Suggested args:
#         channels: number of input channels.
#         reduction: channel-reduction ratio for the shared MLP.
#     """

#     def __init__(self, channels, reduction=16):
#         raise NotImplementedError("TODO(Caolan): implement ChannelAttention.__init__")

#     def forward(self, x):
#         raise NotImplementedError("TODO(Caolan): implement ChannelAttention.forward")


# # TODO(Caolan): subclass torch.nn.Module.
# class SpatialAttention:
#     """Spatial-attention sub-module of CBAM.

#     Suggested args:
#         kernel_size: conv kernel size for the spatial map (commonly 7).
#     """

#     def __init__(self, kernel_size=7):
#         raise NotImplementedError("TODO(Caolan): implement SpatialAttention.__init__")

#     def forward(self, x):
#         raise NotImplementedError("TODO(Caolan): implement SpatialAttention.forward")


# # TODO(Caolan): subclass torch.nn.Module.
# class CBAM:
#     """Full CBAM block = ChannelAttention followed by SpatialAttention.

#     Suggested args:
#         channels: number of input channels.
#         reduction: channel-reduction ratio.
#         kernel_size: spatial-attention kernel size.

#     ``forward`` applies channel then spatial attention and returns the
#     refined feature map (same shape as input).
#     """

#     def __init__(self, channels, reduction=16, kernel_size=7):
#         raise NotImplementedError("TODO(Caolan): implement CBAM.__init__")

#     def forward(self, x):
#         raise NotImplementedError("TODO(Caolan): implement CBAM.forward")



"""CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018).

Owner: Caolan

Provides channel + spatial attention blocks that get inserted into the
convolutional stages of a backbone (see Project Pipeline.md, section 3.2).
This is the "spatial attention" advanced technique required by the rubric.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """Channel-attention sub-module of CBAM.

    Suggested args:
        channels: number of input channels.
        reduction: channel-reduction ratio for the shared MLP.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)  # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial-attention sub-module of CBAM.

    Suggested args:
        kernel_size: conv kernel size for the spatial map (commonly 7).
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))  # (B, 1, H, W)


class CBAM(nn.Module):
    """Full CBAM block = ChannelAttention followed by SpatialAttention.

    Suggested args:
        channels: number of input channels.
        reduction: channel-reduction ratio.
        kernel_size: spatial-attention kernel size.

    ``forward`` applies channel then spatial attention and returns the
    refined feature map (same shape as input).
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x