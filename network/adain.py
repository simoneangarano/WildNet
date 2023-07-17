import torch
import torch.nn as nn


class AdaptiveInstanceNormalization(nn.Module):

    def __init__(self, out_channels=None):
        super(AdaptiveInstanceNormalization, self).__init__()

    def forward(self, x_cont, x_style=None, ):
        if x_style is not None:
            assert (x_cont.size()[:2] == x_style.size()[:2])
            size = x_cont.size()
            style_mean, style_std = calc_mean_std(x_style)
            content_mean, content_std = calc_mean_std(x_cont)

            normalized_x_cont = (x_cont - content_mean.expand(size))/content_std.expand(size)
            denormalized_x_cont = normalized_x_cont * style_std.expand(size) + style_mean.expand(size)

            return denormalized_x_cont

        else:
            return x_cont


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = torch.var(feat.clone().view(N, C, -1), dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = torch.mean(feat.clone().view(N, C, -1),dim=2)
    return feat_mean.view(N, C, 1, 1), feat_std

