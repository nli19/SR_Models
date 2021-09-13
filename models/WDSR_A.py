import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Block(nn.Module):
    def __init__(
        self, n_feats, kernel_size, block_feats, wn, res_scale=1, act=nn.ReLU(True)):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        body.append(
            wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # hyper-params
        scale = 4
        n_resblocks = 32
        n_feats = 128
        kernel_size = 3
        r_mean = 0.4488
        g_mean = 0.4371
        b_mean = 0.4040
        n_colors = 3
        block_feats = 512
        res_scale = 1

        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [r_mean, g_mean, b_mean])).view([1, 3, 1, 1])

        # define head module
        head = []
        head.append(
            wn(nn.Conv2d(n_colors, n_feats, 3, padding=3//2)))

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                Block(n_feats, kernel_size, block_feats, wn=wn, res_scale=res_scale, act=act))

        # define tail module
        tail = []
        out_feats = scale*scale*n_colors
        tail.append(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2)))
        tail.append(nn.PixelShuffle(scale))

        skip = []
        skip.append(
            wn(nn.Conv2d(n_colors, out_feats, 5, padding=5//2))
        )
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = (x - self.rgb_mean.cuda()*255)/127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x = x*127.5 + self.rgb_mean.cuda()*255
        return x

# model = MyModel()
# ii = torch.randn((2,3,51,51))
# out = model(ii)
# print(out.size())