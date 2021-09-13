import torch.nn as nn
import torch
import common

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3 
        scale = 4
        n_colors = 3
        rgb_range = 255
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [common.conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                common.conv, n_feats, kernel_size, act=act, res_scale=scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(common.conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(common.conv, scale, n_feats, act=False),
            common.conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

# model = MyModel()
# ii = torch.randn((2,3,51,51))
# out = model(ii)
# print(out.size())
    