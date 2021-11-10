import math
from torch import nn
import torch


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        num_channels = 3
        scale_factor = 4
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=5//2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=3//2),
            nn.Tanh(),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale_factor)
        )

        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             if m.in_channels == 32:
    #                 nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
    #                 nn.init.zeros_(m.bias.data)
    #             else:
    #                 nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
    #                 nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x

# model = MyModel()
# ii = torch.randn((2,3,51,51))
# out = model(ii)
# print(out.size())
