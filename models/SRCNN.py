from torch import nn
import torch



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        num_channels = 3
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.dtype)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# model = MyModel()
# ii = torch.randn((2,3,51,51))
# out = model(ii)
# print(out.size())