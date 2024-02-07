import torch

# 定义CNN模型
class CNNModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 第一层CNN
        self.cnn1 = torch.nn.Conv2d(in_channels=3,
                                    out_channels=16,
                                    kernel_size=5,
                                    stride=2,
                                    padding=0)
        # 第二层CNN
        self.cnn2 = torch.nn.Conv2d(in_channels=16,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        # 一层池化层
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三层CNN
        self.cnn3 = torch.nn.Conv2d(in_channels=32,
                                    out_channels=128,
                                    kernel_size=7,
                                    stride=1,
                                    padding=0)

        # 激活函数
        self.relu = torch.nn.ReLU()

        # 全连接层
        self.fc = torch.nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # 进入CNN1
        x = self.cnn1(x)
        x = self.relu(x)

        # CNN2
        x = self.cnn2(x)
        x = self.relu(x)

        # 池化层
        x = self.pool(x)

        # CNN3
        x = self.cnn3(x)
        x = self.relu(x)

        # 张量展平成一维
        x = x.flatten(start_dim=1)

        # 进入全连接层
        x = self.fc(x)
        return x

#
