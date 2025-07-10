import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    残差块的1D实现，用于构建处理序列数据的深度网络。
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # [核心修改] 使用 Conv1d 和 BatchNorm1d
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        # 如果输入输出通道不匹配，或步长不为1，则需要通过一个1x1的1D卷积来匹配维度
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # 前向传播逻辑与2D版本完全相同
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomModel(nn.Module):
    """
    自定义的1D ResNet模型，专门用于处理1D信号/序列数据。
    - 它接收的输入形状为 (批次大小, 通道数, 信号长度), e.g., (B, 1, 4096)。
    """

    def __init__(self, num_classes=3, in_channels=1):
        super(CustomModel, self).__init__()

        # [核心修改] 使用 Conv1d 和 BatchNorm1d
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        # 使用1D版本的残差块
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)

        # [核心修改] 使用1D自适应平均池化层，输出长度为1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层保持不变
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # 确保输入是3D的 (B, C, L)
        # print(f"Input shape to model: {x.shape}") # (可选的调试语句)

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)

        # 池化后形状: (B, 128, L') -> (B, 128, 1)
        out = self.avg_pool(out)

        # 展平以用于全连接层: (B, 128, 1) -> (B, 128)
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out


# 明确声明该模块对外暴露的接口
__all__ = ['CustomModel']
