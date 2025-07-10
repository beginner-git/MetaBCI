#
# models/CustomModel.py
#
# 版本: 最终版 - 针对 ShallowNet 的完全自包含与延迟初始化
# 描述: 这是一个完全独立的适配器文件。它包含了 ShallowNet 所需的全部辅助函数，
#       并采用延迟初始化策略，以适应无法在构造时提供完整维度信息的框架。
#       此版本通过包含所有依赖项来避免导入错误。
#

from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor


# ==================================================================
# =================== [ 1. 必需的辅助工具定义 ] =====================
# 我们在这里定义 ShallowNet 需要的所有辅助类，使其不再有外部依赖。
# ==================================================================

class Square(nn.Module):
    """一个计算输入的平方的模块。"""

    def __init__(self):
        super(Square, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.square(x)


class SafeLog(nn.Module):
    """一个计算安全对数的模块，避免 log(0) 出现。"""

    def __init__(self, eps=1e-6):
        super(SafeLog, self).__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=self.eps))


# ==================================================================
# ================= [ 2. 内部 ShallowNet 模型定义 ] =================
# 这是原始的 ShallowNet 实现，现在它只依赖于上面定义的内部函数。
# 我们将其重命名为 _ShallowNetInternal 以表示它是被包装的内部实现。
# ==================================================================

class _ShallowNetInternal(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        super().__init__()

        # 定义模型超参数
        n_time_filters = 40
        time_kernel = 25
        n_space_filters = 40
        pool_kernel = 75
        pool_stride = 15
        dropout_rate = 0.5

        # --- 模块1: 时间卷积 ---
        self.temporal_conv = nn.Conv2d(1, n_time_filters, (1, time_kernel), bias=True)

        # --- 模块2: 空间卷积 + BN ---
        self.spatial_conv = nn.Conv2d(n_time_filters, n_space_filters, (n_channels, 1), bias=False)
        self.batch_norm = nn.BatchNorm2d(n_space_filters)

        # --- 模块3: 平方 -> 池化 -> Log -> Dropout ---
        self.pooling_block = nn.Sequential(
            Square(),
            nn.AvgPool2d((1, pool_kernel), stride=(1, pool_stride)),
            SafeLog(),
            nn.Dropout(p=dropout_rate),
            nn.Flatten()
        )

        # --- 分类器 ---
        # 通过一个假的输入推断出展平后的特征维度
        with torch.no_grad():
            # 创建一个与真实输入形状相同的虚拟张量
            dummy_input = torch.zeros(1, 1, n_channels, n_samples)
            # 逐层传递以计算输出尺寸
            out_temporal = self.temporal_conv(dummy_input)
            out_spatial = self.batch_norm(self.spatial_conv(out_temporal))
            out_pooling = self.pooling_block(out_spatial)
            n_flattened_features = out_pooling.shape[-1]

        self.classifier = nn.Linear(n_flattened_features, n_classes, bias=True)

        self._initialize_weights()

    @torch.no_grad()
    def _initialize_weights(self):
        """自定义权重初始化。"""
        nn.init.xavier_uniform_(self.temporal_conv.weight, gain=1)
        nn.init.constant_(self.temporal_conv.bias, 0)
        nn.init.xavier_uniform_(self.spatial_conv.weight, gain=1)
        nn.init.constant_(self.batch_norm.weight, 1)
        nn.init.constant_(self.batch_norm.bias, 0)
        nn.init.xavier_uniform_(self.classifier.weight, gain=1)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # ShallowNet 需要一个 4D "图像"输入 (B, 1, C, T)
        # 输入 x 的形状为 (B, C, T)
        x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.batch_norm(x)
        x = self.pooling_block(x)
        x = self.classifier(x)

        return x


# ==================================================================
# ================== [ 3. 公开的适配器模型 ] =====================
# 这是你的框架将直接使用的最终类。
# ==================================================================

class CustomModel(nn.Module):
    """
    一个完全延迟初始化的适配器，用于将 ShallowNet 模型集成到您的框架中。
    它只在构造时接收 `num_classes`，并在第一次前向传播时构建完整模型。
    """

    def __init__(self, num_classes: int):
        """
        这个构造函数符合 TrainingApplication 的调用期望，只接收 num_classes。
        """
        super().__init__()
        self.num_classes = num_classes
        self._initialized = False
        self.model = None  # 内部模型将在稍后被构建

    def forward(self, x: Tensor) -> Tensor:
        """
        模型的前向传播逻辑。
        """
        # 检查模型是否已被初始化。如果没有，则在第一次调用时构建它。
        if not self._initialized:
            print("--- First forward pass: Lazily initializing SELF-CONTAINED ShallowNet CustomModel ---")

            # 从第一个数据批次中获取缺失的维度信息
            # x.shape is (B, C, T) -> (batch_size, n_channels, n_samples)
            in_channels = x.shape[1]
            input_length = x.shape[2]

            print(f"Detected data shape: in_channels={in_channels}, input_length={input_length}")

            # 使用所有已知信息，构建真正的内部模型
            self.model = _ShallowNetInternal(
                n_channels=in_channels,
                n_samples=input_length,
                n_classes=self.num_classes,
            )

            # 将构建好的模型移动到与输入数据相同的设备上 (如 GPU)
            self.model.to(x.device)

            # 设置标志，防止重复初始化
            self._initialized = True
            print("--- Lazy initialization complete. Model is now fully built. ---")

        # 一旦模型被构建，就直接调用它的 forward 方法
        return self.model(x)


# 明确声明该模块对外暴露的接口，确保框架能找到 'CustomModel'
__all__ = ['CustomModel']
