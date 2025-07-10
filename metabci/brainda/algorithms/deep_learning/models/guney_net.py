#
# models/CustomModel.py
#
# 版本: 针对 GuneyNet 的完全延迟初始化 (Full Lazy Initialization)
# 描述: 这是一个适配器文件，用于将 GuneyNet 集成到一个无法在 __init__
#       阶段提供完整维度信息的框架中。它假定输入数据为单频带 (n_bands=1)
#       并从 .base 模块导入辅助函数。
#

from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor

# ==================================================================
# ===================== [ 1. 外部依赖项导入 ] ======================
# 根据您的指示，我们从外部模块导入所需的辅助函数，而不再重新定义它们。
# 这要求您的项目结构中存在一个可解析的 'base' 模块。
# ==================================================================
try:
    from .base import (
        compute_same_pad2d,
        _narrow_normal_weight_zero_bias,
        compute_out_size,
    )
except ImportError:
    # 如果直接运行此文件进行测试，上面的相对导入会失败。
    # 这里提供一个备用方案，但这主要用于说明和隔离测试。
    # 在您的框架中，应该使用上面的 from .base import
    import math

    print("Warning: Could not perform relative import from .base. Defining dummy functions.")


    def compute_out_size(in_size, kernel_size, stride=1, padding=0, dilation=1):
        return math.floor((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


    def compute_same_pad2d(input_size, kernel_size, stride=(1, 1)):
        in_h, in_w = input_size;
        k_h, k_w = kernel_size;
        s_h, s_w = stride
        pad_h = max(0, (math.ceil(in_h / s_h) - 1) * s_h + k_h - in_h)
        pad_w = max(0, (math.ceil(in_w / s_w) - 1) * s_w + k_w - in_w)
        return (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)


    @torch.no_grad()
    def _narrow_normal_weight_zero_bias(model, std=0.01):
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0, std=std)
                if m.bias is not None: nn.init.constant_(m.bias, 0)


# ==================================================================
# ================= [ 2. 内部 GuneyNet 模型定义 ] ==================
# 这是原始的 GuneyNet 实现，重命名为 _GuneyNetInternal 以表示它是
# 被包装的内部实现。
# ==================================================================

class _GuneyNetInternal(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes, n_bands):
        super().__init__()

        # 定义模型超参数
        n_spatial_filters = 120
        spatial_dropout = 0.1
        time1_kernel, time1_stride = 2, 2
        n_time1_filters = 120
        time1_dropout = 0.1
        time2_kernel = 10
        n_time2_filters = 120
        time2_dropout = 0.95

        # 计算中间层输出的时间维度
        time1_out_samples = compute_out_size(n_samples, time1_kernel, stride=time1_stride)

        # 构建模型
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("band_layer", nn.Conv2d(n_bands, 1, (1, 1), bias=False)),
                    ("spatial_layer", nn.Conv2d(1, n_spatial_filters, (n_channels, 1))),
                    ("spatial_dropout", nn.Dropout(spatial_dropout)),
                    ("time1_layer",
                     nn.Conv2d(n_spatial_filters, n_time1_filters, (1, time1_kernel), stride=(1, time1_stride))),
                    ("time1_dropout", nn.Dropout(time1_dropout)),
                    ("relu", nn.ReLU()),
                    (
                    "same_padding", nn.ConstantPad2d(compute_same_pad2d((1, time1_out_samples), (1, time2_kernel)), 0)),
                    ("time2_layer", nn.Conv2d(n_time1_filters, n_time2_filters, (1, time2_kernel), stride=(1, 1))),
                    ("time2_dropout", nn.Dropout(time2_dropout)),
                    ("flatten", nn.Flatten()),
                    ("fc_layer", nn.Linear(n_time2_filters * time1_out_samples, n_classes)),
                ]
            )
        )
        self._initialize_weights()

    @torch.no_grad()
    def _initialize_weights(self):
        """自定义权重初始化。"""
        _narrow_normal_weight_zero_bias(self)
        # GuneyNet 的特定初始化
        if hasattr(self.model, 'band_layer'):
            nn.init.ones_(self.model.band_layer.weight)
        if hasattr(self.model, 'fc_layer'):
            nn.init.xavier_normal_(self.model.fc_layer.weight, gain=1)

    def forward(self, x: Tensor) -> Tensor:
        # 这里的输入 x 已经是被适配器处理过的 4D 张量
        return self.model(x)


# ==================================================================
# ================== [ 3. 公开的适配器模型 ] =====================
# 这是你的框架将直接使用的类。它的接口符合框架的调用约定。
# ==================================================================

class CustomModel(nn.Module):
    """
    一个完全延迟初始化的适配器，用于将 GuneyNet 模型集成到您的框架中。
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
            print("--- First forward pass: Lazily initializing GuneyNet CustomModel ---")

            # 从第一个数据批次中获取缺失的维度信息
            # x.shape is (B, C, L) -> (batch_size, n_channels, n_samples)
            in_channels = x.shape[1]
            input_length = x.shape[2]

            # 关键适配：由于输入数据没有频带维度，我们假定 n_bands = 1
            n_bands = 1

            print(f"Detected data shape: in_channels={in_channels}, input_length={input_length}")
            print(f"Adapting for single-band data: Assuming n_bands = {n_bands}")

            # 使用所有已知信息，构建真正的内部模型
            self.model = _GuneyNetInternal(
                n_channels=in_channels,
                n_samples=input_length,
                n_classes=self.num_classes,
                n_bands=n_bands
            )

            # 将构建好的模型移动到与输入数据相同的设备上 (如 GPU)
            self.model.to(x.device)

            # 设置标志，防止重复初始化
            self._initialized = True
            print("--- Lazy initialization complete. Model is now fully built. ---")

        # 关键适配：为输入数据添加一个'band'维度，以匹配模型期望的4D输入
        # (B, C, L) -> (B, 1, C, L)
        x_4d = x.unsqueeze(1)

        # 一旦模型被构建，就调用它的 forward 方法
        return self.model(x_4d)


# 明确声明该模块对外暴露的接口，确保框架能找到 'CustomModel'
__all__ = ['CustomModel']
