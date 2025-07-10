#
# models/CustomModel.py
#
# 版本: 完全延迟初始化 (Full Lazy Initialization)
# 描述: 这是一个自包含的适配器文件，用于将 Deep4Net 集成到一个
#       无法在 __init__ 阶段提供 in_channels 的框架中。
#

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.functional import elu


# ==================================================================
# =================== [ 1. 辅助工具定义 ] ========================
# (这部分定义了 Deep4Net 需要的所有工具，使其自包含)
# ==================================================================

def identity(x):
    """一个什么都不做的函数，直接返回值。"""
    return x


def transpose_time_to_spat(x):
    """维度转换，(B, C, L, 1) -> (B, 1, L, C)"""
    return x.permute(0, 3, 2, 1)


def squeeze_final_output(x):
    """移除最后两个维度，如果它们的大小是1。"""
    assert x.shape[2] == 1 and x.shape[3] == 1, " dimensions must be 1x1 to squeeze"
    return x[:, :, 0, 0]


class Expression(nn.Module):
    """将任何函数包装成一个 nn.Module。"""

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)


class Ensure4d(nn.Module):
    """确保输入张量是4D的。如果输入是3D，则在最后增加一个维度。"""

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(3) if len(x.shape) == 3 else x


AvgPool2dWithConv = nn.AvgPool2d


# ==================================================================
# =============== [ 2. 内部 Deep4Net 模型实现 ] ====================
# 这是 Deep4Net 的核心实现，它需要在构造时知道所有维度信息。
# 它将被我们的适配器在合适的时机调用。
# ==================================================================

class _Deep4NetInternal(nn.Sequential):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int, **kwargs):
        super().__init__()
        # 模型超参数
        self.batch_norm = kwargs.get('batch_norm', True)
        n_filters_time = kwargs.get('n_filters_time', 25)
        n_filters_spat = kwargs.get('n_filters_spat', 25)
        filter_time_length = kwargs.get('filter_time_length', 10)
        pool_time_length = kwargs.get('pool_time_length', 3)
        pool_time_stride = kwargs.get('pool_time_stride', 3)
        n_filters_2, n_filters_3, n_filters_4 = kwargs.get('n_filters_2', 50), kwargs.get('n_filters_3',
                                                                                          100), kwargs.get(
            'n_filters_4', 200)
        filter_length_2, filter_length_3, filter_length_4 = kwargs.get('filter_length_2', 10), kwargs.get(
            'filter_length_3', 10), kwargs.get('filter_length_4', 10)
        drop_prob = kwargs.get('drop_prob', 0.5)
        batch_norm_alpha = kwargs.get('batch_norm_alpha', 0.1)

        # 构建网络层
        self.add_module("ensuredims", Ensure4d())
        self.add_module("dimshuffle", Expression(transpose_time_to_spat))
        self.add_module("conv_time", nn.Conv2d(1, n_filters_time, (filter_time_length, 1), stride=1))
        self.add_module("conv_spat", nn.Conv2d(n_filters_time, n_filters_spat, (1, n_channels), stride=(1, 1),
                                               bias=not self.batch_norm))
        if self.batch_norm: self.add_module("bnorm",
                                            nn.BatchNorm2d(n_filters_spat, momentum=batch_norm_alpha, affine=True,
                                                           eps=1e-5))
        self.add_module("conv_nonlin", Expression(elu))
        self.add_module("pool", nn.MaxPool2d(kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1)))

        # 辅助函数，用于添加卷积池化块
        def add_conv_pool_block(module_seq, n_in, n_out, f_len, b_nr):
            module_seq.add_module(f"drop_{b_nr}", nn.Dropout(p=drop_prob))
            module_seq.add_module(f"conv_{b_nr}",
                                  nn.Conv2d(n_in, n_out, (f_len, 1), stride=(1, 1), bias=not self.batch_norm))
            if self.batch_norm: module_seq.add_module(f"bnorm_{b_nr}",
                                                      nn.BatchNorm2d(n_out, momentum=batch_norm_alpha, affine=True,
                                                                     eps=1e-5))
            module_seq.add_module(f"nonlin_{b_nr}", Expression(elu))
            module_seq.add_module(f"pool_{b_nr}",
                                  nn.MaxPool2d(kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1)))

        add_conv_pool_block(self, n_filters_spat, n_filters_2, filter_length_2, 2)
        add_conv_pool_block(self, n_filters_2, n_filters_3, filter_length_3, 3)
        add_conv_pool_block(self, n_filters_3, n_filters_4, filter_length_4, 4)

        # 动态计算最终分类器的卷积核大小
        self.eval()
        with torch.no_grad():
            dummy_input = torch.ones((1, n_channels, n_samples), dtype=torch.float32)
            out_shape = self(dummy_input).shape
            final_conv_length = out_shape[2]
        self.train()

        # 添加分类器层
        self.add_module("conv_classifier", nn.Conv2d(n_filters_4, n_classes, (final_conv_length, 1), bias=True))
        self.add_module("softmax", nn.LogSoftmax(dim=1))
        self.add_module("squeeze", Expression(squeeze_final_output))

        self._initialize_weights()

    def _initialize_weights(self):
        # 权重初始化逻辑
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)


# ==================================================================
# ================== [ 3. 公开的适配器模型 ] =====================
# 这是你的框架将直接使用的类。它的接口符合框架的调用约定。
# ==================================================================

class CustomModel(nn.Module):
    """
    一个完全延迟初始化的适配器，用于将 Deep4Net 模型集成到您的框架中。
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
            print("--- First forward pass: Lazily initializing CustomModel ---")

            # 从第一个数据批次中获取缺失的维度信息
            in_channels = x.shape[1]
            input_length = x.shape[2]

            print(f"Detected data shape: in_channels={in_channels}, input_length={input_length}")

            # 使用所有已知信息，构建真正的内部模型
            self.model = _Deep4NetInternal(
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

