from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch import Tensor


def compute_same_pad2d(input_size, kernel_size, stride=(1, 1)):
    """
    Calculates the padding amount required to implement 'same' padding in PyTorch.
    Returns a tuple suitable for nn.ConstantPad2d.
    """
    in_h, in_w = input_size
    k_h, k_w = kernel_size
    s_h, s_w = stride

    # Use ceiling to ensure the output size is not smaller than the size in 'same' mode
    pad_h = max(0, (math.ceil(in_h / s_h) - 1) * s_h + k_h - in_h)
    pad_w = max(0, (math.ceil(in_w / s_w) - 1) * s_w + k_w - in_w)

    # PyTorch's ConstantPad2d requires (pad_left, pad_right, pad_top, pad_bottom)
    return (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)


@torch.no_grad()
def _apply_max_norm(module, max_norm_value=1.0, name='weight', eps=1e-8):
    """A general function for applying a max-norm constraint to a weight tensor."""
    weight = getattr(module, name)
    norm = torch.norm(weight, p=2, dim=0, keepdim=True)
    desired = torch.clamp(norm, 0, max_norm_value)
    # Scale the weights proportionally
    weight.data = weight.data * (desired / (eps + norm))


class MaxNormConstraintConv2d(nn.Conv2d):
    """A 2D convolutional layer with a max-norm constraint."""

    def __init__(self, *args, max_norm_value=1.0, **kwargs):
        self.max_norm_value = max_norm_value
        super(MaxNormConstraintConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        _apply_max_norm(self, self.max_norm_value)
        return super(MaxNormConstraintConv2d, self).forward(x)


class MaxNormConstraintLinear(nn.Linear):
    """A linear layer with a max-norm constraint."""

    def __init__(self, *args, max_norm_value=0.25, **kwargs):
        self.max_norm_value = max_norm_value
        super(MaxNormConstraintLinear, self).__init__(*args, **kwargs)

    def forward(self, x):
        _apply_max_norm(self, self.max_norm_value)
        return super(MaxNormConstraintLinear, self).forward(x)


@torch.no_grad()
def _glorot_weight_zero_bias(model):
    """
    Initializes weights with Xavier/Glorot uniform distribution and sets biases to zero.
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.xavier_uniform_(module.weight, gain=1)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)


class SeparableConv2d(nn.Module):
    """
    Separable convolution, composed of a depthwise convolution and a pointwise convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, D=2):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels * D, kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels * D, out_channels, 1, stride=1, padding=0, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ==================================================================
# ================= [ 2. Internal EEGNet Model Definition ] ===================
# This is the original EEGNet implementation, which now only depends on the
# internal functions defined above.
# ==================================================================

class _EEGNetInternal(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes):
        super().__init__()
        # Define model hyperparameters
        F1, D = 8, 2
        F2 = F1 * D
        time_kernel_size = 64
        separa_kernel_size = 16
        pool_size1, pool_stride1 = (1, 4), (1, 4)
        pool_size2, pool_stride2 = (1, 8), (1, 8)
        dropout_rate = 0.5
        fc_norm_rate = 0.25
        depthwise_norm_rate = 1.0

        # --- First block: Temporal convolution + Depthwise convolution ---
        self.block1 = nn.Sequential(
            nn.ConstantPad2d(compute_same_pad2d((n_channels, n_samples), (1, time_kernel_size)), 0),
            nn.Conv2d(1, F1, (1, time_kernel_size), bias=False),
            nn.BatchNorm2d(F1),
            MaxNormConstraintConv2d(F1, F2, (n_channels, 1), groups=F1, bias=False, max_norm_value=depthwise_norm_rate),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(pool_size1, stride=pool_stride1),
            nn.Dropout(dropout_rate)
        )

        # --- Second block: Separable convolution ---
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_channels, n_samples)
            out_block1 = self.block1(dummy_input)
            n_out_block1_samps = out_block1.shape[3]

        self.block2 = nn.Sequential(
            nn.ConstantPad2d(compute_same_pad2d((1, n_out_block1_samps), (1, separa_kernel_size)), 0),
            SeparableConv2d(F2, F2, (1, separa_kernel_size), padding=0, bias=False, D=1),
            # D=1 here as per original Keras implementation of SeparableConv
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(pool_size2, stride=pool_stride2),
            nn.Dropout(dropout_rate),
            nn.Flatten()
        )

        # --- Classifier ---
        with torch.no_grad():
            dummy_out_block2 = self.block2(out_block1)
            n_flattened_features = dummy_out_block2.shape[1]

        self.classifier = MaxNormConstraintLinear(
            n_flattened_features, n_classes, max_norm_value=fc_norm_rate
        )

        self.model = nn.Sequential(self.block1, self.block2, self.classifier)
        _glorot_weight_zero_bias(self)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        return self.model(x)


class CustomModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self._initialized = False
        self.model = None

    def forward(self, x: Tensor) -> Tensor:
        if not self._initialized:
            print("--- First forward pass: Lazily initializing SELF-CONTAINED EEGNet CustomModel ---")
            in_channels = x.shape[1]
            input_length = x.shape[2]
            print(f"Detected data shape: in_channels={in_channels}, input_length={input_length}")

            self.model = _EEGNetInternal(
                n_channels=in_channels,
                n_samples=input_length,
                n_classes=self.num_classes,
            )
            self.model.to(x.device)
            self._initialized = True
            print("--- Lazy initialization complete. Model is now fully built. ---")

        return self.model(x)


__all__ = ['CustomModel']