from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor


try:
    from .base import (
        compute_same_pad2d,
        _narrow_normal_weight_zero_bias,
        compute_out_size,
    )
except ImportError:
    import math

    print("Warning: Could not perform relative import from .base. Defining dummy functions.")


    def compute_out_size(in_size, kernel_size, stride=1, padding=0, dilation=1):
        return math.floor((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


    def compute_same_pad2d(input_size, kernel_size, stride=(1, 1)):
        in_h, in_w = input_size
        k_h, k_w = kernel_size
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


class _GuneyNetInternal(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes, n_bands):
        super().__init__()

        # Define model hyperparameters
        n_spatial_filters = 120
        spatial_dropout = 0.1
        time1_kernel, time1_stride = 2, 2
        n_time1_filters = 120
        time1_dropout = 0.1
        time2_kernel = 10
        n_time2_filters = 120
        time2_dropout = 0.95

        # Calculate the temporal dimension of the intermediate layer's output
        time1_out_samples = compute_out_size(n_samples, time1_kernel, stride=time1_stride)

        # Build the model
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
        """Custom weight initialization."""
        _narrow_normal_weight_zero_bias(self)
        # GuneyNet specific initialization
        if hasattr(self.model, 'band_layer'):
            nn.init.ones_(self.model.band_layer.weight)
        if hasattr(self.model, 'fc_layer'):
            nn.init.xavier_normal_(self.model.fc_layer.weight, gain=1)

    def forward(self, x: Tensor) -> Tensor:
        # The input x here is already a 4D tensor processed by the adapter
        return self.model(x)


class CustomModel(nn.Module):
    """
    A fully lazy-initialized adapter for integrating the GuneyNet model into your framework.
    It only receives `num_classes` at construction and builds the full model during the first forward pass.
    """

    def __init__(self, num_classes: int):
        """
        This constructor meets the calling expectations of TrainingApplication, receiving only num_classes.
        """
        super().__init__()
        self.num_classes = num_classes
        self._initialized = False
        self.model = None  # The internal model will be built later

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward propagation logic of the model.
        """
        # Check if the model has been initialized. If not, build it on the first call.
        if not self._initialized:
            print("--- First forward pass: Lazily initializing GuneyNet CustomModel ---")

            # Get the missing dimension information from the first data batch
            # x.shape is (B, C, L) -> (batch_size, n_channels, n_samples)
            in_channels = x.shape[1]
            input_length = x.shape[2]

            # Key adaptation: Since the input data has no frequency band dimension, we assume n_bands = 1
            n_bands = 1

            print(f"Detected data shape: in_channels={in_channels}, input_length={input_length}")
            print(f"Adapting for single-band data: Assuming n_bands = {n_bands}")

            # Build the actual internal model using all known information
            self.model = _GuneyNetInternal(
                n_channels=in_channels,
                n_samples=input_length,
                n_classes=self.num_classes,
                n_bands=n_bands
            )

            # Move the constructed model to the same device as the input data (e.g., GPU)
            self.model.to(x.device)

            # Set the flag to prevent re-initialization
            self._initialized = True
            print("--- Lazy initialization complete. Model is now fully built. ---")

        # Key adaptation: Add a 'band' dimension to the input data to match the model's expected 4D input
        # (B, C, L) -> (B, 1, C, L)
        x_4d = x.unsqueeze(1)

        # Once the model is built, call its forward method
        return self.model(x_4d)


__all__ = ['CustomModel']