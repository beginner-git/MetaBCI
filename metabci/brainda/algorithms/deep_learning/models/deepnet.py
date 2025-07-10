import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.functional import elu


def identity(x):
    """A function that does nothing and returns the value directly."""
    return x


def transpose_time_to_spat(x):
    """Dimension conversion, (B, C, L, 1) -> (B, 1, L, C)"""
    return x.permute(0, 3, 2, 1)


def squeeze_final_output(x):
    """Removes the last two dimensions if their size is 1."""
    assert x.shape[2] == 1 and x.shape[3] == 1, "dimensions must be 1x1 to squeeze"
    return x[:, :, 0, 0]


class Expression(nn.Module):
    """Wraps any function into an nn.Module."""

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)


class Ensure4d(nn.Module):
    """Ensures the input tensor is 4D. If the input is 3D, it adds a dimension at the end."""

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(3) if len(x.shape) == 3 else x


AvgPool2dWithConv = nn.AvgPool2d


class _Deep4NetInternal(nn.Sequential):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int, **kwargs):
        super().__init__()
        # Model hyperparameters
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

        # Build network layers
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

        # Helper function to add a convolution-pooling block
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

        # Dynamically calculate the kernel size of the final classifier
        self.eval()
        with torch.no_grad():
            dummy_input = torch.ones((1, n_channels, n_samples), dtype=torch.float32)
            out_shape = self(dummy_input).shape
            final_conv_length = out_shape[2]
        self.train()

        # Add classifier layer
        self.add_module("conv_classifier", nn.Conv2d(n_filters_4, n_classes, (final_conv_length, 1), bias=True))
        self.add_module("softmax", nn.LogSoftmax(dim=1))
        self.add_module("squeeze", Expression(squeeze_final_output))

        self._initialize_weights()

    def _initialize_weights(self):
        # Weight initialization logic
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)


class CustomModel(nn.Module):
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
            print("--- First forward pass: Lazily initializing CustomModel ---")

            # Get the missing dimension information from the first data batch
            in_channels = x.shape[1]
            input_length = x.shape[2]

            print(f"Detected data shape: in_channels={in_channels}, input_length={input_length}")

            # Build the actual internal model using all known information
            self.model = _Deep4NetInternal(
                n_channels=in_channels,
                n_samples=input_length,
                n_classes=self.num_classes,
            )

            # Move the constructed model to the same device as the input data (e.g., GPU)
            self.model.to(x.device)

            # Set the flag to prevent re-initialization
            self._initialized = True
            print("--- Lazy initialization complete. Model is now fully built. ---")

        # Once the model is built, call its forward method directly
        return self.model(x)


__all__ = ['CustomModel']