from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor



class Square(nn.Module):
    """A module that computes the square of the input."""

    def __init__(self):
        super(Square, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.square(x)


class SafeLog(nn.Module):
    """A module that computes a safe logarithm, avoiding log(0)."""

    def __init__(self, eps=1e-6):
        super(SafeLog, self).__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=self.eps))



class _ShallowNetInternal(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        super().__init__()

        # Define model hyperparameters
        n_time_filters = 40
        time_kernel = 25
        n_space_filters = 40
        pool_kernel = 75
        pool_stride = 15
        dropout_rate = 0.5

        # --- Module 1: Temporal Convolution ---
        self.temporal_conv = nn.Conv2d(1, n_time_filters, (1, time_kernel), bias=True)

        # --- Module 2: Spatial Convolution + BN ---
        self.spatial_conv = nn.Conv2d(n_time_filters, n_space_filters, (n_channels, 1), bias=False)
        self.batch_norm = nn.BatchNorm2d(n_space_filters)

        # --- Module 3: Square -> Pooling -> Log -> Dropout ---
        self.pooling_block = nn.Sequential(
            Square(),
            nn.AvgPool2d((1, pool_kernel), stride=(1, pool_stride)),
            SafeLog(),
            nn.Dropout(p=dropout_rate),
            nn.Flatten()
        )

        # --- Classifier ---
        # Infer the flattened feature dimension using a dummy input
        with torch.no_grad():
            # Create a dummy tensor with the same shape as the real input
            dummy_input = torch.zeros(1, 1, n_channels, n_samples)
            # Pass it through the layers to calculate the output size
            out_temporal = self.temporal_conv(dummy_input)
            out_spatial = self.batch_norm(self.spatial_conv(out_temporal))
            out_pooling = self.pooling_block(out_spatial)
            n_flattened_features = out_pooling.shape[-1]

        self.classifier = nn.Linear(n_flattened_features, n_classes, bias=True)

        self._initialize_weights()

    @torch.no_grad()
    def _initialize_weights(self):
        """Custom weight initialization."""
        nn.init.xavier_uniform_(self.temporal_conv.weight, gain=1)
        nn.init.constant_(self.temporal_conv.bias, 0)
        nn.init.xavier_uniform_(self.spatial_conv.weight, gain=1)
        nn.init.constant_(self.batch_norm.weight, 1)
        nn.init.constant_(self.batch_norm.bias, 0)
        nn.init.xavier_uniform_(self.classifier.weight, gain=1)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # ShallowNet requires a 4D "image" input (B, 1, C, T)
        # The shape of the input x is (B, C, T)
        x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.batch_norm(x)
        x = self.pooling_block(x)
        x = self.classifier(x)

        return x



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
            print("--- First forward pass: Lazily initializing SELF-CONTAINED ShallowNet CustomModel ---")

            # Get the missing dimension information from the first data batch
            # x.shape is (B, C, T) -> (batch_size, n_channels, n_samples)
            in_channels = x.shape[1]
            input_length = x.shape[2]

            print(f"Detected data shape: in_channels={in_channels}, input_length={input_length}")

            # Build the actual internal model using all known information
            self.model = _ShallowNetInternal(
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