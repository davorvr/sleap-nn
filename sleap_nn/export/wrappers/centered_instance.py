"""Centered-instance ONNX wrapper."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from sleap_nn.export.wrappers.base import BaseExportWrapper


class CenteredInstanceONNXWrapper(BaseExportWrapper):
    """ONNX-exportable wrapper for centered-instance models.

    Expects input images as uint8 tensors in [0, 255].
    """

    def __init__(
        self,
        model: nn.Module,
        output_stride: int = 4,
        input_scale: float = 1.0,
        max_stride: int = 1,
        peak_threshold: float = 0.2,
    ):
        """Initialize centered instance ONNX wrapper.

        Args:
            model: Centered instance model for pose estimation.
            output_stride: Output stride for confidence maps.
            input_scale: Input scaling factor.
            max_stride: Pad scaled input dimensions to this stride.
            peak_threshold: Minimum confidence for a peak to be considered valid.
        """
        super().__init__(model)
        self.output_stride = output_stride
        self.input_scale = input_scale
        self.max_stride = max_stride
        self.peak_threshold = peak_threshold

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run centered-instance inference on crops."""
        image = self._normalize_uint8(image)
        image = self._resize_and_pad(image, self.input_scale, self.max_stride)

        confmaps = self._extract_tensor(
            self.model(image), ["centered", "instance", "confmap"]
        )
        peaks, values = self._find_global_peaks(confmaps, self.peak_threshold)
        peaks = peaks * (self.output_stride / self.input_scale)

        return {
            "peaks": peaks,
            "peak_vals": values,
        }
