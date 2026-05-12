"""Centroid ONNX wrapper."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from sleap_nn.export.wrappers.base import BaseExportWrapper


class CentroidONNXWrapper(BaseExportWrapper):
    """ONNX-exportable wrapper for centroid models.

    Expects input images as uint8 tensors in [0, 255].
    """

    def __init__(
        self,
        model: nn.Module,
        max_instances: int = 20,
        output_stride: int = 2,
        input_scale: float = 1.0,
        max_stride: int = 1,
        peak_threshold: float = 0.2,
    ):
        """Initialize centroid ONNX wrapper.

        Args:
            model: Centroid detection model.
            max_instances: Maximum number of instances to detect.
            output_stride: Output stride for confidence maps.
            input_scale: Input scaling factor.
            max_stride: Pad scaled input dimensions to this stride.
            peak_threshold: Minimum confidence for a peak to be considered valid.
        """
        super().__init__(model)
        self.max_instances = max_instances
        self.output_stride = output_stride
        self.input_scale = input_scale
        self.max_stride = max_stride
        self.peak_threshold = peak_threshold

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run centroid inference and return fixed-size outputs."""
        image = self._normalize_uint8(image)
        image = self._resize_and_pad(image, self.input_scale, self.max_stride)

        confmaps = self._extract_tensor(self.model(image), ["centroid", "confmap"])
        peaks, values, valid = self._find_topk_peaks(
            confmaps, self.max_instances, self.peak_threshold
        )
        peaks = peaks * (self.output_stride / self.input_scale)

        return {
            "centroids": peaks,
            "centroid_vals": values,
            "instance_valid": valid,
        }
