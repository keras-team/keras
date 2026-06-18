import numpy as np

from keras.src import testing
from keras.src.visualization.draw_segmentation_masks import (
    draw_segmentation_masks,
)


class DrawSegmentationMasksTest(testing.TestCase):
    def test_zero_based_color_mapping(self):
        # Regression for the docs/impl mismatch reported in
        # https://github.com/keras-team/keras/issues/23088 — class indices
        # are zero-based (`[0, num_classes - 1]`) and `color_mapping` keys
        # must match that range.
        images = np.zeros((1, 2, 2, 3), dtype="uint8")
        masks = np.array([[[1, 2], [0, 2]]], dtype="int32")[..., None]
        out = draw_segmentation_masks(
            images,
            masks,
            num_classes=3,
            color_mapping={0: [10, 10, 10], 1: [255, 0, 0], 2: [0, 255, 0]},
            blend=False,
            data_format="channels_last",
        )
        self.assertEqual(out.shape, (1, 2, 2, 3))

    def test_num_classes_inference_covers_max_label(self):
        # `num_classes=None` must yield a count that covers the largest
        # observed label, otherwise `ops.one_hot` would silently drop it
        # (previously raised `IndexError` because `num_classes` was set to
        # `max(masks)` instead of `max(masks) + 1`).
        images = np.zeros((1, 2, 2, 3), dtype="uint8")
        masks = np.array([[[1, 2], [0, 2]]], dtype="int32")[..., None]
        out = draw_segmentation_masks(
            images,
            masks,
            num_classes=None,
            blend=False,
            data_format="channels_last",
        )
        self.assertEqual(out.shape, (1, 2, 2, 3))
