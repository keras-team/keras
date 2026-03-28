"""Comprehensive tests for the keras.src.visualization module.

Tests cover:
- draw_bounding_boxes: pixel-level validation, edge cases, error conditions
- draw_segmentation_masks: alpha blending math, class color, error conditions
- plot_image_gallery: file I/O, error conditions, label handling
- plot_bounding_box_gallery: composition, legend conflict, channels_first
- plot_segmentation_mask_gallery: composition, mask overlay accuracy
"""

import os
import tempfile

import numpy as np
import pytest

from keras.src.testing import test_case

# ─────────────────────────────────── helpers ────────────────────────────────


def _solid_images(batch=2, h=32, w=32, c=3, value=100, dtype="uint8"):
    """Return a batch of solid-colour images."""
    return np.full((batch, h, w, c), value, dtype=dtype)


def _solid_images_float(batch=2, h=32, w=32, c=3, value=0.5):
    return np.full((batch, h, w, c), value, dtype=np.float32)


def _bbox_dict(boxes, labels, confidences=None):
    d = {
        "boxes": np.array(boxes, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int32),
    }
    if confidences is not None:
        d["confidences"] = np.array(confidences, dtype=np.float32)
    return d


# ═══════════════════════════════════════════════════════════════════════════
#  draw_bounding_boxes
# ═══════════════════════════════════════════════════════════════════════════


class DrawBoundingBoxesTest(test_case.TestCase):
    def setUp(self):
        pytest.importorskip("cv2")
        from keras.src.visualization.draw_bounding_boxes import (
            draw_bounding_boxes,
        )

        self.draw_bounding_boxes = draw_bounding_boxes

    # ── validation ──────────────────────────────────────────────────────────

    def test_raises_value_error_for_3d_input(self):
        images_3d = np.zeros((32, 32, 3), dtype="uint8")
        bboxes = _bbox_dict([[[2, 2, 10, 10]]], [[0]])
        with self.assertRaises(ValueError):
            self.draw_bounding_boxes(images_3d, bboxes, "xyxy")

    def test_raises_type_error_for_non_dict_bboxes(self):
        images = _solid_images()
        with self.assertRaises(TypeError):
            self.draw_bounding_boxes(images, [[2, 2, 10, 10]], "xyxy")

    def test_raises_value_error_for_missing_keys(self):
        images = _solid_images()
        with self.assertRaises(ValueError):
            self.draw_bounding_boxes(
                images, {"boxes": np.zeros((2, 1, 4))}, "xyxy"
            )

    def test_raises_value_error_for_missing_labels_key(self):
        images = _solid_images()
        with self.assertRaises(ValueError):
            self.draw_bounding_boxes(
                images, {"labels": np.zeros((2, 1))}, "xyxy"
            )

    # ── output shape / type ─────────────────────────────────────────────────

    def test_returns_ndarray_same_shape(self):
        images = _solid_images(batch=2, h=40, w=40, c=3)
        bboxes = _bbox_dict([[[2, 2, 20, 20]], [[5, 5, 15, 15]]], [[1], [1]])
        result = self.draw_bounding_boxes(images, bboxes, "xyxy")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, images.shape)
        self.assertEqual(result.dtype, np.uint8)

    # ── actual pixels change at box boundary ────────────────────────────────

    def test_pixels_change_at_box_boundary(self):
        h, w = 64, 64
        images = _solid_images(batch=1, h=h, w=w, c=3, value=0)
        color = (255, 0, 0)  # red in RGB
        x1, y1, x2, y2 = 10, 10, 40, 40
        bboxes = _bbox_dict([[[x1, y1, x2, y2]]], [[0]])
        result = self.draw_bounding_boxes(
            images,
            bboxes,
            "xyxy",
            color=color,
            line_thickness=1,
        )
        # OpenCV draws BGR; draw_bounding_boxes passes RGB directly to cv2,
        # inspect channel 0 (R-component when cv2 treats as BGR, but here
        # the function passes color directly so channel index 0 = R).
        # At least one of the top-edge pixels must be non-zero.
        top_edge = result[0, y1, x1:x2, :]
        self.assertTrue(np.any(top_edge != 0))

    def test_label_minus_one_not_drawn(self):
        images = _solid_images(batch=1, h=64, w=64, c=3, value=50)
        x1, y1, x2, y2 = 5, 5, 30, 30
        bboxes = _bbox_dict([[[x1, y1, x2, y2]]], [[-1]])
        result = self.draw_bounding_boxes(
            images, bboxes, "xyxy", color=(255, 0, 0), line_thickness=2
        )
        # Output should be all 50 – nothing was drawn
        np.testing.assert_array_equal(result[0], images[0])

    def test_multiple_boxes_in_one_image(self):
        images = _solid_images(batch=1, h=64, w=64, c=3, value=0)
        bboxes = _bbox_dict([[[2, 2, 15, 15], [30, 30, 50, 50]]], [[0, 1]])
        result = self.draw_bounding_boxes(
            images, bboxes, "xyxy", color=(0, 255, 0)
        )
        # Both box regions should have non-zero pixels
        region_1 = result[0, 2:15, 2:15, :]
        region_2 = result[0, 30:50, 30:50, :]
        self.assertTrue(np.any(region_1 != 0))
        self.assertTrue(np.any(region_2 != 0))

    # ── class mapping / confidence ───────────────────────────────────────────

    def test_class_mapping_used_without_error(self):
        images = _solid_images(batch=1, h=64, w=64, c=3, value=128)
        bboxes = _bbox_dict([[[5, 5, 30, 30]]], [[0]])
        class_mapping = {0: "cat"}
        result = self.draw_bounding_boxes(
            images, bboxes, "xyxy", class_mapping=class_mapping
        )
        self.assertEqual(result.shape, images.shape)

    def test_confidences_included(self):
        images = _solid_images(batch=1, h=64, w=64, c=3, value=128)
        bboxes = _bbox_dict([[[5, 5, 30, 30]]], [[0]], confidences=[[0.92]])
        class_mapping = {0: "dog"}
        result = self.draw_bounding_boxes(
            images, bboxes, "xyxy", class_mapping=class_mapping
        )
        self.assertEqual(result.shape, images.shape)

    # ── float images ─────────────────────────────────────────────────────────

    def test_float_images_range_0_to_1(self):
        images = _solid_images_float(batch=1, h=32, w=32, c=3, value=0.5)
        bboxes = _bbox_dict([[[2, 2, 20, 20]]], [[0]])
        result = self.draw_bounding_boxes(images, bboxes, "xyxy")
        self.assertEqual(result.dtype, np.uint8)
        # Background pixels should be near 127 (0.5 * 255)
        self.assertTrue(np.all(result[0, 0, 0, :] > 100))

    def test_float_images_range_0_to_255(self):
        images = np.full((1, 32, 32, 3), 100.0, dtype=np.float32)
        bboxes = _bbox_dict([[[2, 2, 20, 20]]], [[0]])
        result = self.draw_bounding_boxes(images, bboxes, "xyxy")
        self.assertEqual(result.dtype, np.uint8)
        # Background pixels should be ≈ 100
        self.assertTrue(np.all(result[0, 0, 0, :] == 100))

    # ── channels_first ───────────────────────────────────────────────────────

    def test_channels_first_input_is_handled(self):
        # Shape (batch, C, H, W)
        images = np.zeros((1, 3, 32, 32), dtype="uint8")
        bboxes = _bbox_dict([[[2, 2, 20, 20]]], [[0]])
        result = self.draw_bounding_boxes(
            images, bboxes, "xyxy", data_format="channels_first"
        )
        self.assertEqual(result.shape, (1, 32, 32, 3))

    # ── xywh format ──────────────────────────────────────────────────────────

    def test_xywh_format_converted_correctly(self):
        images = _solid_images(batch=1, h=64, w=64, c=3, value=0)
        # Box: x=10, y=10, w=20, h=20 → xyxy: (10,10,30,30)
        bboxes = _bbox_dict([[[10, 10, 20, 20]]], [[0]])
        result = self.draw_bounding_boxes(
            images, bboxes, "xywh", color=(0, 0, 255)
        )
        top_edge = result[0, 10, 10:30, :]
        self.assertTrue(np.any(top_edge != 0))

    # ── batch of N images ─────────────────────────────────────────────────────

    def test_batch_processing(self):
        images = _solid_images(batch=4, h=32, w=32, c=3, value=0)
        bboxes = _bbox_dict([[[2, 2, 10, 10]]] * 4, [[0]] * 4)
        result = self.draw_bounding_boxes(images, bboxes, "xyxy")
        self.assertEqual(result.shape[0], 4)


class FindTextLocationTest(test_case.TestCase):
    """Test the internal _find_text_location helper directly."""

    def setUp(self):
        pytest.importorskip("cv2")
        from keras.src.visualization.draw_bounding_boxes import (
            _find_text_location,
        )

        self._fn = _find_text_location

    def test_text_above_when_space_available(self):
        x, y = 10, 100
        fx, fy = self._fn(x, y, font_scale=1.0, thickness=1)
        self.assertEqual(fx, x)
        self.assertEqual(fy, y - 8)

    def test_text_below_when_near_top(self):
        x, y = 10, 5  # y=5 → target_y=5-8=-3, no room above
        fx, fy = self._fn(x, y, font_scale=1.0, thickness=1)
        # Must be below the original y
        self.assertGreater(fy, y)

    def test_text_location_uses_font_scale(self):
        x, y = 5, 5
        _, fy_small = self._fn(x, y, font_scale=0.5, thickness=1)
        _, fy_large = self._fn(x, y, font_scale=2.0, thickness=1)
        self.assertGreater(fy_large, fy_small)


# ═══════════════════════════════════════════════════════════════════════════
#  draw_segmentation_masks
# ═══════════════════════════════════════════════════════════════════════════


class DrawSegmentationMasksTest(test_case.TestCase):
    def setUp(self):
        from keras.src import ops
        from keras.src.visualization.draw_segmentation_masks import (
            _generate_color_palette,
        )
        from keras.src.visualization.draw_segmentation_masks import (
            draw_segmentation_masks,
        )

        self.ops = ops
        self.draw_segmentation_masks = draw_segmentation_masks
        self.generate_palette = _generate_color_palette

    # ── validation ──────────────────────────────────────────────────────────

    def test_raises_value_error_for_3d_input(self):
        images_3d = np.zeros((32, 32, 3), dtype="uint8")
        masks = np.zeros((32, 32), dtype=np.int32)
        with self.assertRaises(ValueError):
            self.draw_segmentation_masks(images_3d, masks)

    def test_raises_type_error_for_float_masks(self):
        images = _solid_images(batch=1)
        masks = np.zeros((1, 32, 32), dtype=np.float32)
        with self.assertRaises(TypeError):
            self.draw_segmentation_masks(images, masks)

    # ── output shape / type ─────────────────────────────────────────────────

    def test_output_shape_matches_input(self):
        images = _solid_images(batch=2, h=16, w=16, c=3, value=200)
        masks = np.zeros((2, 16, 16), dtype=np.int32)
        result = self.draw_segmentation_masks(images, masks, num_classes=3)
        self.assertEqual(result.shape, images.shape)

    def test_output_dtype_uint8(self):
        images = _solid_images(batch=1, h=8, w=8, c=3, value=200)
        masks = np.ones((1, 8, 8), dtype=np.int32)
        result = self.draw_segmentation_masks(images, masks, num_classes=2)
        self.assertEqual(result.dtype, np.uint8)

    # ── alpha blending math ──────────────────────────────────────────────────

    def test_alpha_blend_all_same_class(self):
        h, w = 8, 8
        bg_value = 200  # original image pixel value
        alpha = 0.8
        num_classes = 2  # class 1 is the only non-BG class

        images = _solid_images(batch=1, h=h, w=w, c=3, value=bg_value)
        masks = np.ones((1, h, w), dtype=np.int32)

        # Use a fixed colour palette so we can predict the output
        palette = self.generate_palette(num_classes)
        class_color = np.array(palette[1], dtype=np.float32)

        result = self.draw_segmentation_masks(
            images, masks, num_classes=num_classes, alpha=alpha, blend=True
        )
        expected = (bg_value * (1 - alpha) + class_color * alpha).astype(
            np.uint8
        )
        # All pixels should equal the expected blended value
        np.testing.assert_array_equal(result[0, 0, 0, :], expected)

    def test_alpha_zero_preserves_original(self):
        images = _solid_images(batch=1, h=8, w=8, c=3, value=123)
        masks = np.ones((1, 8, 8), dtype=np.int32)
        result = self.draw_segmentation_masks(
            images, masks, num_classes=2, alpha=0.0, blend=True
        )
        np.testing.assert_array_equal(result[0], images[0])

    def test_no_blend_replaces_pixel_with_color(self):
        h, w = 8, 8
        images = _solid_images(batch=1, h=h, w=w, c=3, value=50)
        masks = np.ones((1, h, w), dtype=np.int32)

        palette = self.generate_palette(2)
        class_color = np.array(palette[1], dtype=np.uint8)

        result = self.draw_segmentation_masks(
            images, masks, num_classes=2, blend=False
        )
        result_uint8 = self.ops.convert_to_numpy(result).astype(np.uint8)
        np.testing.assert_array_equal(result_uint8[0, 0, 0, :], class_color)

    # ── ignore_index ─────────────────────────────────────────────────────────

    def test_ignore_index_pixels_not_changed(self):
        images = _solid_images(batch=1, h=8, w=8, c=3, value=77)
        # All pixels == ignore_index → nothing should change
        masks = np.full((1, 8, 8), -1, dtype=np.int32)
        result = self.draw_segmentation_masks(
            images, masks, num_classes=3, alpha=0.9, ignore_index=-1
        )
        np.testing.assert_array_equal(result[0], images[0])

    def test_custom_ignore_index(self):
        images = _solid_images(batch=1, h=4, w=4, c=3, value=88)
        masks = np.zeros((1, 4, 4), dtype=np.int32)
        result = self.draw_segmentation_masks(
            images, masks, num_classes=3, ignore_index=0
        )
        np.testing.assert_array_equal(result[0], images[0])

    # ── custom color_mapping ─────────────────────────────────────────────────

    def test_custom_color_mapping(self):
        h, w = 8, 8
        images = _solid_images(batch=1, h=h, w=w, c=3, value=0)
        masks = np.ones((1, h, w), dtype=np.int32)
        color_mapping = {0: [0, 0, 0], 1: [255, 0, 0]}  # class 1 = red
        result = self.draw_segmentation_masks(
            images,
            masks,
            num_classes=2,
            color_mapping=color_mapping,
            alpha=1.0,
            blend=True,
        )
        # With alpha=1.0 and 100% class-1 pixels the output should be ≈ red
        # R-channel must be significantly positive, G and B near 0
        self.assertGreater(int(result[0, 0, 0, 0]), 200)

    # ── num_classes inferred ──────────────────────────────────────────────────

    def test_num_classes_inferred_from_mask(self):
        images = _solid_images(batch=1, h=8, w=8, c=3, value=100)
        masks = np.array([[[0, 1, 2, 0, 1, 2, 0, 1]] * 8], dtype=np.int32)
        # Should not raise even without explicit num_classes
        result = self.draw_segmentation_masks(images, masks)
        self.assertEqual(result.shape, images.shape)

    # ── color palette ─────────────────────────────────────────────────────────

    def test_generate_color_palette_length(self):
        palette = self.generate_palette(5)
        self.assertEqual(len(palette), 5)

    def test_generate_color_palette_deterministic(self):
        p1 = self.generate_palette(10)
        p2 = self.generate_palette(10)
        self.assertEqual(p1, p2)

    def test_generate_color_palette_values_in_range(self):
        palette = self.generate_palette(20)
        for color in palette:
            self.assertEqual(len(color), 3)
            for c in color:
                self.assertGreaterEqual(c, 0)
                self.assertLessEqual(c, 255)

    # ── batch consistency ────────────────────────────────────────────────────

    def test_batch_independence(self):
        images = _solid_images(batch=2, h=8, w=8, c=3, value=100)
        masks = np.array(
            [np.ones((8, 8), dtype=np.int32), np.zeros((8, 8), dtype=np.int32)]
        )
        result_batch = self.draw_segmentation_masks(
            images, masks, num_classes=2, alpha=0.5, blend=True
        )

        result_0 = self.draw_segmentation_masks(
            images[0:1], masks[0:1], num_classes=2, alpha=0.5, blend=True
        )
        result_1 = self.draw_segmentation_masks(
            images[1:2], masks[1:2], num_classes=2, alpha=0.5, blend=True
        )
        np.testing.assert_array_equal(result_batch[0], result_0[0])
        np.testing.assert_array_equal(result_batch[1], result_1[0])


# ═══════════════════════════════════════════════════════════════════════════
#  plot_image_gallery
# ═══════════════════════════════════════════════════════════════════════════


class PlotImageGalleryTest(test_case.TestCase):
    def setUp(self):
        pytest.importorskip("matplotlib")
        from keras.src.visualization.plot_image_gallery import (
            plot_image_gallery,
        )

        self.plot_image_gallery = plot_image_gallery

    # ── validation ──────────────────────────────────────────────────────────

    def test_raises_value_error_path_and_show(self):
        images = _solid_images(batch=1)
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            with self.assertRaises(ValueError):
                self.plot_image_gallery(images, path=f.name, show=True)

    def test_raises_value_error_ytrue_without_labelmap(self):
        images = _solid_images(batch=2)
        with self.assertRaises(ValueError):
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                self.plot_image_gallery(
                    images, y_true=np.array([0, 1]), path=f.name
                )

    def test_raises_value_error_ypred_without_labelmap(self):
        images = _solid_images(batch=2)
        with self.assertRaises(ValueError):
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                self.plot_image_gallery(
                    images, y_pred=np.array([0, 1]), path=f.name
                )

    def test_raises_value_error_for_3d_input(self):
        images_3d = np.zeros((32, 32, 3), dtype="uint8")
        with self.assertRaises(ValueError):
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                self.plot_image_gallery(images_3d, path=f.name)

    # ── file creation ────────────────────────────────────────────────────────

    def test_saves_png_file(self):
        images = _solid_images(batch=4, h=16, w=16, c=3, value=128)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_image_gallery(images, path=path, rows=2, cols=2)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_saves_with_labels(self):
        images = _solid_images(batch=2, h=16, w=16, c=3, value=100)
        label_map = {0: "cat", 1: "dog"}
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_image_gallery(
                images,
                y_true=np.array([0, 1]),
                y_pred=np.array([1, 1]),
                label_map=label_map,
                path=path,
                rows=1,
                cols=2,
            )
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    # ── value_range normalization ─────────────────────────────────────────────

    def test_value_range_0_to_1_normalizes_to_255(self):
        images = _solid_images_float(batch=1, h=8, w=8, c=3, value=0.5)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            # Should not raise – normalization handles float images
            self.plot_image_gallery(
                images, value_range=(0, 1), path=path, rows=1, cols=1
            )
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    # ── channels_first ───────────────────────────────────────────────────────

    def test_channels_first_creates_file(self):
        # Shape (batch, C, H, W)
        images = np.zeros((1, 3, 16, 16), dtype="uint8")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_image_gallery(
                images,
                path=path,
                rows=1,
                cols=1,
                data_format="channels_first",
            )
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    # ── auto rows/cols ────────────────────────────────────────────────────────

    def test_auto_rows_cols_from_batch(self):
        batch = 9
        images = _solid_images(batch=batch, h=8, w=8, c=3)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            # Rows = ceil(sqrt(9)) = 3, cols = ceil(9/3) = 3
            self.plot_image_gallery(images, path=path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
#  plot_bounding_box_gallery
# ═══════════════════════════════════════════════════════════════════════════


class PlotBoundingBoxGalleryTest(test_case.TestCase):
    def setUp(self):
        pytest.importorskip("cv2")
        pytest.importorskip("matplotlib")
        from keras.src.visualization.plot_bounding_box_gallery import (
            plot_bounding_box_gallery,
        )

        self.plot_bounding_box_gallery = plot_bounding_box_gallery

    def _make_bboxes(self):
        return {
            "boxes": np.array(
                [[[5, 5, 20, 20]], [[3, 3, 15, 15]]], dtype=np.float32
            ),
            "labels": np.array([[0], [1]], dtype=np.int32),
        }

    # ── validation ──────────────────────────────────────────────────────────

    def test_raises_for_3d_images(self):
        images_3d = np.zeros((32, 32, 3), dtype="uint8")
        bboxes = _bbox_dict([[[2, 2, 10, 10]]], [[0]])
        with self.assertRaises(ValueError):
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                self.plot_bounding_box_gallery(
                    images_3d, "xyxy", y_true=bboxes, path=f.name
                )

    def test_raises_legend_and_handles_conflict(self):
        images = _solid_images(batch=2)
        bboxes = self._make_bboxes()
        from matplotlib import patches as mpatches

        handles = [mpatches.Patch(color="red", label="X")]
        with self.assertRaises(ValueError):
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                self.plot_bounding_box_gallery(
                    images,
                    "xyxy",
                    y_true=bboxes,
                    legend=True,
                    legend_handles=handles,
                    path=f.name,
                )

    # ── file creation ────────────────────────────────────────────────────────

    def test_saves_with_ytrue(self):
        images = _solid_images(batch=2, h=32, w=32, c=3)
        bboxes = self._make_bboxes()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_bounding_box_gallery(
                images,
                "xyxy",
                y_true=bboxes,
                path=path,
                rows=1,
                cols=2,
                show=False,
            )
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_saves_with_ytrue_and_ypred(self):
        images = _solid_images(batch=2, h=32, w=32, c=3)
        gt = self._make_bboxes()
        pred = {
            "boxes": np.array(
                [[[6, 6, 21, 21]], [[4, 4, 16, 16]]], dtype=np.float32
            ),
            "labels": np.array([[0], [1]], dtype=np.int32),
        }
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_bounding_box_gallery(
                images,
                "xyxy",
                y_true=gt,
                y_pred=pred,
                path=path,
                rows=1,
                cols=2,
                show=False,
            )
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_saves_with_legend(self):
        images = _solid_images(batch=2, h=32, w=32, c=3)
        bboxes = self._make_bboxes()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_bounding_box_gallery(
                images,
                "xyxy",
                y_true=bboxes,
                legend=True,
                path=path,
                rows=1,
                cols=2,
                show=False,
            )
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_class_mapping_applied(self):
        images = _solid_images(batch=1, h=32, w=32, c=3)
        bboxes = _bbox_dict([[[5, 5, 20, 20]]], [[0]])
        class_mapping = {0: "bird"}
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_bounding_box_gallery(
                images,
                "xyxy",
                y_true=bboxes,
                class_mapping=class_mapping,
                path=path,
                rows=1,
                cols=1,
                show=False,
            )
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_channels_first_images(self):
        images = np.zeros((1, 3, 32, 32), dtype="uint8")
        bboxes = _bbox_dict([[[2, 2, 15, 15]]], [[0]])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_bounding_box_gallery(
                images,
                "xyxy",
                y_true=bboxes,
                path=path,
                rows=1,
                cols=1,
                show=False,
                data_format="channels_first",
            )
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
#  plot_segmentation_mask_gallery
# ═══════════════════════════════════════════════════════════════════════════


class PlotSegmentationMaskGalleryTest(test_case.TestCase):
    def setUp(self):
        pytest.importorskip("matplotlib")
        from keras.src.visualization.plot_segmentation_mask_gallery import (
            plot_segmentation_mask_gallery,
        )

        self.plot_segmentation_mask_gallery = plot_segmentation_mask_gallery

    # ── validation ──────────────────────────────────────────────────────────

    def test_raises_for_3d_images(self):
        images_3d = np.zeros((32, 32, 3), dtype="uint8")
        with self.assertRaises(ValueError):
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                self.plot_segmentation_mask_gallery(
                    images_3d, num_classes=3, path=f.name
                )

    # ── file creation ────────────────────────────────────────────────────────

    def test_saves_without_masks(self):
        images = _solid_images(batch=2, h=16, w=16, c=3, value=128)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_segmentation_mask_gallery(
                images, num_classes=3, path=path, show=False
            )
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_saves_with_ytrue(self):
        images = _solid_images(batch=2, h=16, w=16, c=3, value=100)
        masks = np.ones((2, 16, 16), dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_segmentation_mask_gallery(
                images,
                num_classes=3,
                y_true=masks,
                path=path,
                show=False,
            )
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_saves_with_ytrue_and_ypred(self):
        images = _solid_images(batch=2, h=16, w=16, c=3, value=100)
        y_true = np.ones((2, 16, 16), dtype=np.int32)
        y_pred = np.full((2, 16, 16), 2, dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_segmentation_mask_gallery(
                images,
                num_classes=3,
                y_true=y_true,
                y_pred=y_pred,
                path=path,
                show=False,
            )
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_channels_first(self):
        images = np.zeros((1, 3, 16, 16), dtype="uint8")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_segmentation_mask_gallery(
                images,
                num_classes=3,
                path=path,
                show=False,
                data_format="channels_first",
            )
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_rows_cols_reflect_content(self):
        batch = 3
        images = _solid_images(batch=batch, h=8, w=8, c=3, value=100)
        y_true = np.ones((batch, 8, 8), dtype=np.int32)
        y_pred = np.ones((batch, 8, 8), dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_segmentation_mask_gallery(
                images,
                num_classes=2,
                y_true=y_true,
                y_pred=y_pred,
                path=path,
                show=False,
            )
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_custom_color_mapping_used(self):
        images = _solid_images(batch=1, h=8, w=8, c=3, value=128)
        masks = np.ones((1, 8, 8), dtype=np.int32)
        color_mapping = {0: [0, 0, 0], 1: [200, 50, 50], 2: [50, 200, 50]}
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            self.plot_segmentation_mask_gallery(
                images,
                num_classes=3,
                y_true=masks,
                color_mapping=color_mapping,
                path=path,
                show=False,
            )
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    from keras.src import testing as _testing

    _testing.run_tests()
