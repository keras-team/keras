"""Tests for bounding box format conversion, clipping, cropping, padding."""

import numpy as np

from keras.src import testing
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.bounding_box import (  # noqa: E501
    SUPPORTED_FORMATS,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.bounding_box import (  # noqa: E501
    BoundingBox,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.formats import (  # noqa: E501
    CENTER_XYWH,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.formats import (  # noqa: E501
    REL_XYXY,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.formats import (  # noqa: E501
    REL_YXYX,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.formats import (  # noqa: E501
    XYWH,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.formats import (  # noqa: E501
    XYXY,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.formats import (  # noqa: E501
    YXYX,
)


class BoundingBoxFormatsTest(testing.TestCase):
    """Test the format index classes are consistent."""

    def test_xyxy_indices(self):
        self.assertEqual(XYXY.LEFT, 0)
        self.assertEqual(XYXY.TOP, 1)
        self.assertEqual(XYXY.RIGHT, 2)
        self.assertEqual(XYXY.BOTTOM, 3)

    def test_yxyx_indices(self):
        self.assertEqual(YXYX.TOP, 0)
        self.assertEqual(YXYX.LEFT, 1)
        self.assertEqual(YXYX.BOTTOM, 2)
        self.assertEqual(YXYX.RIGHT, 3)

    def test_xywh_indices(self):
        self.assertEqual(XYWH.X, 0)
        self.assertEqual(XYWH.Y, 1)
        self.assertEqual(XYWH.WIDTH, 2)
        self.assertEqual(XYWH.HEIGHT, 3)

    def test_center_xywh_indices(self):
        self.assertEqual(CENTER_XYWH.X, 0)
        self.assertEqual(CENTER_XYWH.Y, 1)
        self.assertEqual(CENTER_XYWH.WIDTH, 2)
        self.assertEqual(CENTER_XYWH.HEIGHT, 3)

    def test_rel_xyxy_matches_xyxy(self):
        self.assertEqual(REL_XYXY.LEFT, XYXY.LEFT)
        self.assertEqual(REL_XYXY.TOP, XYXY.TOP)
        self.assertEqual(REL_XYXY.RIGHT, XYXY.RIGHT)
        self.assertEqual(REL_XYXY.BOTTOM, XYXY.BOTTOM)

    def test_rel_yxyx_matches_yxyx(self):
        self.assertEqual(REL_YXYX.TOP, YXYX.TOP)
        self.assertEqual(REL_YXYX.LEFT, YXYX.LEFT)
        self.assertEqual(REL_YXYX.BOTTOM, YXYX.BOTTOM)
        self.assertEqual(REL_YXYX.RIGHT, YXYX.RIGHT)


class BoundingBoxConvertFormatTest(testing.TestCase):
    def setUp(self):
        self.bb = BoundingBox()

    def test_xyxy_to_xyxy_identity(self):
        boxes = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32")
        result = self.bb.convert_format(boxes, source="xyxy", target="xyxy")
        self.assertAllClose(result, boxes)

    def test_xyxy_to_yxyx(self):
        # xyxy: [x1, y1, x2, y2] -> yxyx: [y1, x1, y2, x2]
        boxes = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32")
        result = self.bb.convert_format(boxes, source="xyxy", target="yxyx")
        expected = np.array([[[20.0, 10.0, 60.0, 50.0]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_yxyx_to_xyxy(self):
        boxes = np.array([[[20.0, 10.0, 60.0, 50.0]]], dtype="float32")
        result = self.bb.convert_format(boxes, source="yxyx", target="xyxy")
        expected = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_xyxy_to_xywh(self):
        # xyxy: [x1, y1, x2, y2] -> xywh: [x1, y1, w, h]
        boxes = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32")
        result = self.bb.convert_format(boxes, source="xyxy", target="xywh")
        expected = np.array([[[10.0, 20.0, 40.0, 40.0]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_xywh_to_xyxy(self):
        boxes = np.array([[[10.0, 20.0, 40.0, 40.0]]], dtype="float32")
        result = self.bb.convert_format(boxes, source="xywh", target="xyxy")
        expected = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_xyxy_to_center_xywh(self):
        # xyxy: [10, 20, 50, 60] -> center: cx=30, cy=40, w=40, h=40
        boxes = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32")
        result = self.bb.convert_format(
            boxes, source="xyxy", target="center_xywh"
        )
        expected = np.array([[[30.0, 40.0, 40.0, 40.0]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_center_xywh_to_xyxy(self):
        boxes = np.array([[[30.0, 40.0, 40.0, 40.0]]], dtype="float32")
        result = self.bb.convert_format(
            boxes, source="center_xywh", target="xyxy"
        )
        expected = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_xyxy_to_rel_xyxy(self):
        boxes = np.array([[[50.0, 50.0, 100.0, 100.0]]], dtype="float32")
        result = self.bb.convert_format(
            boxes, source="xyxy", target="rel_xyxy", height=200, width=200
        )
        expected = np.array([[[0.25, 0.25, 0.5, 0.5]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_rel_xyxy_to_xyxy(self):
        boxes = np.array([[[0.25, 0.25, 0.5, 0.5]]], dtype="float32")
        result = self.bb.convert_format(
            boxes, source="rel_xyxy", target="xyxy", height=200, width=200
        )
        expected = np.array([[[50.0, 50.0, 100.0, 100.0]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_roundtrip_xywh(self):
        boxes = np.array([[[15.0, 25.0, 30.0, 40.0]]], dtype="float32")
        intermediate = self.bb.convert_format(
            boxes, source="xyxy", target="xywh"
        )
        result = self.bb.convert_format(
            intermediate, source="xywh", target="xyxy"
        )
        self.assertAllClose(result, boxes)

    def test_roundtrip_center_xywh(self):
        boxes = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32")
        intermediate = self.bb.convert_format(
            boxes, source="xyxy", target="center_xywh"
        )
        result = self.bb.convert_format(
            intermediate, source="center_xywh", target="xyxy"
        )
        self.assertAllClose(result, boxes)

    def test_roundtrip_rel_xyxy(self):
        boxes = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32")
        intermediate = self.bb.convert_format(
            boxes, source="xyxy", target="rel_xyxy", height=100, width=100
        )
        result = self.bb.convert_format(
            intermediate,
            source="rel_xyxy",
            target="xyxy",
            height=100,
            width=100,
        )
        self.assertAllClose(result, boxes)

    def test_multiple_boxes(self):
        boxes = np.array(
            [[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]],
            dtype="float32",
        )
        result = self.bb.convert_format(boxes, source="xyxy", target="xywh")
        expected = np.array(
            [[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 10.0, 10.0]]],
            dtype="float32",
        )
        self.assertAllClose(result, expected)

    def test_dict_input(self):
        bboxes = {
            "boxes": np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32"),
            "labels": np.array([[1]]),
        }
        result = self.bb.convert_format(bboxes, source="xyxy", target="xywh")
        self.assertIn("boxes", result)
        expected_boxes = np.array([[[10.0, 20.0, 40.0, 40.0]]], dtype="float32")
        self.assertAllClose(result["boxes"], expected_boxes)


class BoundingBoxValidationTest(testing.TestCase):
    def setUp(self):
        self.bb = BoundingBox()

    def test_invalid_last_dim_raises(self):
        boxes = np.array([[[1.0, 2.0, 3.0]]], dtype="float32")  # Last dim = 3
        with self.assertRaisesRegex(ValueError, "last dimension of 4"):
            self.bb.convert_format(boxes, source="xyxy", target="yxyx")

    def test_invalid_source_format_raises(self):
        boxes = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype="float32")
        with self.assertRaisesRegex(ValueError, "Invalid source or target"):
            self.bb.convert_format(
                boxes, source="invalid_format", target="xyxy"
            )

    def test_invalid_target_format_raises(self):
        boxes = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype="float32")
        with self.assertRaisesRegex(ValueError, "Invalid source or target"):
            self.bb.convert_format(
                boxes, source="xyxy", target="invalid_format"
            )

    def test_rel_format_requires_height_width(self):
        boxes = np.array([[[0.1, 0.2, 0.5, 0.6]]], dtype="float32")
        with self.assertRaisesRegex(ValueError, "height.*width"):
            self.bb.convert_format(boxes, source="rel_xyxy", target="xyxy")

    def test_supported_formats_is_tuple(self):
        self.assertIsInstance(SUPPORTED_FORMATS, tuple)
        self.assertIn("xyxy", SUPPORTED_FORMATS)
        self.assertIn("yxyx", SUPPORTED_FORMATS)
        self.assertIn("xywh", SUPPORTED_FORMATS)
        self.assertIn("rel_xyxy", SUPPORTED_FORMATS)


class BoundingBoxClipTest(testing.TestCase):
    def setUp(self):
        self.bb = BoundingBox()

    def test_clip_boxes_within_image(self):
        bboxes = {
            "boxes": np.array([[[10.0, 10.0, 50.0, 50.0]]], dtype="float32"),
        }
        result = self.bb.clip_to_image_size(
            bboxes, height=100, width=100, bounding_box_format="xyxy"
        )
        self.assertAllClose(result["boxes"], bboxes["boxes"])

    def test_clip_boxes_exceeding_bounds(self):
        bboxes = {
            "boxes": np.array(
                [[[-10.0, -10.0, 150.0, 150.0]]], dtype="float32"
            ),
        }
        result = self.bb.clip_to_image_size(
            bboxes, height=100, width=100, bounding_box_format="xyxy"
        )
        expected = np.array([[[0.0, 0.0, 100.0, 100.0]]], dtype="float32")
        self.assertAllClose(result["boxes"], expected)

    def test_clip_with_labels_marks_zero_area(self):
        bboxes = {
            "boxes": np.array(
                [[[10.0, 10.0, 10.0, 10.0]]], dtype="float32"
            ),  # zero area
            "labels": np.array([[5]]),
        }
        result = self.bb.clip_to_image_size(
            bboxes, height=100, width=100, bounding_box_format="xyxy"
        )
        # Zero-area box should get label -1
        self.assertEqual(int(result["labels"][0][0]), -1)

    def test_clip_rel_format(self):
        bboxes = {
            "boxes": np.array([[[-0.1, -0.1, 1.5, 1.5]]], dtype="float32"),
        }
        result = self.bb.clip_to_image_size(
            bboxes, bounding_box_format="rel_xyxy"
        )
        expected = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype="float32")
        self.assertAllClose(result["boxes"], expected)

    def test_clip_unsupported_format_raises(self):
        bboxes = {"boxes": np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype="float32")}
        with self.assertRaises(NotImplementedError):
            self.bb.clip_to_image_size(
                bboxes, height=100, width=100, bounding_box_format="xywh"
            )


class BoundingBoxCropTest(testing.TestCase):
    def setUp(self):
        self.bb = BoundingBox()

    def test_crop_shifts_boxes(self):
        boxes = np.array([[[20.0, 30.0, 60.0, 70.0]]], dtype="float32")
        result = self.bb.crop(boxes, top=10, left=10, height=100, width=100)
        expected = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_crop_clips_to_bounds(self):
        boxes = np.array([[[5.0, 5.0, 50.0, 50.0]]], dtype="float32")
        result = self.bb.crop(boxes, top=10, left=10, height=30, width=30)
        # After shifting: [-5, -5, 40, 40], clipped to [0, 0, 30, 30]
        expected = np.array([[[0.0, 0.0, 30.0, 30.0]]], dtype="float32")
        self.assertAllClose(result, expected)


class BoundingBoxPadTest(testing.TestCase):
    def setUp(self):
        self.bb = BoundingBox()

    def test_pad_shifts_boxes(self):
        boxes = np.array([[[10.0, 10.0, 50.0, 50.0]]], dtype="float32")
        result = self.bb.pad(boxes, top=5, left=10)
        expected = np.array([[[20.0, 15.0, 60.0, 55.0]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_pad_zero(self):
        boxes = np.array([[[10.0, 10.0, 50.0, 50.0]]], dtype="float32")
        result = self.bb.pad(boxes, top=0, left=0)
        self.assertAllClose(result, boxes)


class BoundingBoxMissingFormatTests(testing.TestCase):
    """Test formats not yet covered: center_yxhw, rel_yxyx, rel_xywh,
    rel_center_xywh — including roundtrips with non-square images.
    """

    def setUp(self):
        self.bb = BoundingBox()

    # ── center_yxhw ──────────────────────────────────────────────────────────

    def test_xyxy_to_center_yxhw(self):
        # xyxy [10,20,50,60] → center_yxhw: cy=40, cx=30, h=40, w=40
        boxes = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32")
        result = self.bb.convert_format(
            boxes, source="xyxy", target="center_yxhw"
        )
        expected = np.array([[[40.0, 30.0, 40.0, 40.0]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_center_yxhw_to_xyxy(self):
        boxes = np.array([[[40.0, 30.0, 40.0, 40.0]]], dtype="float32")
        result = self.bb.convert_format(
            boxes, source="center_yxhw", target="xyxy"
        )
        expected = np.array([[[10.0, 20.0, 50.0, 60.0]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_roundtrip_center_yxhw(self):
        boxes = np.array([[[5.0, 15.0, 45.0, 55.0]]], dtype="float32")
        inter = self.bb.convert_format(
            boxes, source="xyxy", target="center_yxhw"
        )
        back = self.bb.convert_format(
            inter, source="center_yxhw", target="xyxy"
        )
        self.assertAllClose(back, boxes)

    # ── rel_yxyx ─────────────────────────────────────────────────────────────

    def test_xyxy_to_rel_yxyx(self):
        # xyxy [50, 80, 150, 200], h=400, w=200 →
        # rel_yxyx: [80/400, 50/200, 200/400, 150/200] = [0.2, 0.25, 0.5, 0.75]
        h, w = 400, 200
        boxes = np.array([[[50.0, 80.0, 150.0, 200.0]]], dtype="float32")
        result = self.bb.convert_format(
            boxes, source="xyxy", target="rel_yxyx", height=h, width=w
        )
        expected = np.array([[[0.2, 0.25, 0.5, 0.75]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_rel_yxyx_to_xyxy(self):
        h, w = 400, 200
        boxes = np.array([[[0.2, 0.25, 0.5, 0.75]]], dtype="float32")
        result = self.bb.convert_format(
            boxes, source="rel_yxyx", target="xyxy", height=h, width=w
        )
        expected = np.array([[[50.0, 80.0, 150.0, 200.0]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_roundtrip_rel_yxyx_non_square(self):
        h, w = 300, 200
        boxes = np.array([[[0.0, 0.0, 100.0, 150.0]]], dtype="float32")
        inter = self.bb.convert_format(
            boxes, source="xyxy", target="rel_yxyx", height=h, width=w
        )
        back = self.bb.convert_format(
            inter, source="rel_yxyx", target="xyxy", height=h, width=w
        )
        self.assertAllClose(back, boxes)

    # ── rel_xywh ─────────────────────────────────────────────────────────────

    def test_xyxy_to_rel_xywh(self):
        # xyxy [50, 80, 150, 200], h=400, w=200:
        # rel_xywh: [50/200, 80/400, 100/200, 120/400] = [0.25, 0.2, 0.5, 0.3]
        h, w = 400, 200
        boxes = np.array([[[50.0, 80.0, 150.0, 200.0]]], dtype="float32")
        result = self.bb.convert_format(
            boxes, source="xyxy", target="rel_xywh", height=h, width=w
        )
        expected = np.array([[[0.25, 0.2, 0.5, 0.3]]], dtype="float32")
        self.assertAllClose(result, expected)

    def test_roundtrip_rel_xywh_non_square(self):
        h, w = 300, 200
        boxes = np.array([[[10.0, 20.0, 80.0, 120.0]]], dtype="float32")
        inter = self.bb.convert_format(
            boxes, source="xyxy", target="rel_xywh", height=h, width=w
        )
        back = self.bb.convert_format(
            inter, source="rel_xywh", target="xyxy", height=h, width=w
        )
        self.assertAllClose(back, boxes)

    # ── rel_center_xywh (catches swapped height/width in converter) ───────────

    def test_xyxy_to_rel_center_xywh_non_square(self):
        # Non-square image: height=400, width=200
        # xyxy [50, 80, 150, 200]: cx=100, cy=140, w=100, h=120
        # rel_cx = 100/200=0.5, rel_cy = 140/400=0.35
        # rel_w = 100/200=0.5, rel_h = 120/400=0.3
        h, w = 400, 200
        boxes = np.array([[[50.0, 80.0, 150.0, 200.0]]], dtype="float32")
        result = self.bb.convert_format(
            boxes, source="xyxy", target="rel_center_xywh", height=h, width=w
        )
        expected = np.array([[[0.5, 0.35, 0.5, 0.3]]], dtype="float32")
        self.assertAllClose(result, expected, atol=1e-5)

    def test_roundtrip_rel_center_xywh_non_square(self):
        """Roundtrip must reconstruct the original xyxy box exactly.

        This catches a bug where height and width are swapped in
        `_rel_center_xywh_to_xyxy`.  Use height ≠ width so the swap is
        detectable.
        """
        h, w = 400, 200  # deliberately non-square
        boxes = np.array([[[50.0, 80.0, 150.0, 200.0]]], dtype="float32")
        inter = self.bb.convert_format(
            boxes, source="xyxy", target="rel_center_xywh", height=h, width=w
        )
        back = self.bb.convert_format(
            inter, source="rel_center_xywh", target="xyxy", height=h, width=w
        )
        self.assertAllClose(back, boxes, atol=1e-4)


class BoundingBoxBatchTest(testing.TestCase):
    """Tests with batch_size > 1 to verify per-sample independence."""

    def setUp(self):
        self.bb = BoundingBox()

    def test_convert_format_batch(self):
        boxes = np.array(
            [
                [[10.0, 20.0, 50.0, 60.0]],
                [[5.0, 5.0, 25.0, 25.0]],
            ],
            dtype="float32",
        )
        result = self.bb.convert_format(boxes, source="xyxy", target="xywh")
        expected = np.array(
            [
                [[10.0, 20.0, 40.0, 40.0]],
                [[5.0, 5.0, 20.0, 20.0]],
            ],
            dtype="float32",
        )
        self.assertAllClose(result, expected)

    def test_clip_batch_mixed_clipping(self):
        # Image size 100×100; first box already within, second exceeds
        bboxes = {
            "boxes": np.array(
                [
                    [[10.0, 10.0, 50.0, 50.0]],  # inside
                    [[-5.0, -5.0, 120.0, 120.0]],  # outside on all sides
                ],
                dtype="float32",
            ),
            "labels": np.array([[1], [2]]),
        }
        result = self.bb.clip_to_image_size(
            bboxes, height=100, width=100, bounding_box_format="xyxy"
        )
        # First box unchanged
        self.assertAllClose(
            result["boxes"][0], np.array([[10.0, 10.0, 50.0, 50.0]])
        )
        # Second box clipped
        self.assertAllClose(
            result["boxes"][1], np.array([[0.0, 0.0, 100.0, 100.0]])
        )

    def test_pad_batch(self):
        boxes = np.array(
            [
                [[10.0, 10.0, 50.0, 50.0]],
                [[5.0, 5.0, 20.0, 20.0]],
            ],
            dtype="float32",
        )
        result = self.bb.pad(boxes, top=10, left=5)
        expected = np.array(
            [
                [[15.0, 20.0, 55.0, 60.0]],
                [[10.0, 15.0, 25.0, 30.0]],
            ],
            dtype="float32",
        )
        self.assertAllClose(result, expected)


if __name__ == "__main__":
    testing.run_tests()
