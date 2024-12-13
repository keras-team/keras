import itertools

import numpy as np
from absl.testing import parameterized

from keras.src import ops
from keras.src import testing
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    affine_transform,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    clip_to_image_size,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    convert_format,
)


class ConvertersTest(testing.TestCase):
    def setUp(self):
        xyxy_box = np.array(
            [[[10, 20, 110, 120], [20, 30, 120, 130]]], dtype="float32"
        )
        yxyx_box = np.array(
            [[[20, 10, 120, 110], [30, 20, 130, 120]]], dtype="float32"
        )
        rel_xyxy_box = np.array(
            [[[0.01, 0.02, 0.11, 0.12], [0.02, 0.03, 0.12, 0.13]]],
            dtype="float32",
        )
        rel_yxyx_box = np.array(
            [[[0.02, 0.01, 0.12, 0.11], [0.03, 0.02, 0.13, 0.12]]],
            dtype="float32",
        )
        center_xywh_box = np.array(
            [[[60, 70, 100, 100], [70, 80, 100, 100]]], dtype="float32"
        )
        center_yxhw_box = np.array(
            [[[70, 60, 100, 100], [80, 70, 100, 100]]], dtype="float32"
        )
        xywh_box = np.array(
            [[[10, 20, 100, 100], [20, 30, 100, 100]]], dtype="float32"
        )
        rel_xywh_box = np.array(
            [[[0.01, 0.02, 0.1, 0.1], [0.02, 0.03, 0.1, 0.1]]], dtype="float32"
        )

        self.images = np.ones([2, 1000, 1000, 3], dtype="float32")
        self.height = 1000
        self.width = 1000

        self.boxes = {
            "xyxy": xyxy_box,
            "center_xywh": center_xywh_box,
            "rel_xywh": rel_xywh_box,
            "xywh": xywh_box,
            "rel_xyxy": rel_xyxy_box,
            "yxyx": yxyx_box,
            "rel_yxyx": rel_yxyx_box,
            "center_yxhw": center_yxhw_box,
        }

    @parameterized.named_parameters(
        *[
            (f"{source}_{target}", source, target)
            for (source, target) in itertools.permutations(
                [
                    "xyxy",
                    "yxyx",
                    "xywh",
                    "rel_xyxy",
                    "rel_yxyx",
                    "center_xywh",
                    "center_yxhw",
                ],
                2,
            )
        ]
        + [("xyxy_xyxy", "xyxy", "xyxy")]
    )
    def test_convert_all_formats(self, source, target):
        source_box = self.boxes[source]
        target_box = self.boxes[target]
        self.assertAllClose(
            convert_format(
                source_box,
                source=source,
                target=target,
                height=self.height,
                width=self.width,
            ),
            target_box,
        )

    def test_convert_format_invalid_source(self):
        boxes = self.boxes["xywh"]
        with self.assertRaises(ValueError):
            convert_format(boxes, source="invalid", target="xywh")

    def test_convert_format_invalid_target(self):
        boxes = self.boxes["xyxy"]
        with self.assertRaises(ValueError):
            convert_format(boxes, source="xyxy", target="invalid")

    def test_convert_format_missing_dimensions(self):
        boxes = self.boxes["xyxy"]
        with self.assertRaisesRegex(
            ValueError, r"must receive `height` and `width`"
        ):
            convert_format(boxes, source="xyxy", target="rel_xyxy")

    def test_clip_to_image_size(self):
        boxes = {
            "boxes": np.array([[0.0, 0.0, 1.5, 1.6], [0.5, 0.4, 0.7, 0.8]]),
            "labels": np.array([0, 1]),
        }

        expected_clipped = {
            "boxes": np.array([[0.0, 0.0, 1.0, 1.0], [0.5, 0.4, 0.7, 0.8]]),
            "labels": np.array([0, 1]),
        }

        clipped_boxes = clip_to_image_size(
            boxes, bounding_box_format="rel_xyxy"
        )

        self.assertAllEqual(clipped_boxes, expected_clipped)

    def test_affine_identity(self):
        # Test identity transform (no change)
        batch_size = self.boxes["xyxy"].shape[0]
        transformed_boxes = affine_transform(
            boxes=self.boxes["xyxy"],
            angle=np.zeros([batch_size], dtype="float32"),
            translate_x=np.zeros([batch_size], dtype="float32"),
            translate_y=np.zeros([batch_size], dtype="float32"),
            scale=np.ones([batch_size], dtype="float32"),
            shear_x=np.zeros([batch_size], dtype="float32"),
            shear_y=np.zeros([batch_size], dtype="float32"),
            height=self.height,
            width=self.width,
        )
        transformed_boxes = ops.convert_to_numpy(transformed_boxes)
        self.assertAllClose(self.boxes["xyxy"], transformed_boxes)
