import pytest
import tensorflow as tf

from keras.src import backend
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes import (
    validation,
)
from keras.src.testing import test_case


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="The test targets TensorFlow-specific ragged tensors.",
)
class DensifyBoundingBoxesTest(test_case.TestCase):
    def test_densify_ragged_bounding_boxes_batched(self):
        ragged_boxes = tf.ragged.constant(
            [
                [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]],
                [[0.5, 0.5, 0.6, 0.6]],
            ],
            dtype=tf.float32,
        )
        ragged_labels = tf.ragged.constant(
            [
                [0, 1],
                [2],
            ],
            dtype=tf.int32,
        )
        bounding_boxes = {"boxes": ragged_boxes, "labels": ragged_labels}
        max_boxes = 3
        densified_data = validation.densify_bounding_boxes(
            bounding_boxes.copy(), is_batched=True, max_boxes=max_boxes
        )
        densified_boxes = densified_data["boxes"]
        densified_labels = densified_data["labels"]
        self.assertEqual(densified_boxes.shape, (2, max_boxes, 4))
        self.assertEqual(densified_labels.shape, (2, max_boxes))
        expected_boxes = [
            [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.0, 0.0, 0.0, 0.0]],
            [[0.5, 0.5, 0.6, 0.6], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
        expected_labels = [
            [0, 1, -1],
            [2, -1, -1],
        ]
        self.assertAllClose(densified_boxes, expected_boxes)
        self.assertAllEqual(densified_labels, expected_labels)

    def test_densify_ragged_bounding_boxes_unbatched(self):
        ragged_boxes = tf.ragged.constant(
            [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]],
            dtype=tf.float32,
        )
        ragged_labels = tf.ragged.constant([[0], [1]], dtype=tf.int32)
        bounding_boxes = {"boxes": ragged_boxes, "labels": ragged_labels}
        max_boxes = 4
        densified_data = validation.densify_bounding_boxes(
            bounding_boxes.copy(), is_batched=False, max_boxes=max_boxes
        )
        densified_boxes = densified_data["boxes"]
        densified_labels = densified_data["labels"]

        self.assertEqual(densified_boxes.shape, (max_boxes, 4))
        self.assertEqual(densified_labels.shape, (max_boxes, 1))
        expected_boxes = [
            [0.1, 0.1, 0.2, 0.2],
            [0.3, 0.3, 0.4, 0.4],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        expected_labels = [[0], [1], [-1], [-1]]
        self.assertAllClose(densified_boxes, expected_boxes)
        self.assertAllEqual(densified_labels, expected_labels)
