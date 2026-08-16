import tensorflow as tf

from keras.src.backend.tensorflow.core import name_scope
from keras.src.testing import TestCase


class TFNameScopeTest(TestCase):
    def test_stacking(self):
        self.assertEqual(tf.Variable(0, name="x").name, "x:0")
        with name_scope("outer") as outer:
            self.assertEqual(outer.name, "outer")
            self.assertEqual(tf.Variable(0, name="x").name, "outer/x:0")
            with name_scope("middle") as middle:
                self.assertEqual(middle.name, "middle")
                self.assertEqual(
                    tf.Variable(0, name="x").name, "outer/middle/x:0"
                )
                with name_scope("inner") as inner:
                    self.assertEqual(inner.name, "inner")
                    self.assertEqual(
                        tf.Variable(0, name="x").name, "outer/middle/inner/x:0"
                    )
                self.assertEqual(
                    tf.Variable(0, name="x").name, "outer/middle/x:0"
                )
            self.assertEqual(tf.Variable(0, name="x").name, "outer/x:0")
        self.assertEqual(tf.Variable(0, name="x").name, "x:0")

    def test_deduplicate(self):
        self.assertEqual(tf.Variable(0, name="x").name, "x:0")
        with name_scope("name", caller=1):
            with name_scope("name", caller=1):
                self.assertEqual(tf.Variable(0, name="x").name, "name/x:0")
        self.assertEqual(tf.Variable(0, name="x").name, "x:0")
        with name_scope("name"):
            with name_scope("name"):
                self.assertEqual(tf.Variable(0, name="x").name, "name/name/x:0")
