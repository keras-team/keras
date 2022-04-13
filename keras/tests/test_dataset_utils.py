# ==============================================================================
"""Tests for dataset_utils.py."""
import tensorflow.compat.v2 as tf

import numpy as np

import keras

from keras.utils import dataset_utils


class DatasetUtilstest(tf.test.TestCase):
    def datasetcheck(self):
        with self.assertRaises(Exception):
            dataset_utils.split_dataset(
                dataset=None, left_size=0.8, right_size=0.2)
        with self.asserRaises(Exception):
            dataset_utils.split_dataset(dataset=None, left_size=0.7)
        with self.asserRaises(Exception):
            dataset_utils.split_dataset(dataset=None, right_size=0.5)
        test_dict = dict()
        with self.asserRaises(Exception):
            dataset_utils.split_dataset(test_dict, left_size=0.7)
        with self.asserRaises(Exception):
            dataset_utils.split_dataset(5, left_size=0.7)
        with self.asserRaises(Exception):
            dataset_utils.split_dataset('file_path', left_size=0.7)
        with self.asserRaises(Exception):
            dataset_utils.split_dataset(0.6, left_size=0.7)

    def sizecheck(self):
        data = [10, 20, 30, 40]
        with self.asserRaises(Exception):
            dataset_utils.split_dataset(data, left_size=0.7, right_size=1.5)
        with self.asserRaises(Exception):
            dataset_utils.split_dataset(data)
        with self.asserRaises(Exception):
            dataset_utils.split_dataset(data, left_size=5)
        with self.asserRaises(Exception):
            dataset_utils.split_dataset(data, left_size=2, right_size=4)
        with self.asserRaises(Exception):
            dataset_utils.split_dataset(data, left_size=1.2)
        with self.asserRaises(Exception):
            dataset_utils.split_dataset(data, left_size=0.4, right_size=0.8)


if __name__ == '__main__':
    tf.test.main()
