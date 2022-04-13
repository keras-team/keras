"""Tests for dataset_utils."""

import tensorflow.compat.v2 as tf

from keras.utils import dataset_utils


class TestSplitDataset(tf.test.TestCase):
  
  def test_invalid_dataset_cases(self):
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(dataset=None, left_size=5)
      
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(dataset=1, left_size=5)
      
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(dataset=float(1.2), left_size=5)
      
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(dataset=dict({}))
      
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(dataset=float('INF'))

  def test_valid_left_size_cases(self):
    
    dataset = [1,2,3]
    splitted_dataset = dataset_utils.split_dataset(dataset, left_size=1,right_size=2)
    assert(len(splitted_dataset) == 2)
    left_dataset,right_dataset = splitted_dataset
    self.assertEqual(len(left_dataset), 1)
    self.assertEqual(len(right_dataset), 2)
    self.assertEqual(list(left_dataset), [1])
    self.assertEqual(list(right_dataset), [2,3])
     
    
  def test_invalid_left_and_right_case(self):
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset(dataset=[1,2,3], left_size=None)
      
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset([1,2,3], left_size=None,right_size=None)
      
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset([1,2,3], left_size=3,right_size=None)
    



if __name__ == "__main__":
  tf.test.main()
