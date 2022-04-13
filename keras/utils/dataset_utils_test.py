"""Tests for dataset_utils."""

import tensorflow.compat.v2 as tf

import numpy as np

from keras.utils import dataset_utils


class SplitDatasetTest(tf.test.TestCase):
  
  def test_with_list_dataset(self):
    dataset = [np.ones(shape=(10,10,10)) for _ in range(10)]
    left_dataset,right_dataset = dataset_utils.split_dataset(dataset,
                                                             left_size=5,
                                                             right_size=5)
    self.assertEqual(len(left_dataset), len(right_dataset))
    self.assertIsInstance(left_dataset, tf.data.Dataset)
    self.assertIsInstance(left_dataset, tf.data.Dataset)
        
    dataset = [np.ones(shape=(10,10,10)) for _ in range(10)]
    left_dataset,right_dataset = dataset_utils.split_dataset(dataset,
                                                             left_size=0.6,
                                                             right_size=0.4)
    self.assertEqual(len(left_dataset), 6)
    self.assertEqual(len(right_dataset), 4)

    
  def test_with_tuple_dataset(self):
    dataset = (np.ones(shape=(10,10,10)),np.zeros(shape=(10,10,10)))
    left_dataset,right_dataset = dataset_utils.split_dataset(dataset,
                                                             left_size=0.75,
                                                             right_size=0.25)
    self.assertLen(left_dataset, 8)
    self.assertLen(right_dataset, 2)
    
    left_dataset,right_dataset = dataset_utils.split_dataset(dataset,
                                                             left_size=0.35,
                                                             right_size=0.65)
    self.assertLen(left_dataset, 4)
    self.assertLen(right_dataset, 6)
    self.assertIsInstance(left_dataset, tf.data.Dataset)
    self.assertIsInstance(right_dataset, tf.data.Dataset)
    
  
  def test_with_invalid_dataset(self):
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(dataset=None, left_size=5)  
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(dataset=1, left_size=5)  
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(dataset=float(1.2), left_size=5)  
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(dataset=dict({}), left_size=5) 
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(dataset=float('INF'), left_size=5)

  def test_with_valid_left_and_right_sizes(self):
    
    dataset = [1,2,3]
    splitted_dataset = dataset_utils.split_dataset(dataset, 
                                                   left_size=1,
                                                   right_size=2)
    assert(len(splitted_dataset) == 2)
    left_dataset,right_dataset = splitted_dataset
    self.assertEqual(len(left_dataset), 1)
    self.assertEqual(len(right_dataset), 2)
    self.assertEqual(list(left_dataset), [1])
    self.assertEqual(list(right_dataset), [2,3])
    
    
    dataset = [1,2,3,4,5,6,7,8,9,10]
    splitted_dataset = dataset_utils.split_dataset(dataset,
                                                   left_size=0.1,
                                                   right_size=0.9)
    assert(len(splitted_dataset) == 2)
    left_dataset,right_dataset = splitted_dataset
    self.assertEqual(len(left_dataset), 1 )
    self.assertEqual(len(right_dataset), 9 )
    self.assertEqual(list(left_dataset), [1])
    self.assertEqual(list(right_dataset), [2,3,4,5,6,7,8,9,10])
    
    dataset = [1,2,3,4,5,6,7,8,9,10]
    splitted_dataset = dataset_utils.split_dataset(dataset,
                                                   left_size=2,
                                                   right_size=5)
    assert(len(splitted_dataset) == 2)
    left_dataset,right_dataset = splitted_dataset
    self.assertEqual(len(left_dataset), 2 )
    self.assertEqual(len(right_dataset), 5 )
    self.assertEqual(list(left_dataset), [1,2])
    self.assertEqual(list(right_dataset), [6,7,8,9,10])
    
  def test_with_float_left_and_right_sizes(self):
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[0.1,0.2,0.3],
                                                           [0.4,0.5,0.6],
                                                           [0.7,0.8,0.9]]))
    left_dataset,right_dataset = dataset_utils.split_dataset(dataset, 
                                                             left_size=0.8,
                                                             right_size=0.2)
    self.assertEqual(len(left_dataset), 2)
    self.assertEqual(len(right_dataset), 1)
    
  def test_with_invalid_float_left_and_right_sizes(self):
    with self.assertRaises(ValueError):
      dataset = [np.ones(shape=(200, 32,32)), np.zeros(shape=(200, 32,32))]
      dataset_utils.split_dataset(dataset, left_size=0.8,right_size=0.2)
    with self.assertRaises(ValueError):
      dataset = [1]
      dataset_utils.split_dataset(dataset, left_size=0.8,right_size=0.2)

    
      
  def test_with_None_and_zero_left_and_right_size(self):    
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset(dataset=[1,2,3], left_size=None)  
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset([1,2,3], left_size=None,right_size=None)
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset([1,2,3], left_size=3,right_size=None)
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset([1,2], left_size=3,right_size=None)
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset([1,2], left_size=0,right_size=0)
      
  def test_with_invalid_left_and_right_size_types(self):      
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset([1,2], left_size='1',right_size='1')
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset([1,2], left_size=0,right_size='1')
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset([1,2], left_size='100',right_size=None)
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset([1,2], right_size='1')
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset([1,2], left_size=0.5,right_size='1')
      
    



if __name__ == "__main__":
  tf.test.main()
