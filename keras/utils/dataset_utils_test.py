"""Tests for dataset_utils."""

import tensorflow.compat.v2 as tf

import numpy as np

from keras.utils import dataset_utils


class SplitDatasetTest(tf.test.TestCase):
  
  def test_with_basic_dataset_values(self):
    # numpy array
    dataset=np.ones(shape=(200, 32))
    res = dataset_utils.split_dataset(dataset, left_size=0.8,right_size=0.2)
    self.assertLen(res, 2)
    left_split,right_split = res
    self.assertIsInstance(left_split, tf.data.Dataset)
    self.assertIsInstance(right_split, tf.data.Dataset)
    
    self.assertLen(left_split, 160)
    self.assertLen(right_split, 40)
    self.assertAllEqual(dataset[:160] ,list(left_split))
    self.assertAllEqual(dataset[-40:] ,list(right_split))
    
    
    # list of numpy array
    dataset=[np.ones(shape=(200, 32)), np.zeros(shape=(200, 32))]
    res = dataset_utils.split_dataset(dataset, left_size=4)
    self.assertLen(res, 2)
    left_split,right_split = res
    self.assertIsInstance(left_split, tf.data.Dataset)
    self.assertIsInstance(right_split, tf.data.Dataset)
    self.assertLen(left_split, 4)
    self.assertLen(right_split, 196)
    self.assertAllEqual(list(zip(*dataset))[:4] ,list(left_split))
    self.assertAllEqual(list(zip(*dataset))[4:] ,list(right_split)) 
    self.assertAllEqual(list(left_split)+list(right_split), 
                        list(zip(*dataset)))
    
    
    # fail with invalid shape
    with self.assertRaises(ValueError):
      dataset=[np.ones(shape=(200, 32)), np.zeros(shape=(100, 32))]
      dataset_utils.split_dataset(dataset, left_size=4)
      
    with self.assertRaises(ValueError):
      dataset=(np.ones(shape=(200, 32)), np.zeros(shape=(201, 32)))
      dataset_utils.split_dataset(dataset, left_size=4)
    
    # list with single np array
    dataset=[np.ones(shape=(200, 32))]
    left_split,right_split = dataset_utils.split_dataset(dataset, 
                                                             left_size=4)
    self.assertAllEqual(list(zip(*dataset)),
                        list(left_split)+list(right_split))
    
    # Tule of np arrays
    dataset=(np.ones(shape=(200, 32)), np.zeros(shape=(200, 32)))
    left_split,right_split = dataset_utils.split_dataset(dataset, 
                                                             left_size=80)
    self.assertAllEqual(list(zip(*dataset))[:80] ,list(left_split))
    self.assertAllEqual(list(zip(*dataset))[80:] ,list(right_split))
    
    
    # Batched tf.data.Dataset that yields batches of vectors
    dataset = tf.data.Dataset.from_tensor_slices(np.ones(shape=(16, 16,3)))
    dataset = dataset.batch(10)
    left_split,right_split=dataset_utils.split_dataset(dataset,left_size=2)
    self.assertAllEqual(list(dataset.unbatch()),
                        list(left_split)+list(right_split)) 
    
    # Batched tf.data.Dataset that yields batches of tuples of vectors
    dataset = tf.data.Dataset.from_tensor_slices((np.random.rand(32,32), np.random.rand(32,32)))
    dataset = dataset.batch(2)
    left_split,right_split=dataset_utils.split_dataset(dataset,left_size=5)
    self.assertAllEqual(list(dataset.unbatch()),
                        list(left_split)+list(right_split))  
    


    
    
    
    
  def test_with_list_dataset(self):
    dataset = [np.ones(shape=(10,10,10)) for _ in range(10)]
    left_split,right_split = dataset_utils.split_dataset(dataset,
                                                             left_size=5,
                                                             right_size=5)
    self.assertEqual(len(left_split), len(right_split))
    self.assertIsInstance(left_split, tf.data.Dataset)
    self.assertIsInstance(left_split, tf.data.Dataset)
        
    dataset = [np.ones(shape=(10,10,10)) for _ in range(10)]
    left_split,right_split = dataset_utils.split_dataset(dataset,
                                                             left_size=0.6,
                                                             right_size=0.4)
    self.assertEqual(len(left_split), 6)
    self.assertEqual(len(right_split), 4)

    
  def test_with_tuple_dataset(self):
    dataset = (np.ones(shape=(10,10,10)),np.zeros(shape=(10,10,10)))
    left_split,right_split = dataset_utils.split_dataset(dataset,
                                                             left_size=0.75,
                                                             right_size=0.25)
    self.assertLen(left_split, 8)
    self.assertLen(right_split, 2)
    
    left_split,right_split = dataset_utils.split_dataset(dataset,
                                                             left_size=0.35,
                                                             right_size=0.65)
    self.assertLen(left_split, 4)
    self.assertLen(right_split, 6)
    self.assertIsInstance(left_split, tf.data.Dataset)
    self.assertIsInstance(right_split, tf.data.Dataset)
    
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
    
    dataset = np.array([1,2,3])
    splitted_dataset = dataset_utils.split_dataset(dataset, 
                                                   left_size=1,
                                                   right_size=2)
    assert(len(splitted_dataset) == 2)
    left_split,right_split = splitted_dataset
    self.assertEqual(len(left_split), 1)
    self.assertEqual(len(right_split), 2)
    self.assertEqual(list(left_split), [1])
    self.assertEqual(list(right_split), [2,3])
    
    
    dataset=np.ones(shape=(200, 32))
    res = dataset_utils.split_dataset(dataset, left_size=150,right_size=50)
    self.assertLen(res, 2)
    self.assertIsInstance(res[0], tf.data.Dataset)
    self.assertIsInstance(res[1], tf.data.Dataset)
    
    self.assertLen(res[0], 150)
    self.assertLen(res[1], 50)
    
    dataset=np.ones(shape=(200, 32))
    res = dataset_utils.split_dataset(dataset, left_size=120)
    self.assertLen(res, 2)
    self.assertIsInstance(res[0], tf.data.Dataset)
    self.assertIsInstance(res[1], tf.data.Dataset)
    
    self.assertLen(res[0], 120)
    self.assertLen(res[1], 80)
    
    
    dataset=np.ones(shape=(10000, 16))
    res = dataset_utils.split_dataset(dataset, right_size=20)
    self.assertLen(res, 2)
    self.assertIsInstance(res[0], tf.data.Dataset)
    self.assertIsInstance(res[1], tf.data.Dataset)
    
    self.assertLen(res[0], 9980)
    self.assertLen(res[1], 20)
    
    
    dataset = np.array([1,2,3,4,5,6,7,8,9,10])
    splitted_dataset = dataset_utils.split_dataset(dataset,
                                                   left_size=0.1,
                                                   right_size=0.9)
    assert(len(splitted_dataset) == 2)
    left_split,right_split = splitted_dataset
    self.assertEqual(len(left_split), 1 )
    self.assertEqual(len(right_split), 9 )
    self.assertEqual(list(left_split), [1])
    self.assertEqual(list(right_split), [2,3,4,5,6,7,8,9,10])
    
    dataset = np.array([1,2,3,4,5,6,7,8,9,10])
    splitted_dataset = dataset_utils.split_dataset(dataset,
                                                   left_size=2,
                                                   right_size=5)
    assert(len(splitted_dataset) == 2)
    left_split,right_split = splitted_dataset
    self.assertEqual(len(left_split), 2 )
    self.assertEqual(len(right_split), 5 )
    self.assertEqual(list(left_split), [1,2])
    self.assertEqual(list(right_split), [6,7,8,9,10])
    
  def test_with_float_left_and_right_sizes(self):
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[0.1,0.2,0.3],
                                                           [0.4,0.5,0.6],
                                                           [0.7,0.8,0.9]]))
    left_split,right_split = dataset_utils.split_dataset(dataset, 
                                                             left_size=0.8,
                                                             right_size=0.2)
    self.assertEqual(len(left_split), 2)
    self.assertEqual(len(right_split), 1)
    
  def test_with_invalid_float_left_and_right_sizes(self):
    with self.assertRaises(ValueError):
      dataset = [np.ones(shape=(200, 32,32)), np.zeros(shape=(200, 32,32))]
      dataset_utils.split_dataset(dataset, left_size=1.5,right_size=0.2)
    with self.assertRaises(ValueError):
      dataset = [1]
      dataset_utils.split_dataset(dataset, left_size=0.8,right_size=0.2)

    
      
  def test_with_None_and_zero_left_and_right_size(self):    
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset(dataset=np.array([1,2,3]), left_size=None)  
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset(np.array([1,2,3]), 
                                  left_size=None,
                                  right_size=None)
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset(np.array([1,2,3]), 
                                  left_size=3,
                                  right_size=None)
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset(np.array([1,2,3]), 
                                  left_size=3,
                                  right_size=None)
    with self.assertRaises(ValueError):
      dataset_utils.split_dataset(np.array([1,2,3]), left_size=0,right_size=0)
      
  def test_with_invalid_left_and_right_size_types(self):      
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(np.array([1,2,3]), 
                                  left_size='1',
                                  right_size='1')
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(np.array([1,2,3]), 
                                  left_size=0,
                                  right_size='1')
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(np.array([1,2,3]), 
                                  left_size='100',
                                  right_size=None)
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(np.array([1,2,3]),
                                  right_size='1')
    with self.assertRaises(TypeError):
      dataset_utils.split_dataset(np.array([1,2,3]), 
                                  left_size=0.5,
                                  right_size='1')
      
    



if __name__ == "__main__":
  tf.test.main()
