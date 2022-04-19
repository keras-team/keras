"""Tests for Dataset Utils"""

import tensorflow.compat.v2 as tf
# pylint: disable=g-classes-have-attributes

import numpy as np
from keras.utils import dataset_utils
from keras.datasets import mnist

class SplitDatasetTest(tf.test.TestCase):
  def test_numpy_array(self):
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

  def test_list_of_numpy_arrays(self):
    # test with list of np arrays with same shapes
    dataset=[np.ones(shape=(200, 32)), np.zeros(shape=(200, 32))]
    res = dataset_utils.split_dataset(dataset, left_size=4)

    self.assertLen(res, 2)
    left_split,right_split = res

    self.assertIsInstance(left_split, tf.data.Dataset)
    self.assertIsInstance(right_split, tf.data.Dataset)

    self.assertEqual(np.array(list(left_split)).shape,(4,2,32))
    self.assertEqual(np.array(list(right_split)).shape,(196,2,32))

    # test with different shapes
    dataset = [np.ones(shape=(5, 3)), np.ones(shape=(5, ))]
    left_split,right_split = dataset_utils.split_dataset(dataset,left_size=0.3)

    self.assertEqual(np.array(list(left_split)).shape,(2,2))
    self.assertEqual(np.array(list(right_split)).shape,(3,2))

    self.assertEqual(np.array(list(left_split)[0]).shape,(2,))
    self.assertEqual(np.array(list(left_split)[0][0]).shape,(3,))
    self.assertEqual(np.array(list(left_split)[0][1]).shape,())

    self.assertEqual(np.array(list(right_split)[0]).shape,(2,))
    self.assertEqual(np.array(list(right_split)[0][0]).shape,(3,))
    self.assertEqual(np.array(list(right_split)[0][1]).shape,())

  def test_dataset_with_invalid_shape(self):
    with self.assertRaisesRegex(ValueError,
                                'Received a list of numpy arrays '
                                'with different length'):
      dataset=[np.ones(shape=(200, 32)), np.zeros(shape=(100, 32))]
      dataset_utils.split_dataset(dataset, left_size=4)

    with self.assertRaisesRegex(ValueError,
                                'Received a tuple of numpy arrays '
                                'with different length'):
      dataset=(np.ones(shape=(200, 32)), np.zeros(shape=(201, 32)))
      dataset_utils.split_dataset(dataset, left_size=4)

  def test_tuple_of_numpy_arrays(self):
    dataset=(np.random.rand(4, 3), np.random.rand(4, 3))
    left_split,right_split = dataset_utils.split_dataset(dataset, left_size=2)

    self.assertIsInstance(left_split, tf.data.Dataset)
    self.assertIsInstance(right_split, tf.data.Dataset)

    self.assertEqual(len(left_split), 2)
    self.assertEqual(len(right_split), 2)

    self.assertEqual(np.array(list(left_split)[0]).shape, (2, 3))
    self.assertEqual(np.array(list(left_split)[1]).shape, (2, 3))

    # test with fractional size
    dataset = (np.random.rand(5, 32,32), np.random.rand(5, 32,32))
    left_split,right_split = dataset_utils.split_dataset(dataset,
                                                         right_size=0.4)
    self.assertIsInstance(left_split, tf.data.Dataset)
    self.assertIsInstance(right_split, tf.data.Dataset)

    self.assertEqual(np.array(list(left_split)).shape,(3,2,32,32))
    self.assertEqual(np.array(list(right_split)).shape,(2,2,32,32))

    self.assertEqual(np.array(list(left_split))[0].shape,(2,32,32))
    self.assertEqual(np.array(list(left_split))[1].shape,(2,32,32))

    self.assertEqual(np.array(list(right_split))[0].shape,(2,32,32))
    self.assertEqual(np.array(list(right_split))[1].shape,(2,32,32))

    # test with tuple of np arrays with different shapes
    dataset = (np.random.rand(5, 32,32), np.random.rand(5, ))
    left_split,right_split = dataset_utils.split_dataset(dataset,
                                                         left_size=2,
                                                         right_size=3)
    self.assertIsInstance(left_split, tf.data.Dataset)
    self.assertIsInstance(right_split, tf.data.Dataset)

    self.assertEqual(np.array(list(left_split)).shape,(2,2))
    self.assertEqual(np.array(list(right_split)).shape,(3,2))

    self.assertEqual(np.array(list(left_split)[0]).shape,(2,))
    self.assertEqual(np.array(list(left_split)[0][0]).shape,(32,32))
    self.assertEqual(np.array(list(left_split)[0][1]).shape,())

    self.assertEqual(np.array(list(right_split)[0]).shape,(2,))
    self.assertEqual(np.array(list(right_split)[0][0]).shape,(32,32))
    self.assertEqual(np.array(list(right_split)[0][1]).shape,())

  def test_batched_tf_dataset_of_vectors(self):
    dataset = tf.data.Dataset.from_tensor_slices(np.ones(shape=(100,32, 32,1)))
    dataset = dataset.batch(10)
    left_split,right_split=dataset_utils.split_dataset(dataset,left_size=2)

    # Ensure that the splits are batched
    self.assertAllEqual(np.array(list(right_split)).shape,(10,))

    left_split,right_split = left_split.unbatch(),right_split.unbatch()
    self.assertAllEqual(np.array(list(left_split)).shape,(2,32,32,1))
    self.assertAllEqual(np.array(list(right_split)).shape,(98,32,32,1))
    dataset = dataset.unbatch()
    self.assertAllEqual(list(dataset),list(left_split)+list(right_split))

  def test_batched_tf_dataset_of_tuple_of_vectors(self):
    dataset = tf.data.Dataset.from_tensor_slices((np.random.rand(10,32,32),
                                                  np.random.rand(10,32,32)))
    dataset = dataset.batch(2)
    left_split,right_split=dataset_utils.split_dataset(dataset,left_size=4)

    # Ensure that the splits are batched
    self.assertEqual(np.array(list(right_split)).shape,(3, 2, 2, 32, 32))
    self.assertEqual(np.array(list(left_split)).shape,(2, 2, 2, 32, 32))

    left_split,right_split = left_split.unbatch(),right_split.unbatch()
    self.assertAllEqual(np.array(list(left_split)).shape,(4,2,32,32))
    self.assertAllEqual(np.array(list(right_split)).shape,(6,2,32,32))

    dataset = dataset.unbatch()
    self.assertAllEqual(list(dataset),list(left_split)+list(right_split))

  def test_unbatched_tf_dataset_of_vectors(self):
    dataset = tf.data.Dataset.from_tensor_slices(np.ones(shape=(100,16, 16,3)))

    left_split,right_split=dataset_utils.split_dataset(dataset,left_size=0.25)

    self.assertAllEqual(np.array(list(left_split)).shape,(25,16, 16,3))
    self.assertAllEqual(np.array(list(right_split)).shape,(75,16, 16,3))

    self.assertAllEqual(list(dataset),list(left_split)+list(right_split))

  def test_unbatched_tf_dataset_of_tuple_of_vectors(self):
    X,Y = (np.random.rand(10,32,32,1),np.random.rand(10,32,32,1))
    dataset = tf.data.Dataset.from_tensor_slices((X,Y))

    left_split,right_split=dataset_utils.split_dataset(dataset,left_size=5)

    self.assertAllEqual(np.array(list(left_split)).shape,(5,2,32,32,1))
    self.assertAllEqual(np.array(list(right_split)).shape,(5,2,32,32,1))

    self.assertAllEqual(list(dataset),list(left_split)+list(right_split))

  def test_unbatched_tf_dataset_of_dict_of_vectors(self):
    # test with dict of np arrays of same shape
    dict_samples = {'X':np.random.rand(10,2),
                    'Y':np.random.rand(10,2)}
    dataset = tf.data.Dataset.from_tensor_slices(dict_samples)
    left_split,right_split=dataset_utils.split_dataset(dataset,left_size=2)
    self.assertEqual(len(list(left_split)),2)
    self.assertEqual(len(list(right_split)),8)
    for i in range(10):
      if i < 2:
        self.assertEqual(list(left_split)[i],list(dataset)[i])
      else:
        self.assertEqual(list(right_split)[i-2],list(dataset)[i])

    # test with dict of np arrays with different shapes
    dict_samples = {'images':np.random.rand(10,16,16,3),
                    'labels':np.random.rand(10,)}
    dataset = tf.data.Dataset.from_tensor_slices(dict_samples)
    left_split,right_split=dataset_utils.split_dataset(dataset,left_size=0.3)
    self.assertEqual(len(list(left_split)),3)
    self.assertEqual(len(list(right_split)),7)
    for i in range(10):
      if i < 3:
        self.assertEqual(list(left_split)[i],list(dataset)[i])
      else:
        self.assertEqual(list(right_split)[i-3],list(dataset)[i])

  def test_batched_tf_dataset_of_dict_of_vectors(self):
    dict_samples = {'X':np.random.rand(10,3),
                    'Y':np.random.rand(10,3)}
    dataset = tf.data.Dataset.from_tensor_slices(dict_samples)
    dataset = dataset.batch(2)
    left_split,right_split=dataset_utils.split_dataset(dataset,left_size=2)

    self.assertAllEqual(np.array(list(left_split)).shape,(1,))
    self.assertAllEqual(np.array(list(right_split)).shape,(4,))

    left_split,right_split = left_split.unbatch(),right_split.unbatch()
    self.assertEqual(len(list(left_split)),2)
    self.assertEqual(len(list(right_split)),8)
    for i in range(10):
      if i < 2:
        self.assertEqual(list(left_split)[i],list(dataset.unbatch())[i])
      else:
        self.assertEqual(list(right_split)[i-2],list(dataset.unbatch())[i])

    # test with dict of np arrays with different shapes
    dict_samples = {'images':np.random.rand(10,16,16,3),
                    'labels':np.random.rand(10,)}
    dataset = tf.data.Dataset.from_tensor_slices(dict_samples)
    dataset = dataset.batch(1)
    left_split,right_split=dataset_utils.split_dataset(dataset,right_size=0.3)

    self.assertAllEqual(np.array(list(left_split)).shape,(7,))
    self.assertAllEqual(np.array(list(right_split)).shape,(3,))

    dataset = dataset.unbatch()
    left_split,right_split = left_split.unbatch(),right_split.unbatch()
    self.assertEqual(len(list(left_split)),7)
    self.assertEqual(len(list(right_split)),3)
    for i in range(10):
      if i < 7:
        self.assertEqual(list(left_split)[i],list(dataset)[i])
      else:
        self.assertEqual(list(right_split)[i-7],list(dataset)[i])

  def test_list_dataset(self):
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

  def test_invalid_dataset(self):
    with self.assertRaisesRegex(TypeError,
                                '`dataset` must be either a tf.data.Dataset '
                               f'object or a list/tuple of arrays. Received '
                                ': <class \'NoneType\'>'):
      dataset_utils.split_dataset(dataset=None, left_size=5)
    with self.assertRaisesRegex(TypeError,
                                '`dataset` must be either a tf.data.Dataset '
                               f'object or a list/tuple of arrays. Received '
                                ': <class \'int\'>'):
      dataset_utils.split_dataset(dataset=1, left_size=5)
    with self.assertRaisesRegex(TypeError,
                                '`dataset` must be either a tf.data.Dataset '
                               f'object or a list/tuple of arrays. Received '
                                ': <class \'float\'>'):
      dataset_utils.split_dataset(dataset=float(1.2), left_size=5)
    with self.assertRaisesRegex(TypeError,
                                '`dataset` must be either a tf.data.Dataset '
                               f'object or a list/tuple of arrays. Received '
                                ': <class \'dict\'>'):
      dataset_utils.split_dataset(dataset=dict({}), left_size=5)
    with self.assertRaisesRegex(TypeError,
                                '`dataset` must be either a tf.data.Dataset '
                               f'object or a list/tuple of arrays. Received '
                                ': <class \'float\'>'):
      dataset_utils.split_dataset(dataset=float('INF'), left_size=5)

  def test_valid_left_and_right_sizes(self):
    dataset = np.array([1,2,3])
    splitted_dataset = dataset_utils.split_dataset(dataset,1,2)
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

  def test_float_left_and_right_sizes(self):
    X = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
    dataset = tf.data.Dataset.from_tensor_slices(X)
    left_split,right_split = dataset_utils.split_dataset(dataset,
                                                         left_size=0.8,
                                                         right_size=0.2)
    self.assertEqual(len(left_split), 2)
    self.assertEqual(len(right_split), 1)

  def test_invalid_float_left_and_right_sizes(self):
    expected_regex = (r'^(.*?(\bleft_size\b).*?(\bshould be\b)'
                      r'.*?(\bwithin the range\b).*?(\b0\b).*?(\b1\b))')
    with self.assertRaisesRegexp(ValueError,expected_regex):
      dataset = [np.ones(shape=(200, 32,32)), np.zeros(shape=(200, 32,32))]
      dataset_utils.split_dataset(dataset, left_size=1.5,right_size=0.2)

    expected_regex = (r'^(.*?(\bright_size\b).*?(\bshould be\b)'
                      r'.*?(\bwithin the range\b).*?(\b0\b).*?(\b1\b))')
    with self.assertRaisesRegex(ValueError,expected_regex):
      dataset = [np.ones(shape=(200, 32)), np.zeros(shape=(200, 32))]
      dataset_utils.split_dataset(dataset, left_size=0.8,right_size=-0.8)

  def test_None_and_zero_left_and_right_size(self):
    expected_regex = (r'^.*?(\bleft_size\b).*?(\bright_size\b).*?(\bmust '
                      r'be specified\b).*?(\bReceived: left_size=None and'
                      r' right_size=None\b)')

    with self.assertRaisesRegex(ValueError,expected_regex):
      dataset_utils.split_dataset(dataset=np.array([1,2,3]), left_size=None)
    with self.assertRaisesRegex(ValueError, expected_regex):
      dataset_utils.split_dataset(np.array([1,2,3]),left_size=None,
                                  right_size=None)

    expected_regex = (r'^.*?(\bleft_size\b).*?(\bshould be\b)'
                      r'.*?(\bpositive\b).*?(\bsmaller than 3\b)')
    with self.assertRaisesRegex(ValueError,expected_regex):
      dataset_utils.split_dataset(np.array([1,2,3]),left_size=3)

    expected_regex = ('Both `left_size` and `right_size` are zero. '
                     'Atleast one of the split sizes must be non-zero.')
    with self.assertRaisesRegex(ValueError,expected_regex):
      dataset_utils.split_dataset(np.array([1,2,3]), left_size=0,
                                  right_size=0)

  def test_invalid_left_and_right_size_types(self):
    expected_regex = (r'^.*?(\bInvalid `left_size` and `right_size` Types'
                      r'\b).*?(\bExpected: integer or float or None\b)')
    with self.assertRaisesRegex(TypeError,expected_regex):
      dataset_utils.split_dataset(np.array([1,2,3]), left_size='1',
                                  right_size='1')

    expected_regex = (r'^.*?(\bInvalid `right_size` Type\b)')
    with self.assertRaisesRegex(TypeError,expected_regex):
      dataset_utils.split_dataset(np.array([1,2,3]),left_size=0,
                                  right_size='1')

    expected_regex = (r'^.*?(\bInvalid `left_size` Type\b)')
    with self.assertRaisesRegex(TypeError,expected_regex):
      dataset_utils.split_dataset(np.array([1,2,3]),left_size='100',
                                  right_size=None)

    expected_regex = (r'^.*?(\bInvalid `right_size` Type\b)')
    with self.assertRaisesRegex(TypeError,expected_regex):
      dataset_utils.split_dataset(np.array([1,2,3]),right_size='1')

    expected_regex = (r'^.*?(\bInvalid `right_size` Type\b)')
    with self.assertRaisesRegex(TypeError,expected_regex):
      dataset_utils.split_dataset(np.array([1,2,3]),left_size=0.5,
                                  right_size='1')

  def test_mnist_dataset(self):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    dataset = (x_train[:100], y_train[:100])
    left_split,right_split = dataset_utils.split_dataset(dataset,left_size=0.8)

    self.assertIsInstance(left_split, tf.data.Dataset)
    self.assertIsInstance(right_split, tf.data.Dataset)

    self.assertEqual(len(left_split), 80)
    self.assertEqual(len(right_split), 20)

if __name__ == "__main__":
  tf.test.main()
