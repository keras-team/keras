__author__ = 'Jake Varley'

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def load_data(test_split=0.2, dataset_size=5000, patch_size=32):
    """
    Description
    -----------
    creates a dataset with total "dataset_size" samples.
    Class of a sample (sphere, and cube) is chosen at random with equal probability.
    Based on the "test_split", the dataset is divided in test and train subsets.
    The "patch_size" defines the size of a 3D array for storing voxel.


    Output shape
    ------------                                size
    (4D array, 1D array), (4D array, 1D array) ====> (Train_Voxels, Train_Lables), (Test_Voxels, Test_Labels)
        Train and test split of total 'dataset_size' voxels with labels

    Arguments
    ------------
    test_split: float
        percentage of total samples for training

    dataset_size: int
        total number of samples

    patch_size:
        size of each dimension of a 3D array to store voxel
    """

    if patch_size < 10:
       raise NotImplementedError

    num_labels = 2
        
    # Using same probability for each class
    geometry_types = np.random.randint(0, num_labels, dataset_size)
    random.shuffle(geometry_types)

    # Getting the training set
    y_train = geometry_types[0:abs((1-test_split)*dataset_size)]
    x_train = __generate_solid_figures(geometry_types=y_train, patch_size=patch_size)

    # Getting the testing set
    y_test = geometry_types[abs((1-test_split)*dataset_size):]
    x_test = __generate_solid_figures(geometry_types=y_test, patch_size=patch_size)

    return (x_train, y_train),(x_test, y_test)

def __generate_solid_figures(geometry_types, patch_size):

    """
    Output shape
    ------------
    4D array (samples, patch_size(Z), patch_size(X), patch_size(Y))
        Voxel for each label passed as input through geometry_types

    Arguments
        geometry_types: numpy array (samples, 1)
             An array of class labels (0 for sphere, 1 for cube)
        patch_size: int
             Size of 3d array to store voxel

    """
    shapes_no = geometry_types.shape[0]

    # Assuming data is centered
    (x0, y0, z0) = ((patch_size-1)/2,)*3

    # Allocate 3D data array, data is in cube(all dimensions are same)
    solid_figures = np.zeros((len(geometry_types), 1, patch_size,
                                  patch_size, patch_size))
    for i in range(0, len(geometry_types)):
        # # radius is a random number in [3, self.patch_size/2)
        radius = (patch_size/2 - 3) * np.random.rand() + 3

        # bounding box values for optimization
        x_min = int(max(math.ceil(x0-radius), 0))
        y_min = int(max(math.ceil(y0-radius), 0))
        z_min = int(max(math.ceil(z0-radius), 0))
        x_max = int(min(math.floor(x0+radius), patch_size-1))
        y_max = int(min(math.floor(y0+radius), patch_size-1))
        z_max = int(min(math.floor(z0+radius), patch_size-1))

        if geometry_types[i] == 0: #Sphere
            radius_squared = radius**2
            for z in xrange(z_min, z_max+1):
                for x in xrange(x_min, x_max+1):
                    for y in xrange(y_min, y_max+1):
                        if (x-x0)**2 + (y-y0)**2 + (z-z0)**2 <= radius_squared:
                            # inside the sphere
                            solid_figures[i, 0, z, x, y] = 1
        elif geometry_types[i] == 1: #Cube
            solid_figures[i, 0, z_min:z_max+1, x_min:x_max+1, y_min:y_max+1] = 1
        else:
            raise NotImplementedError

    return solid_figures
