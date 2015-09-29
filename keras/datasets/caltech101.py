from __future__ import absolute_import
from .data_utils import get_file
import numpy as np
import os


def load_paths(train_imgs_per_category=15, test_imgs_per_category=15, shuffle=True):
    dirname = "101_ObjectCategories"
    origin = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
    path = get_file(dirname, origin=origin, untar=True)

    X_train = np.array([])
    y_train = np.array([])

    X_test = np.array([])
    y_test = np.array([])

    # directories are the labels
    labels = sorted([d for d in os.listdir(path)])

    # loop over all subdirs
    for i, label in enumerate(labels):
        label_dir = os.path.join(path, label)
        fpaths = np.array([os.path.join(label_dir, img_fname) for img_fname in os.listdir(label_dir)])

        np.random.shuffle(fpaths)
        if train_imgs_per_category + test_imgs_per_category > len(fpaths):
            print("not enough samples for label " + label)

        X_train = np.append(X_train, fpaths[:train_imgs_per_category])
        y_train = np.append(y_train, [i for x in range(train_imgs_per_category)])

        X_test = np.append(X_test, fpaths[train_imgs_per_category:train_imgs_per_category+test_imgs_per_category])
        y_test = np.append(y_test, [i for x in range(test_imgs_per_category)])

    # shuffle/permutation
    if shuffle:
        # shuffle training data
        shuffle_index_training = np.arange(X_train.shape[0])
        np.random.shuffle(shuffle_index_training)
        X_train = X_train[shuffle_index_training]
        y_train = y_train[shuffle_index_training]

        # shuffle test data
        shuffle_index_test = np.arange(X_test.shape[0])
        np.random.shuffle(shuffle_index_test)
        X_test = X_test[shuffle_index_test]
        y_test = y_test[shuffle_index_test]

    return (X_train, y_train), (X_test, y_test)
