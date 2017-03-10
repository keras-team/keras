# -*- coding: utf-8 -*-

"""Utility file for the HASYv2 dataset.

See https://arxiv.org/abs/1701.08380 for details.
"""

from __future__ import absolute_import
from ..utils.data_utils import get_file
from .. import backend as K
import numpy as np
import scipy.ndimage
import os
import tarfile
import shutil
import csv
from six.moves import cPickle as pickle


n_classes = 369
labels = []


def _load_csv(filepath, delimiter=',', quotechar="'"):
    """
    Load a CSV file.

    Parameters
    ----------
    filepath : str
        Path to a CSV file
    delimiter : str, optional
    quotechar : str, optional

    Returns
    -------
    list of dicts : Each line of the CSV file is one element of the list.
    """
    data = []
    csv_dir = os.path.dirname(filepath)
    with open(filepath, 'rb') as csvfile:
        reader = csv.DictReader(csvfile,
                                delimiter=delimiter,
                                quotechar=quotechar)
        for row in reader:
            for el in ['path', 'path1', 'path2']:
                if el in row:
                    row[el] = os.path.abspath(os.path.join(csv_dir, row[el]))
            data.append(row)
    return data


def _generate_index(csv_filepath):
    """
    Generate an index 0...k for the k labels.

    Parameters
    ----------
    csv_filepath : str
        Path to 'test.csv' or 'train.csv'

    Returns
    -------
    dict : Maps a symbol_id as in test.csv and
        train.csv to an integer in 0...k, where k is the total
        number of unique labels.
    """
    symbol_id2index = {}
    data = _load_csv(csv_filepath)
    i = 0
    labels = []
    for item in data:
        if item['symbol_id'] not in symbol_id2index:
            symbol_id2index[item['symbol_id']] = i
            labels.append(item['latex'])
            i += 1
    return symbol_id2index, labels


def load_data():
    """Loads HASYv2 dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    # Download if not already done
    fname = 'HASYv2.tar.bz2'
    origin = 'https://zenodo.org/record/259444/files/HASYv2.tar.bz2'
    fpath = get_file(fname, origin=origin, untar=False,
                     md5_hash='fddf23f36e24b5236f6b3a0880c778e3')
    path = os.path.dirname(fpath)

    # Extract content if not already done
    untar_fpath = os.path.join(path, "HASYv2")
    if not os.path.exists(untar_fpath):
        print('Untaring file...')
        tfile = tarfile.open(fpath, 'r:bz2')
        try:
            tfile.extractall(path=untar_fpath)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(untar_fpath):
                if os.path.isfile(untar_fpath):
                    os.remove(untar_fpath)
                else:
                    shutil.rmtree(untar_fpath)
            raise
        tfile.close()

    # Create pickle if not already done
    pickle_fpath = os.path.join(untar_fpath, "fold1.pickle")
    if not os.path.exists(pickle_fpath):
        # Load mapping from symbol names to indices
        symbol_csv_fpath = os.path.join(untar_fpath, "symbols.csv")
        symbol_id2index, labels = _generate_index(symbol_csv_fpath)
        globals()["labels"] = labels

        # Load first fold
        fold_dir = os.path.join(untar_fpath, "classification-task/fold-1")
        train_csv_fpath = os.path.join(fold_dir, "train.csv")
        test_csv_fpath = os.path.join(fold_dir, "test.csv")
        train_csv = _load_csv(train_csv_fpath)
        test_csv = _load_csv(test_csv_fpath)

        WIDTH = 32
        HEIGHT = 32
        x_train = np.zeros((len(train_csv), 1, WIDTH, HEIGHT), dtype=np.uint8)
        x_test = np.zeros((len(test_csv), 1, WIDTH, HEIGHT), dtype=np.uint8)
        y_train, s_train = [], []
        y_test, s_test = [], []

        # Load training data
        for i, data_item in enumerate(train_csv):
            fname = os.path.join(untar_fpath, data_item['path'])
            s_train.append(fname)
            x_train[i, 0, :, :] = scipy.ndimage.imread(fname,
                                                       flatten=False,
                                                       mode='L')
            label = symbol_id2index[data_item['symbol_id']]
            y_train.append(label)
        y_train = np.array(y_train, dtype=np.int64)

        # Load test data
        for i, data_item in enumerate(test_csv):
            fname = os.path.join(untar_fpath, data_item['path'])
            s_test.append(fname)
            x_train[i, 0, :, :] = scipy.ndimage.imread(fname,
                                                       flatten=False,
                                                       mode='L')
            label = symbol_id2index[data_item['symbol_id']]
            y_test.append(label)
        y_test = np.array(y_test, dtype=np.int64)

        data = {'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test,
                'labels': labels
                }

        # Store data as pickle to speed up later calls
        with open(pickle_fpath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_fpath, 'rb') as f:
            data = pickle.load(f)
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
        globals()["labels"] = data['labels']

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_dim_ordering() == 'tf':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)
