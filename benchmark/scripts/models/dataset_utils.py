"""Dataset Utilities used in benchmark script
"""

import re
import io
import os

from keras.utils.data_utils import get_file


def get_dataset(dataset_name):
    """
    Download and read the dataset into a string

    # Arguments
        dataset_name: dataset type

    # Returns
        A text string
    """
    if dataset_name == 'nietzsche':
        path = get_file(
            'nietzsche.txt',
            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with io.open(path, encoding='utf-8') as f:
            text = f.read().lower()
        return text
    elif dataset_name == 'wikitext2':
        path = get_file(fname='wikitext-2-raw.zip', extract=True,
                        archive_format='zip',
                        origin='https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip')
        path = path.split('.zip')[0]
        path = os.path.join(path, 'wiki.train.raw')
        text = ''
        with io.open(path, encoding='utf-8') as f:
            for line in f:
                line = " ".join(re.findall("[a-zA-Z0-9]+", line))
                text += line.lower()
        return text
    else:
        raise ValueError('Only support `nietzsche` and `wikitext2` dataset. '
                         'Please add the functionality here for your own '
                         'dataset')
