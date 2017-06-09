
# coding: utf-8

# In[ ]:


import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
import multiprocessing
import time


# In[ ]:


# Data Class
class ImageBatchImporter:

    """
        Arguments:
            train_path - Path to the folder with training data in it
            img_dim - Array with the target image dimensions. Must be the same for all images
            classes - Array with the list of class names. Must match the labels of the dataset

        Data Structure:
            All data images should be put in a folder named in the format class_index.ext
                eg.
                    plane_1.png
    """

    def buffer(self, batch_size):
        name = multiprocessing.current_process().name

        # Load data into buffer
        buffer_x, buffer_y = self.load_data(batch_size)
        return

    # Initiate variables on creation
    def __init__(self, train_path, img_dim, classes):
        # Instantiate Variables
        self.img_dim = img_dim
        self.train_path = train_path
        self.classes = classes
        # Set File List
        self.file_list = os.listdir(self.train_path)
        # Training Set Size
        self.size_training_set = len(self.file_list)
        # Buffer Check
        self.buffer_set = False

    def load_data(self, batch_size):
        # Setup Variables Locally
        size_training_set = self.size_training_set
        train_path = self.train_path
        file_list = self.file_list
        classes = self.classes
        img_dim = self.img_dim

        # Output X
        img_dim.insert(0, batch_size)
        x_batch = np.zeros((img_dim), dtype='float32')
        # Output Y
        image_classes = list()
        for i in range(batch_size):
            # Determine index of next file
            next_file_index = random.randint(0, size_training_set - 1)
            # Get file from list
            next_file = file_list[next_file_index]
            # Set the batch array
            x_batch[i] = mpimg.imread(train_path + "/" + next_file)
            # Set the Y
            num_classes = len(classes)
            class_one_hot = np.zeros(num_classes)
            class_name = file_list[next_file_index].split('_')[0]
            class_set = False
            # Make Y a one-hot array
            for j in range(num_classes):
                if class_name == classes[j]:
                    class_one_hot[j] = 1
                    class_set = True
            # Throw error if no class was identified
            if not class_set:
                raise ValueError('Class not found: {}'.format(class_name))
        return x_batch, class_one_hot

    # Get the next training batch
    def next_training_batch(self, batch_size):
        if self.buffer_set:
            # Set the return data
            return_x = self.buffer_x
            return_y = self.buffer_y
            # Start process to get data
            self.buffer(batch_size)
            # Return the buffer data
            return return_x, return_y

        return_x, return_y = self.load_data(batch_size)

        # Setup the buffer files
        b = multiprocessing.Process(target=self.buffer, args=(batch_size,))
        b.start()

        # If buffer isn't set
        return return_x, return_y
