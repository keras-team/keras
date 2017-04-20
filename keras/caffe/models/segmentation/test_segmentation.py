'''
author: Akshay Chawla (https://github.com/akshaychawla)
TEST:
Test convert.py's ability to handle Deconvolution and Crop layers 
by converting voc-fcn8s .prototxt and .caffemodel present in the caffe/models/segmentation folder
'''  

import numpy as np
import keras.caffe.convert as convert
import cv2
import matplotlib.pyplot as plt 
import os

# check whether files are present in folder 
assert os.path.exists("./deploy.prototxt"), "Err. Couldn't find the debug.prototxt file"
assert os.path.exists("./fcn8s-heavy-pascal.caffemodel"), "Err. Cannot find .caffemodel file.	\
please download file using command : wget http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel "

model = convert.caffe_to_keras("./deploy.prototxt", "./fcn8s-heavy-pascal.caffemodel", debug=1)

print "Yay!"

# 1. load image
img = cv2.imread("./horse.png")

# modify it
img = np.rollaxis(img,2)
img = np.expand_dims(img, 0)

# 2. run forward pass
op = model.predict(img)

# 3. reshape output 
op = op[0] 
op = op.reshape((500,500,21))
op_arg = np.argmax(op, axis=2)

# 4. plot output 
plt.imshow(op_arg)
plt.show()

print "..done"