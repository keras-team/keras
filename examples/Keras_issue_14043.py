# After tensorflow 2, keras is being used in Backend
# comment shows an alternative way to run the command
#Documentation:- https://keras.io/api/datasets/cifar10/

# Version for your reference, Downgrade/ Ugrade/ Reinstall Accordingly
import keras
print(keras.__version__) #2.2.4 in my case

from keras.datasets import cifar10

(x_train,y_train), (x_test, y_test) = cifar10.load_data()
print(len(x_train))
print(x_train[0])

# Screenshot of Output, For Your Reference :- 

# Using Tf2 :- Recommended
#pip install tensorflow

# from tensorflow.keras.datasets import cifar10

# (x_train,y_train), (x_test, y_test) = cifar10.load_data()

# print(len(x_train))
# print(x_train[0])
