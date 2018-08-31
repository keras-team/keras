import sys

if sys.version_info[0] == 3:
    import pickle
else:
    import cPickle as pickle


from keras.layers import *
from keras.models import *

a = Input((3, 2, 5))
b = Dense(10)(a)
c = Dense(5)(b)

model = Model(a, c)

state = pickle.dumps(model)

model2 = pickle.loads(state)

