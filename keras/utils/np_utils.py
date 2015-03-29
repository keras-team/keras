import numpy as np
import scipy as sp

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def binary_logloss(p, y):
    epsilon = 1e-15
    p = sp.maximum(epsilon, p)
    p = sp.minimum(1-epsilon, p)
    res = sum(y*sp.log(p) + sp.subtract(1,y)*sp.log(sp.subtract(1,p)))
    res *= -1.0/len(y)
    return res

def multiclass_logloss(P, Y):
    score = 0.
    npreds = [P[i][Y[i]-1] for i in range(len(Y))]
    score = -(1./len(Y)) * np.sum(np.log(npreds))
    return score

def accuracy(p, y):
    return np.mean([a==b for a, b in zip(p, y)])

def probas_to_classes(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return categorical_probas_to_classes(y_pred)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])

def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)


def save_array(array, name):
    import tables
    f = tables.open_file(name, 'w')
    atom = tables.Atom.from_dtype(array.dtype)
    ds = f.createCArray(f.root, 'data', atom, array.shape)
    ds[:] = array
    f.close()

def load_array(name):
    import tables
    f = tables.open_file(name)
    array = f.root.data
    a=np.empty(shape=array.shape, dtype=array.dtype)
    a[:]=array[:]
    f.close()
    return a