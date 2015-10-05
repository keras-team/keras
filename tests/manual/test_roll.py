"""
This is a simple example I'm including to help debug new layers.
The idea is to debug directly from keras root folder without the need to run
python setup.py install

Modify this script as needed to test whatever layer I'm creating
"""
from nose.tools import assert_raises
from numpy.testing import assert_array_equal
from keras.models import Graph
from keras.layers.core import Roll
import numpy as np

def roll(n, axis, ndim):
    '''
    if axis is None, uses roll1D, else roll2D
    '''
    graph = Graph()

    graph.add_input(name='input', ndim=ndim)

    graph.add_node(Roll(n, axis),
            name='rolled', input='input')
    
    graph.add_output(name='output', input='rolled')

    graph.compile('rmsprop', {'output':'mse'})

    return graph

def test_roll():
    T3 = np.array(range(64)).reshape(4,4,4)
    T4 = np.array(range(256)).reshape(4,4,4,4)

    # test axis 0 can't be rolled
    #assert_raises(Exception, roll, 1, 0, 3)

    # test forward rollion along axis 1 in 3D tensor
    graph = roll(1, 1, 3)
    out = graph.predict({'input':T3})['output']
    assert_array_equal(out, np.roll(T3,1,1))

    # test forward rollion along axis 2 in 3D tensor
    graph = roll(-1, 2, 3)
    out = graph.predict({'input':T3})['output']
    assert_array_equal(out, np.roll(T3,-1,2))

    # test backwards rollion along axis 2 in 4D tensor
    graph = roll(-2, 2, 4)
    out = graph.predict({'input':T4})['output']
    assert_array_equal(out, np.roll(T4,-2, 2))

    # test forward rollion along axis 3 in 4D tensor
    graph = roll(-3, 3, 4)
    out = graph.predict({'input':T4})['output']
    assert_array_equal(out, np.roll(T4,-3, 3))

    # test forward rollion along axis 3 in 4D tensor
    graph = roll(-3, 3, 4)
    out = graph.predict({'input':T4})['output']
    assert_array_equal(out, np.roll(T4,-3, 3))
