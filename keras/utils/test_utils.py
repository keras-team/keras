import numpy as np


def get_test_data(nb_train=1000, nb_test=500, input_shape=(10,), output_shape=(2,),
                  classification=True, nb_class=2):
    '''
        classification=True overrides output_shape
        (i.e. output_shape is set to (1,)) and the output
        consists in integers in [0, nb_class-1].

        Otherwise: float output with shape output_shape.
    '''
    nb_sample = nb_train + nb_test
    if classification:
        y = np.random.randint(0, nb_class, size=(nb_sample, 1))
        X = np.zeros((nb_sample,) + input_shape)
        for i in range(nb_sample):
            X[i] = np.random.normal(loc=y[i], scale=1.0, size=input_shape)
    else:
        y_loc = np.random.random((nb_sample,))
        X = np.zeros((nb_sample,) + input_shape)
        y = np.zeros((nb_sample,) + output_shape)
        for i in range(nb_sample):
            X[i] = np.random.normal(loc=y_loc[i], scale=1.0, size=input_shape)
            y[i] = np.random.normal(loc=y_loc[i], scale=1.0, size=output_shape)

    return (X[:nb_train], y[:nb_train]), (X[nb_train:], y[nb_train:])
