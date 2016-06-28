from keras.layers.core import Layer

import theano.tensor as T


class LRN2D(Layer):
    """
    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n")
      
    def call(self, x, mask=None):
        X = x
        input_dim = X.shape
        half_n = self.n // 2
        input_sqr = T.sqr(X)
        b, ch, r, c = input_dim
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],input_sqr)
        scale = self.k
        norm_alpha = self.alpha / self.n
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return X / scale
        
    def get_output(self, train):
        X = self.get_input(train)
        input_dim = X.shape
        half_n = self.n // 2
        input_sqr = T.sqr(X)
        b, ch, r, c = input_dim
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],input_sqr)
        scale = self.k
        norm_alpha = self.alpha / self.n
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        return {
            "alpha":self.alpha,
            "k":self.k,
            "beta":self.beta,
            "n": self.n}
