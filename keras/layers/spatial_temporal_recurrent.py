from .. import backend as K
from .. import activations
from .. import initializers
from ..engine import Layer


class SpatioTemporalLSTMCell(Layer):
    def __init__(self, units,
                 num_hidden_in,
                 kernel_size,
                 sequence_shape,
                 use_bias=True,
                 forget_bias=1.0,
                 activation='tanh',
                 kernel_initializer='random_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(SpatioTemporalLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.num_hidden_in = num_hidden_in
        self.kernel_size = kernel_size
        self.batch_size = sequence_shape[0]
        self.height = sequence_shape[2]
        self.width = sequence_shape[3]
        self.use_bias = use_bias
        self.forget_bias = forget_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.activation = activations.get(activation)
    
    def build(self, input_shape=None):
        kernel_shape_hcc = self.kernel_size + (self.num_hidden_in, self.units * 4)
        self.kernel_hcc = self.add_weight(shape=kernel_shape_hcc,
                                          initializer=self.kernel_initializer,
                                          name='kernel_hcc')
        if self.use_bias:
            self.bias_hcc = self.add_weight(shape=(self.units * 4,),
                                            initializer=self.bias_initializer,
                                            name='bias_hcc')
        else:
            self.bias_hcc = None
        
        kernel_shape_mcc = self.kernel_size + (self.num_hidden_in, self.units * 3)
        self.kernel_mcc = self.add_weight(shape=kernel_shape_mcc,
                                          initializer=self.kernel_initializer,
                                          name='kernel_mcc')
        if self.use_bias:
            self.bias_mcc = self.add_weight(shape=(self.units * 3,),
                                            initializer=self.bias_initializer,
                                            name='bias_mcc')
        else:
            self.bias_hcc = None
        
        if input_shape:
            kernel_shape_xcc = self.kernel_size + (input_shape[-1], self.units * 7)
            self.kernel_xcc = self.add_weight(shape=kernel_shape_xcc,
                                              initializer=self.kernel_initializer,
                                              name='kernel_xcc')
            if self.use_bias:
                self.bias_xcc = self.add_weight(shape=(self.units * 7,),
                                                initializer=self.bias_initializer,
                                                name='bias_xcc')
            else:
                self.bias_xcc = None
        
        kernel_shape_ccc = self.kernel_size + (self.units, self.units * 4)
        self.kernel_ccc = self.add_weight(shape=kernel_shape_ccc,
                                          initializer=self.kernel_initializer,
                                          name='kernel_mcc')
        if self.use_bias:
            self.bias_ccc = self.add_weight(shape=(self.units * 4,),
                                            initializer=self.bias_initializer,
                                            name='bias_ccc')
        else:
            self.bias_ccc = None
        
        kernel_shape_om = self.kernel_size + (self.units, self.units)
        self.kernel_om = self.add_weight(shape=kernel_shape_om,
                                         initializer=self.kernel_initializer,
                                         name='kernel_om')
        if self.use_bias:
            self.bias_om = self.add_weight(shape=(self.units,),
                                           initializer=self.bias_initializer,
                                           name='bias_om')
        else:
            self.bias_om = None
        
        kernel_shape_cell = (1, 1) + (self.units, self.units)
        self.kernel_cell = self.add_weight(shape=kernel_shape_cell,
                                           initializer=self.kernel_initializer,
                                           name='kernel_cell')
        if self.use_bias:
            self.bias_cell = self.add_weight(shape=(self.units,),
                                             initializer=self.bias_initializer,
                                             name='bias_cell')
        else:
            self.bias_cell = None
    
    def _kernel_conv(self, x, kernel, bias=None):
        tmp = K.conv2d(x, kernel, 1, padding='same')
        if bias is not None:
            tmp = K.bias_add(x, bias)
        return tmp
    
    def call(self, inputs, states, **kwargs):
        h, c, m = states
        
        if h is None:
            h = K.zeros([self.batch_size, self.height, self.width, self.num_hidden_in], dtype='float32')
        if c is None:
            c = K.zeros([self.batch_size, self.height, self.width, self.units], dtype='float32')
        if m is None:
            m = K.zeros([self.batch_size, self.height, self.width, self.num_hidden_in], dtype='float32')
        
        h_cc = self._kernel_conv(h, self.kernel_hcc, self.bias_hcc)
        
        i_h = h_cc[:, :, :, :self.units]
        g_h = h_cc[:, :, :, self.units:self.units * 2]
        f_h = h_cc[:, :, :, self.units * 2:self.units * 3]
        o_h = h_cc[:, :, :, self.units * 2:self.units * 3]
        
        if inputs is None:
            i = K.tanh(i_h)
            f = K.tanh(f_h + self.forget_bias)
            g = K.tanh(g_h)
        else:
            x_cc = self._kernel_conv(inputs, self.kernel_xcc, self.bias_xcc)
            
            i_x = x_cc[:, :, :, :self.units]
            g_x = x_cc[:, :, :, self.units:self.units * 2]
            f_x = x_cc[:, :, :, self.units * 2:self.units * 3]
            o_x = x_cc[:, :, :, self.units * 3:self.units * 4]
            i_x_ = x_cc[:, :, :, self.units * 4:self.units * 5]
            g_x_ = x_cc[:, :, :, self.units * 5:self.units * 6]
            f_x_ = x_cc[:, :, :, self.units * 6:self.units * 7]
            
            i = K.tanh(i_x + i_h)
            f = K.tanh(f_x + f_h + self.forget_bias)
            g = K.tanh(g_x + g_h)
        
        c_new = f * c + i * g
        
        c_cc = self._kernel_conv(c_new, self.kernel_ccc, self.bias_xcc)
        
        i_c = c_cc[:, :, :, :self.units]
        g_c = c_cc[:, :, :, self.units:self.units * 2]
        f_c = c_cc[:, :, :, self.units * 2:self.units * 3]
        o_c = c_cc[:, :, :, self.units * 3:self.units * 4]
        
        m_cc = self._kernel_conv(m, self.kernel_mcc, self.bias_mcc)
        
        i_m = m_cc[:, :, :, :self.units]
        f_m = m_cc[:, :, :, self.units:self.units * 2]
        m_m = m_cc[:, :, :, self.units * 2:self.units * 3]
        
        if inputs is None:
            ii = K.tanh(i_c + i_m)
            ff = K.tanh(f_c + f_m + self.forget_bias)
            gg = K.tanh(g_c)
        
        else:
            ii = K.tanh(i_c + i_x_ + i_m)
            ff = K.tanh(f_c + f_x_ + f_m + self.forget_bias)
            gg = K.tanh(g_c + g_x_)
        m_new = ff * f * K.tanh(m_m) + i * ii * gg
        
        o_m = self._kernel_conv(m_new, self.kernel_om, self.bias_om)
        if inputs is None:
            o = K.tanh(o_h + o_c + o_m)
        else:
            o = K.tanh(o_x + o_h + o_c + o_m)
        
        cell = K.concatenate([c_new, m_new])
        cell = self._kernel_conv(cell, self.kernel_cell, self.bias_cell)
        h_new = o * K.tanh(cell)
        
        return h_new, [h_new, c_new, m_new]
    
    def get_config(self):
        config = {'units': self.units,
                  'num_hidden_in': self.num_hidden_in,
                  'kernel_size': self.kernel_size,
                  'use_bias': self.use_bias,
                  'forget_bias': self.forget_bias,
                  'activation': self.activation,
                  'kernel_initializer':self.kernel_initializer,
                  'bias_initializer':self.bias_initializer,
        }
        base_config = super(SpatioTemporalLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
