
class Siamese(Layer):

    '''Shared layer with multiple inputs

    Output shape
    ------------
    Depends on merge_mode argument

    Arguments
    ---------
    layer - The layer to be shared across multiple inputs
    inputs - inputs to the shared layer
    merge_mode - Similar to mode argument of Merge layer
    concat_axis - Similar to concat_axis argument of Merge layer
    dot_axes - Similar to dot_axes argument of Merge layer
    '''

    def __init__(self, layer, inputs, merge_mode='concat', concat_axis=1, dot_axes=-1):

        if merge_mode not in ['sum', 'mul', 'concat', 'ave', 'join', 'cos', 'dot', None]:
            raise Exception("Invalid merge mode: " + str(mode))

        if merge_mode in {'cos', 'dot'}:
            if len(inputs) > 2:
                raise Exception(mode + " merge takes exactly 2 layers")

        self.layer = layer
        self.inputs = inputs
        self.params = []
        self.merge_mode = merge_mode
        self.concat_axis = concat_axis
        self.dot_axes = dot_axes
        layer.set_previous(inputs[0])
        self.regularizers = []
        self.constraints = []
        self.updates = []
        layers = [layer]
        if merge_mode:
            layers += inputs
        for l in layers:
            params, regs, consts, updates = l.get_params()
            self.regularizers += regs
            self.updates += updates
            # params and constraints have the same size
            for p, c in zip(params, consts):
                if p not in self.params:
                    self.params.append(p)
                    self.constraints.append(c)

    @property
    def output_shape(self):
        if merge_mode is None:
            return self.layer.output_shape
        input_shapes = [self.get_output_shape(i) for i in range(len(inputs))]
        if self.merge_mode in ['sum', 'mul', 'ave']:
            return input_shapes[0]
        elif self.merge_mode == 'concat':
            output_shape = list(input_shapes[0])
            for shape in input_shapes[1:]:
                output_shape[self.concat_axis] += shape[self.concat_axis]
            return tuple(output_shape)
        elif self.merge_mode == 'join':
            return None
        elif self.merge_mode == 'dot':
            shape1 = list(input_shapes[0])
            shape2 = list(input_shapes[1])
            for i in self.dot_axes[0]:
                shape1.pop(i)
            for i in self.dot_axes[1]:
                shape2.pop(i)
            shape = shape1 + shape2[1:]
            if len(shape) == 1:
                shape.append(1)
            return tuple(shape)
        elif self.merge_mode == 'cos':
            return tuple(input_shapes[0][0], 1)

    def get_params(self):
        return self.params, self.regularizers, self.constraints, self.updates

    def get_output_at(self, head, train=False):
        if hasattr(self.layer, 'previous'):
            self.layer.previous = self.inputs[head]
        else:
            self.layer.layers[0].previous = self.inputs[head]
        return self.layer.get_output(train)

    def get_output_shape(self, head, train=False):
        self.layer.set_previous(self.inputs[head])
        return self.layer.output_shape

    def get_output_join(self, train=False):
        o = OrderedDict()
        for i in range(len(inputs)):
            X = self.get_output_at(i, train)
            if X.name is None:
                raise ValueError("merge_mode='join' only works with named inputs")
            o[X.name] = X
        return o

    def get_output_sum(self, train=False):
        s = self.get_output_at(0, train)
        for i in range(1, len(self.inputs)):
            s += self.get_output_at(i, train)
        return s

    def get_output_ave(self, train=False):
        n = len(self.inputs)
        s = self.get_output_at(0, train)
        for i in range(1, n):
            s += self.get_output_at(i, train)
        s /= n
        return s

    def get_output_concat(self, train=False):
        inputs = [self.get_output_at(i, train) for i in range(len(self.inputs))]
        return T.concatenate(inputs, axis=self.concat_axis)

    def get_output_mul(self, train=False):
        s = self.get_output_at(0, train)
        for i in range(1, len(self.inputs)):
            s *= self.get_output_at(i, train)
        return s

    def get_output_dot(self, train=False):
        l1 = self.get_output_at(0, train)
        l2 = self.get_output_at(1, train)
        output = T.batched_tensordot(l1, l2, self.dot_axes)
        output = output.dimshuffle((0, 'x'))
        return output

    def get_output_cos(self, train=False):
        l1 = self.get_output_at(0, train)
        l2 = self.get_output_at(1, train)
        output, _ = theano.scan(lambda v1, v2: T.dot(v1, v2)/T.sqrt(T.dot(v1, v1) * T.dot(v2, v2)), sequences=[l1, l2], outputs_info=None)
        output = output.dimshuffle((0, 'x'))
        return output

    def get_output(self, train=False):
        mode = self.merge_mode
        if mode == 'join':
            return self.get_output_join(train)
        elif mode == 'concat':
            return self.get_output_concat(train)
        elif mode == 'sum':
            return self.get_output_sum(train)
        elif mode == 'ave':
            return self.get_output_ave(train)
        elif mode == 'mul':
            return self.get_output_mul(train)
        elif mode == 'dot':
            return self.get_output_dot(train)
        elif mode == 'cos':
            return self.get_output_dot(train)

    def get_input(self, train=False):
        res = []
        for i in range(len(self.inputs)):
            o = self.inputs[i].get_input(train)
            if type(o) != list:
                o = [o]
            for output in o:
                if output not in res:
                    res.append(output)
        return res

    @property
    def input(self):
        return self.get_input()

    def supports_masked_input(self):
        return False

    def get_output_mask(self, train=None):
        return None

    def get_weights(self):
        weights = layer.get_weights()
        if merge_mode:
            for m in self.inputs:
                weights += m.get_weights()
        return weights

    def set_weights(self, weights):
        nb_param = len(self.layer.params)
        self.layer.set_weights(weights[:nb_param])
        weights = weights[nb_param:]
        if merge_mode:
            for i in range(len(self.inputs)):
                nb_param = len(self.inputs[i].params)
                self.inputs[i].set_weights(weights[:nb_param])
                weights = weights[nb_param:]

    def get_config(self):

        config = {"name": self.__class__.__name__,
                  "layer": self.layer.get_config,
                  "inputs": [m.get_config() for m in self.inputs],
                  "merge_mode": self.merge_mode,
                  "concat_axis": self.concat_axis,
                  "dot_axes": self.dot_axes
                  }
        base_config = super(Siamese, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SiameseHead(Layer):

    '''This layer should be added only on top of a Siamese layer with merge_mode = None

    Outputs the output of the Siamese layer at a given index, specified by the head argument

    Arguments
    ---------
    head - The index at which the output of the Siamese layer should be obtained
    '''
    def __init__(self, head):
        self.head = head
        self.params = []

    def get_output(self, train=False):
        return self.get_input(train)

    @property
    def input_shape(self):
        return self.previous.get_output_shape(self.head)

    def get_input(self, train=False):
        return self.previous.get_output_at(self.head, train)

    def get_config(self):

        config = {"name": self.__class__.__name__,
                  "head": self.head
                  }

        base_config = super(SiameseHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def set_previous(self, layer):
        self.previous = layer


def add_shared_layer(layer, inputs):
    '''
    Use this function to add a shared layer across multiple Sequential models without merging the outputs
    '''
    input_layers = [l.layers[-1] for l in inputs]
    s = Siamese(layer, input_layers, merge_mode=None)
    for i in range(len(inputs)):
        sh = SiameseHead(i)
        inputs[i].add(s)
        inputs[i].add(sh)
