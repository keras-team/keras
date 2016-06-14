from collections import OrderedDict
import warnings
import copy

from .. import backend as K
from ..layers import InputLayer, Layer, Merge
from ..engine.training import Model


class Graph(Model):
    '''Arbitrary connection graph.

    THIS IS A LEGACY MODEL AND SHOULD NOT BE USED
    except for backwards compatibility support.

    For multi-inputs/multi-outputs models, or
    models using shared layers, use the functional API instead.
    '''

    def __init__(self, name=None):
        # model attributes
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.built = False
        self.supports_masking = False

        # legacy attributes (we prefix them with _graph_)
        self._graph_namespace = set()  # strings
        self._graph_nodes = OrderedDict()  # layer-like
        self._graph_inputs = OrderedDict()  # layer-like
        self._graph_outputs = OrderedDict()  # layer-like
        self._graph_input_config = []  # dicts
        self._graph_output_config = []  # dicts
        self._graph_node_config = []  # dicts
        self._graph_shared_nodes_names = []

        if not name:
            prefix = 'graph_'
            name = prefix + str(K.get_uid(prefix))
        self.name = name

    def __call__(self, x, mask=None):
        self.build()
        return super(Graph, self).__call__(x, mask)

    def build(self, input_shape=None):
        # this will crash if the input/output layers have multiple nodes
        # no plans to support that case since Graph is deprecated
        input_tensors = [layer.output for layer in self._graph_inputs.values()]
        output_tensors = [layer.output for layer in self._graph_outputs.values()]
        # actually create the model
        super(Graph, self).__init__(input_tensors,
                                    output_tensors,
                                    name=self.name)
        self.built = True

    def compile(self, optimizer, loss,
                metrics=[],
                sample_weight_modes=None,
                loss_weights=None,
                **kwargs):
        '''Configures the learning process.

        # Arguments
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](optimizers.md).
            loss: dictionary mapping the name(s) of the output(s) to
                a loss function (string name of objective function or
                objective function. See [objectives](objectives.md)).
            metrics: list of str (name of metrics) or
                list of metrics functions. See [metrics](metrics.md).
            sample_weight_modes: optional dictionary mapping certain
                output names to a sample weight mode ("temporal" and None
                are the only supported modes). If you need to do
                timestep-wise loss weighting on one of your graph outputs,
                you will need to set the sample weight mode for this output
                to "temporal".
            loss_weights: dictionary you can pass to specify a weight
                coefficient for each loss function (in a multi-output model).
                If no loss weight is specified for an output,
                the weight for this output's loss will be considered to be 1.
            kwargs: for Theano backend, these are passed into K.function.
                Ignored for Tensorflow backend.
        '''
        # create the underlying Model
        if not self.built:
            self.build()
        super(Graph, self).compile(optimizer, loss,
                                   metrics=metrics,
                                   sample_weight_mode=sample_weight_modes,
                                   loss_weights=loss_weights,
                                   **kwargs)

    def add_input(self, name, input_shape=None,
                  batch_input_shape=None, dtype='float'):
        '''Adds an input to the graph.

        # Arguments:
            name: string. The name of the new input.
                Must be unique in the graph.
            input_shape: a tuple of integers,
                the expected shape of the input samples.
                Does not include the batch size.
            batch_input_shape: a tuple of integers,
                the expected shape of the whole input batch,
                including the batch size.
            dtype: 'float', or 'int'.
        '''
        if name in self._graph_namespace:
            raise Exception('Duplicate node identifier: ' + name)
        self._graph_namespace.add(name)
        self.built = False

        if dtype[:3] == 'int':
            dtype = 'int32'
        elif dtype[:5] == 'float':
            dtype = K.floatx()
        else:
            raise Exception('Uknown dtype (should be "int" or "float"): ' +
                            str(dtype))

        # create input layer
        input_layer = InputLayer(input_shape=input_shape,
                                 batch_input_shape=batch_input_shape,
                                 name=name, input_dtype=dtype)
        self._graph_inputs[name] = input_layer

        # append input config to self._graph_input_config
        config = {'name': name, 'dtype': dtype}
        if batch_input_shape:
            config['batch_input_shape'] = batch_input_shape
        else:
            config['input_shape'] = input_shape
        self._graph_input_config.append(config)

    def add_node(self, layer, name, input=None, inputs=[],
                 merge_mode='concat', concat_axis=-1, dot_axes=-1,
                 create_output=False):
        '''Adds a node in the graph. It can be connected to multiple
        inputs, which will first be merged into one tensor
        according to the mode specified.

        # Arguments
            layer: the layer at the node.
            name: name for the node.
            input: when connecting the layer to a single input,
                this is the name of the incoming node.
            inputs: when connecting the layer to multiple inputs,
                this is a list of names of incoming nodes.
            merge_mode: one of {concat, sum, dot, ave, mul}
            concat_axis: when `merge_mode=='concat'`, this is the
                input concatenation axis.
            dot_axes: when `merge_mode='dot'`,
                this is the contraction axes specification;
                see the `Merge` layer for details.
            create_output: boolean. Set this to `True` if you want the output
                of your node to be an output of the graph.
        '''
        if name in self._graph_namespace:
            raise Exception('Duplicate node identifier: ' + name)
        self._graph_namespace.add(name)
        layer.name = name
        self.built = False

        if input:
            if input not in self._graph_namespace:
                raise Exception('Unknown node/input identifier: ' + input)
            if input in self._graph_nodes:
                layer.add_inbound_node(self._graph_nodes[input])
            elif input in self._graph_inputs:
                layer.add_inbound_node(self._graph_inputs[input])
        if inputs:
            to_merge = []
            for n in inputs:
                if n in self._graph_nodes:
                    to_merge.append(self._graph_nodes[n])
                elif n in self._graph_inputs:
                    to_merge.append(self._graph_inputs[n])
                else:
                    raise Exception('Unknown identifier: ' + n)
            merge = Merge(to_merge, mode=merge_mode,
                          concat_axis=concat_axis, dot_axes=dot_axes,
                          name='merge_inputs_for_' + name)
            layer.add_inbound_node(merge)
        self._graph_nodes[name] = layer
        self._graph_node_config.append({'name': name,
                                        'input': input,
                                        'inputs': inputs,
                                        'merge_mode': merge_mode,
                                        'concat_axis': concat_axis,
                                        'dot_axes': dot_axes,
                                        'create_output': create_output})
        if create_output:
            self.add_output(name, input=name)

    def add_shared_node(self, layer, name, inputs=[], merge_mode=None,
                        concat_axis=-1, dot_axes=-1, outputs=[],
                        create_output=False):
        '''Used to share a same layer across multiple nodes.

        Supposed, for instance, that you want to apply one same `Dense` layer
        after two different nodes ('node_a' and 'node_b').
        You can then add the dense layer as a shared node by calling:

        ```python
        model.add_shared_node(my_dense, name='shared_dense', inputs=['node_a', 'node_b'], ...)
        ```

        If you want access to the output of dense(node_a) and dense(node_b) separately,
        you can add these outputs to the Graph by passing an `outputs` argument:

        ```python
        model.add_shared_node(my_dense, name='shared_dense', inputs=['node_a', 'node_b'],
                              outputs=['dense_output_a', 'dense_outputs_b'])
        ```

        Otherwise you can merge these different outputs via `merge_mode`.
        In that case you can access the merged output
        under the identifier `name`.

        # Arguments
            layer: The layer to be shared across multiple inputs
            name: Name of the shared node
            inputs: List of names of input nodes
            merge_mode: Same meaning as `merge_mode` argument of `add_node()`
            concat_axis: Same meaning as `concat_axis` argument of `add_node()`
            dot_axes: Same meaning as `dot_axes` argument of `add_node()`
            outputs: Used when `merge_mode=None`. Names for the output nodes.
            create_output: Same meaning as `create_output` argument of `add_node()`.
        '''
        if name in self._graph_namespace:
            raise Exception('Duplicate node identifier: ' + name)
        self._graph_namespace.add(name)
        self.built = False

        for o in outputs:
            if o in self._graph_namespace:
                raise Exception('Duplicate node identifier: ' + o)
        if merge_mode:
            if merge_mode not in {'sum', 'ave', 'mul', 'dot', 'cos', 'concat'}:
                raise Exception('Invalid merge mode:', merge_mode)
        input_layers = []
        for i in range(len(inputs)):
            input = inputs[i]
            if input in self._graph_nodes:
                n = self._graph_nodes[input]
                input_layers.append(n)
            elif input in self._graph_inputs:
                n = self._graph_inputs[input]
                input_layers.append(n)
            else:
                raise Exception('Unknown identifier: ' + input)

        created_node_indices = []
        for input_layer in input_layers:
            created_node_indices.append(len(layer.inbound_nodes))
            layer.add_inbound_node(input_layer)

        if merge_mode:
            layer.name = 'input_for_' + name
            # collect all output nodes of layer and merge them into a single output
            merge = Merge([layer for _ in range(len(inputs))],
                          mode=merge_mode,
                          concat_axis=concat_axis, dot_axes=dot_axes,
                          node_indices=created_node_indices,
                          name=name)
            self._graph_nodes[name] = merge
            if create_output:
                self.add_output(name, input=name)
        else:
            layer.name = name
            # create one new layer per output node of layer,
            # and add them to the Graph with their own identifiers
            if len(outputs) != len(inputs):
                raise Exception('When using merge_mode=None, '
                                'you should provide a list of '
                                'output names (`output` argument) '
                                'the same size as `input`.')
            for i in range(len(outputs)):
                output_layer_name = outputs[i]
                output_layer = Layer(name=output_layer_name)
                output_layer.add_inbound_node(layer, created_node_indices[i])
                self._graph_namespace.add(output_layer_name)
                self._graph_nodes[output_layer_name] = output_layer
                if create_output:
                    self.add_output(output_layer_name, input=output_layer_name)

        self._graph_node_config.append({'name': name,
                                        'layer': {
                                            'config': layer.get_config(),
                                            'class_name': layer.__class__.__name__,
                                        },
                                        'inputs': inputs,
                                        'merge_mode': merge_mode,
                                        'concat_axis': concat_axis,
                                        'dot_axes': dot_axes,
                                        'outputs': outputs,
                                        'create_output': create_output if merge_mode else False})
        self._graph_shared_nodes_names.append(name)

    def add_output(self, name, input=None, inputs=[],
                   merge_mode='concat', concat_axis=-1, dot_axes=-1):
        '''Adds an output to the graph.

        This output can merge several node outputs into a single output.

        # Arguments
            name: name of the output.
            input: when connecting the layer to a single input,
                this is the name of the incoming node.
            inputs: when connecting the layer to multiple inputs,
                this is a list of names of incoming nodes.
            merge_mode: one of {concat, sum, dot, ave, mul}
            concat_axis: when `merge_mode=='concat'`, this is the
                input concatenation axis.
            dot_axes: when `merge_mode='dot'`,
                this is the contraction axes specification;
                see the `Merge layer for details.
        '''
        if name not in self._graph_namespace:
            self._graph_namespace.add(name)
        if name in self._graph_outputs:
            raise Exception('Duplicate output identifier:', name)
        self.built = False

        if input:
            if input in self._graph_nodes:
                layer = self._graph_nodes[input]
            elif input in self._graph_inputs:
                layer = self._graph_inputs[input]
            else:
                raise Exception('Unknown node/input identifier: ' + input)
            if layer.name == name:
                self._graph_outputs[name] = layer
            else:
                layer.name = name
                self._graph_outputs[name] = layer
        if inputs:
            to_merge = []
            for n in inputs:
                if n not in self._graph_nodes:
                    raise Exception('Unknown identifier: ' + n)
                to_merge.append(self._graph_nodes[n])
            merge = Merge(to_merge, mode=merge_mode,
                          concat_axis=concat_axis, dot_axes=dot_axes,
                          name=name)
            self._graph_outputs[name] = merge

        self._graph_output_config.append({'name': name,
                                          'input': input,
                                          'inputs': inputs,
                                          'merge_mode': merge_mode,
                                          'concat_axis': concat_axis,
                                          'dot_axes': dot_axes})

    def _get_x(self, data):
        x = []
        for key in self._graph_inputs.keys():
            if key not in data:
                raise Exception('Expected to be provided an array '
                                '(in dict argument `data`) for input "' +
                                key + '".')
            x.append(data[key])
        return x

    def _get_y(self, data):
        y = []
        for key in self._graph_outputs.keys():
            if key not in data:
                raise Exception('Expected to be provided an array '
                                '(in dict argument `data`) for output "' +
                                key + '".')
            y.append(data[key])
        return y

    def fit(self, data, batch_size=32, nb_epoch=10, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, **kwargs):
        '''Trains the model for a fixed number of epochs.

        Returns a history object. Its `history` attribute is a record of
        training loss values at successive epochs,
        as well as validation loss values (if applicable).

        # Arguments
            data: dictionary mapping input names and outputs names to
                appropriate Numpy arrays. All arrays should contain
                the same number of samples.
            batch_size: int. Number of samples per gradient update.
            nb_epoch: int.
            verbose: 0 for no logging to stdout,
                1 for progress bar logging, 2 for one log line per epoch.
            callbacks: `keras.callbacks.Callback` list. List of callbacks
                to apply during training. See [callbacks](callbacks.md).
            validation_split: float (0. < x < 1). Fraction of the data to
                use as held-out validation data.
            validation_data: dictionary mapping input names and outputs names
                to appropriate Numpy arrays to be used as
                held-out validation data.
                All arrays should contain the same number of samples.
                Will override validation_split.
            shuffle: boolean. Whether to shuffle the samples at each epoch.
            class_weight: dictionary mapping output names to
                class weight dictionaries.
            sample_weight: dictionary mapping output names to
                numpy arrays of sample weights.
        '''
        if 'show_accuracy' in kwargs:
            kwargs.pop('show_accuracy')
            warnings.warn('The "show_accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))
        x = self._get_x(data)
        y = self._get_y(data)

        if type(validation_data) is tuple:
            raise Exception('Cannot used sample_weight with '
                            'validation data with legacy Graph model. '
                            'validation_data should be a dictionary.')
        if validation_data:
            val_x = self._get_x(validation_data)
            val_y = self._get_y(validation_data)
            validation_data = (val_x, val_y)
        return super(Graph, self).fit(x, y,
                                      batch_size=batch_size,
                                      nb_epoch=nb_epoch,
                                      verbose=verbose,
                                      callbacks=callbacks,
                                      validation_split=validation_split,
                                      validation_data=validation_data,
                                      shuffle=shuffle,
                                      class_weight=class_weight,
                                      sample_weight=sample_weight)

    def evaluate(self, data, batch_size=128,
                 verbose=0, sample_weight={}, **kwargs):
        '''Computes the loss on some input data, batch by batch.

        Returns the scalar test loss over the data,
        or a list of metrics values (starting with the test loss)
        if applicable.

        Arguments: see `fit` method.
        '''
        if 'show_accuracy' in kwargs:
            kwargs.pop('show_accuracy')
            warnings.warn('The "show_accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))
        x = self._get_x(data)
        y = self._get_y(data)
        return super(Graph, self).evaluate(x, y,
                                           batch_size=batch_size,
                                           verbose=verbose,
                                           sample_weight=sample_weight)

    def predict(self, data, batch_size=128, verbose=0):
        '''Generates output predictions for the input samples
        batch by batch.

        Arguments: see `fit` method.
        '''
        x = self._get_x(data)
        output_list = super(Graph, self).predict(x, batch_size=batch_size,
                                                 verbose=verbose)
        if not isinstance(output_list, list):
            output_list = [output_list]
        return dict(zip(self._graph_outputs, output_list))

    def train_on_batch(self, data,
                       class_weight={},
                       sample_weight={}, **kwargs):
        '''Single gradient update on a batch of samples.

        Returns the scalar train loss over the data,
        or a list of metrics values (starting with the test loss)
        if applicable.

        Arguments: see `fit` method.
        '''
        if 'accuracy' in kwargs:
            kwargs.pop('accuracy')
            warnings.warn('The "accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))
        x = self._get_x(data)
        y = self._get_y(data)
        return super(Graph, self).train_on_batch(x, y,
                                                 sample_weight=sample_weight,
                                                 class_weight=class_weight)

    def test_on_batch(self, data, sample_weight={}, **kwargs):
        '''Test the network on a single batch of samples.

        Returns the scalar test loss over the data,
        or a list of metrics values (starting with the test loss)
        if applicable.

        Arguments: see `fit` method.
        '''
        if 'accuracy' in kwargs:
            kwargs.pop('accuracy')
            warnings.warn('The "accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))
        x = self._get_x(data)
        y = self._get_y(data)
        return super(Graph, self).test_on_batch(x, y,
                                                sample_weight=sample_weight)

    def predict_on_batch(self, data):
        output_list = super(Graph, self).predict_on_batch(data)
        if not isinstance(output_list, list):
            output_list = [output_list]
        return dict(zip(self._graph_outputs, output_list))

    def fit_generator(self, generator, samples_per_epoch, nb_epoch,
                      verbose=1, callbacks=[],
                      validation_data=None, nb_val_samples=None,
                      class_weight={},
                      max_q_size=10, **kwargs):
        '''Fits a model on data generated batch-by-batch by a Python generator.
        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.

        # Arguments
            generator: a generator.
                The output of the generator must be either a tuple
                of dictionaries `(input_data, sample_weight)`
                or a dictionary `input_data`
                (mapping names of inputs and outputs to Numpy arrays).
                All arrays should contain the same number of samples.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `samples_per_epoch`
                samples have been seen by the model.
            samples_per_epoch: integer, number of samples to process before
                going to the next epoch.
            nb_epoch: integer, total number of iterations on the data.
            verbose: verbosity mode, 0, 1, or 2.
            callbacks: list of callbacks to be called during training.
            validation_data: dictionary mapping input names and outputs names
                to appropriate Numpy arrays to be used as
                held-out validation data, or a generator yielding such
                dictionaries. All arrays should contain the same number
                of samples. If a generator, will be called until more than
                `nb_val_samples` examples have been generated at the
                end of every epoch. These examples will then be used
                as the validation data.
            nb_val_samples: number of samples to use from validation
                generator at the end of every epoch.
            class_weight: dictionary mapping class indices to a weight
                for the class.

        # Returns
            A `History` object.

        # Examples

        ```python
            def generate_arrays_from_file(path):
                while 1:
                    f = open(path)
                    for line in f:
                        # create Numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2, 'output': y})
                    f.close()

            graph.fit_generator(generate_arrays_from_file('/my_file.txt'),
                                samples_per_epoch=10000, nb_epoch=10)
        ```
        '''
        if 'show_accuracy' in kwargs:
            kwargs.pop('show_accuracy')
            warnings.warn('The "show_accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if 'nb_worker' in kwargs:
            kwargs.pop('nb_worker')
            warnings.warn('The "nb_worker" argument is deprecated, '
                          'please remove it from your code.')
        if 'nb_val_worker' in kwargs:
            kwargs.pop('nb_val_worker')
            warnings.warn('The "nb_val_worker" argument is deprecated, '
                          'please remove it from your code.')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))

        self._train_on_batch = self.train_on_batch
        self.train_on_batch = super(Graph, self).train_on_batch
        self._evaluate = self.evaluate
        self.evaluate = super(Graph, self).evaluate

        if validation_data and type(validation_data) is tuple:
            raise Exception('Cannot use sample_weight with '
                            'validation_data in legacy Graph model.')
        if validation_data and type(validation_data) is dict:
            validation_data = (self._get_x(validation_data),
                               self._get_y(validation_data))

        original_generator = generator

        def fixed_generator():
            while 1:
                data = next(original_generator)
                if type(data) is tuple:
                    data, sample_weight = data
                    x = self._get_x(data)
                    y = self._get_y(data)
                    yield x, y, sample_weight
                else:
                    x = self._get_x(data)
                    y = self._get_y(data)
                    yield x, y

        generator = fixed_generator()
        history = super(Graph, self).fit_generator(generator,
                                                   samples_per_epoch,
                                                   nb_epoch,
                                                   verbose=verbose,
                                                   callbacks=callbacks,
                                                   validation_data=validation_data,
                                                   nb_val_samples=nb_val_samples,
                                                   class_weight=class_weight,
                                                   max_q_size=max_q_size)
        self.train_on_batch = self._train_on_batch
        self.evaluate = self._evaluate
        return history

    def evaluate_generator(self, generator, val_samples,
                           verbose=1, max_q_size=10, **kwargs):
        '''Evaluates the model on a generator. The generator should
        return the same kind of data with every yield as accepted
        by `evaluate`.

        If `show_accuracy`, it returns a tuple `(loss, accuracy)`,
        otherwise it returns the loss value.

        Arguments:
            generator:
                generator yielding dictionaries of the kind accepted
                by `evaluate`, or tuples of such dictionaries and
                associated dictionaries of sample weights.
            val_samples:
                total number of samples to generate from `generator`
                to use in validation.

            Other arguments are the same as for `fit`.
        '''
        if 'show_accuracy' in kwargs:
            kwargs.pop('show_accuracy')
            warnings.warn('The "show_accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if 'verbose' in kwargs:
            kwargs.pop('verbose')
            warnings.warn('The "verbose" argument is deprecated.')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))

        self._test_on_batch = self.test_on_batch
        self.test_on_batch = super(Graph, self).test_on_batch

        original_generator = generator

        def fixed_generator():
            while 1:
                data = next(original_generator)
                if type(data) is tuple:
                    data, sample_weight = data
                    x = self._get_x(data)
                    y = self._get_y(data)
                    yield x, y, sample_weight
                else:
                    x = self._get_x(data)
                    y = self._get_y(data)
                    yield x, y

        generator = fixed_generator()
        history = super(Graph, self).evaluate_generator(generator,
                                                        val_samples,
                                                        max_q_size=max_q_size)
        self.test_on_batch = self._test_on_batch
        return history

    # get_weights, set_weights: inherited
    def get_config(self):
        config = {'input_config': self._graph_input_config,
                  'node_config': self._graph_node_config,
                  'output_config': self._graph_output_config}
        nodes = {}
        for name, node in self._graph_nodes.items():
            nodes[name] = {'class_name': node.__class__.__name__,
                           'config': node.get_config()}
            if name in self._graph_shared_nodes_names:
                nodes[name]['shared'] = True
        config['nodes'] = nodes
        return copy.deepcopy(config)

    @classmethod
    def from_config(cls, config):
        # TODO: test legacy support
        from keras.utils.layer_utils import layer_from_config

        def normalize_legacy_config(conf):
            if 'class_name' not in conf:
                class_name = conf['name']
                name = conf.get('custom_name')
                conf['name'] = name
                new_config = {
                    'class_name': class_name,
                    'config': conf,
                }
                return new_config
            return conf

        graph = cls()
        inputs = config.get('input_config')
        for input in inputs:
            graph.add_input(**input)

        nodes = config.get('node_config')
        for node in nodes:
            layer_config = config['nodes'][node['name']]
            layer_config = normalize_legacy_config(layer_config)
            if 'layer' in node:
                # for add_shared_node
                node['layer'] = layer_from_config(node['layer'])
            else:
                layer = layer_from_config(layer_config)
                node['layer'] = layer

            node['create_output'] = False  # outputs will be added below
            if layer_config.get('shared'):
                graph.add_shared_node(**node)
            else:
                graph.add_node(**node)

        outputs = config.get('output_config')
        for output in outputs:
            graph.add_output(**output)
        return graph

    def load_weights(self, fname):
        if not self.built:
            self.build()
        super(Graph, self).load_weights(fname)
