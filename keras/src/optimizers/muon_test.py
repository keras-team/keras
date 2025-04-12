import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.layers import Dense
from keras.src.layers import Embedding
from keras.src.optimizers.muon import Muon


class MuonTest(testing.TestCase):
    def test_config(self):
        optimizer = Muon(
            learning_rate=0.5,
            epsilon=1e-5,
        )
        self.run_class_serialization_test(optimizer)

    def test_Newton_Schulz(self):
        optimizer = Muon()
        tensor_input = ops.array([[0.2499, 0.9105], [0.2655, 0.8824]])
        except_output = ops.array([[-0.4422, 0.6457], [0.7285, 0.2968]])
        output = optimizer.zeropower_via_newtonschulz5(tensor_input, 5)
        self.assertAllClose(output, except_output, rtol=1e-3, atol=1e-3)

    def test_adamw_single_step(self):
        optimizer = Muon()
        grads = ops.array([1.0, 6.0, 7.0, 2.0])
        vars = backend.Variable([1.0, 2.0, 3.0, 4.0], name="test_vars")
        optimizer.build([vars])
        optimizer._adamw_update_step(grads, vars, 0.5)
        self.assertAllClose(vars, [0.5, 1.5, 2.5, 3.5], rtol=1e-4, atol=1e-4)

    def test_should_use_adamw(self):
        vars = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        optimizer = Muon(exclude_layers=["var"])
        self.assertAllClose(
            True,
            optimizer._should_use_adamw(vars),
        )
        embeding = Embedding(2, 2)
        embeding.build()
        self.assertAllClose(
            True,
            optimizer._should_use_adamw(embeding.weights[0]),
        )
        vars = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        optimizer = Muon()
        self.assertAllClose(
            False,
            optimizer._should_use_adamw(vars),
        )
        dense = Dense(2)
        dense.build([None, 2])
        self.assertAllClose(
            False,
            optimizer._should_use_adamw(dense.weights[0]),
        )

    def test_muon_single_step(self):
        optimizer = Muon(
            learning_rate=0.5,
            weight_decay=0,
        )
        grads = ops.array([[1.0, 6.0], [7.0, 2.0]])
        vars = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        optimizer.build([vars])
        optimizer._muon_update_step(grads, vars, 0.5)
        self.assertAllClose(
            vars, [[1.13, 1.51], [2.57, 4.06]], rtol=1e-2, atol=1e-2
        )

    def test_clip_norm(self):
        optimizer = Muon(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = Muon(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])
