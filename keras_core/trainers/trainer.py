class Trainer:
    def compile(self, optimizer, loss=None, metrics=None, jit_compile=False):
        self.optimizer = optimizer
        self.loss = loss
        self._metrics = metrics
        self.jit_compile = jit_compile

    def call(self, inputs):
        raise NotImplementedError

    @property
    def metrics(self):
        return self._metrics

    def reset_metrics(self):
        for m in self.metrics:
            m.reset_state()

    def train_step(self, data):
        raise NotImplementedError

    def test_step(self, data):
        raise NotImplementedError

    def predict_step(self, data):
        raise NotImplementedError

    def make_train_function(self):
        raise NotImplementedError

    def make_test_function(self):
        raise NotImplementedError

    def make_predict_function(self):
        raise NotImplementedError

    def fit(self, x, y=None):
        raise NotImplementedError

    def evaluate(self, x, y=None):
        raise NotImplementedError

    def predict(self, x, y=None):
        raise NotImplementedError

    def get_compile_config(self):
        return {}

    def compile_from_config(self):
        pass
