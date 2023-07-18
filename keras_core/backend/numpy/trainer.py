class NumpyTrainer:
    def fit(self):
        raise NotImplementedError("Trainer not implemented for NumPy backend.")

    def predict(self):
        raise NotImplementedError("Trainer not implemented for NumPy backend.")

    def evaluate(self):
        raise NotImplementedError("Trainer not implemented for NumPy backend.")

    def train_on_batch(self):
        raise NotImplementedError("Trainer not implemented for NumPy backend.")

    def test_on_batch(self):
        raise NotImplementedError("Trainer not implemented for NumPy backend.")

    def predict_on_batch(self):
        raise NotImplementedError("Trainer not implemented for NumPy backend.")
