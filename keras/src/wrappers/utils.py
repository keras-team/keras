try:
    import sklearn
    from sklearn.base import BaseEstimator
    from sklearn.base import TransformerMixin
except ImportError:
    sklearn = None

    class BaseEstimator:
        pass

    class TransformerMixin:
        pass


def assert_sklearn_installed(symbol_name):
    if sklearn is None:
        raise ImportError(
            f"{symbol_name} requires `scikit-learn` to be installed. "
            "Run `pip install scikit-learn` to install it."
        )


def _check_model(model):
    """Check whether the model need sto be compiled."""
    # compile model if user gave us an un-compiled model
    if not model.compiled or not model.loss or not model.optimizer:
        raise RuntimeError(
            "Given model needs to be compiled, and have a loss and an "
            "optimizer."
        )


class TargetReshaper(TransformerMixin, BaseEstimator):
    """Convert 1D targets to 2D and back.

    For use in pipelines with transformers that only accept
    2D inputs, like OneHotEncoder and OrdinalEncoder.

    Attributes:
        ndim_ : int
            Dimensions of y that the transformer was trained on.
    """

    def fit(self, y):
        """Fit the transformer to a target y.

        Returns:
            TargetReshaper
                A reference to the current instance of TargetReshaper.
        """
        self.ndim_ = y.ndim
        return self

    def transform(self, y):
        """Makes 1D y 2D.

        Args:
            y : np.ndarray
                Target y to be transformed.

        Returns:
            np.ndarray
                A numpy array, of dimension at least 2.
        """
        if y.ndim == 1:
            return y.reshape(-1, 1)
        return y

    def inverse_transform(self, y):
        """Revert the transformation of transform.

        Args:
            y: np.ndarray
                Transformed numpy array.

        Returns:
            np.ndarray
                If the transformer was fit to a 1D numpy array,
                and a 2D numpy array with a singleton second dimension
                is passed, it will be squeezed back to 1D. Otherwise, it
                will eb left untouched.
        """
        sklearn.base.check_is_fitted(self)
        xp, _ = sklearn.utils._array_api.get_namespace(y)
        if self.ndim_ == 1 and y.ndim == 2:
            return xp.squeeze(y, axis=1)
        return y
