import numpy as np
from scipy.stats import pearsonr

from keras.src import testing
from keras.src.metrics import ConcordanceCorrelationCoefficient
from keras.src.metrics import PearsonCorrelationCoefficient
from keras.src.metrics import correlation_metrics


class CorrelationsTest(testing.TestCase):
    def setUp(self):
        # Sample data for testing
        self.y_true = np.array(
            [[0, 1, 0.5], [1, 1, 0.2], [1, 1, 0.1], [0.1, 0.7, 0.0]],
            dtype="float32",
        )
        self.y_pred = np.array(
            [[0.1, 0.9, 0.5], [1, 0.9, 0.2], [0.2, 0.8, 0], [0.3, 0.3, 0.9]],
            dtype="float32",
        )

        self.ccc_expected = np.array(
            [0.97560976, 0.98765432, 0.46511628, -0.46376812]
        )
        # self.pcc_expected = np.array([1, 0.99339927, 0.69337525, -0.60999428])
        self.pcc_expected = np.array(
            [
                pearsonr(yt, yp).statistic
                for yt, yp in zip(self.y_true, self.y_pred)
            ]
        )

    def test_pearson_function(self):
        """Test the functional API for Pearson Correlation Coefficient."""
        result = correlation_metrics.pearson_correlation_coefficient(
            self.y_true, self.y_pred, axis=-1
        )
        self.assertAllClose(result, self.pcc_expected)

    def test_concordance_function(self):
        """Test the functional API for Concordance Correlation Coefficient."""
        result = correlation_metrics.concordance_correlation_coefficient(
            self.y_true, self.y_pred, axis=-1
        )
        self.assertAllClose(result, self.ccc_expected)

    def test_pearson_class(self):
        """Test the PearsonCorrelationCoefficient metric class."""
        m = PearsonCorrelationCoefficient(axis=-1, dtype="float32")
        m.update_state(self.y_true[:2], self.y_pred[:2])
        self.assertAllClose(m.result(), np.mean(self.pcc_expected[:2]))
        m.update_state(self.y_true[2:], self.y_pred[2:])
        self.assertAllClose(m.result(), np.mean(self.pcc_expected))

    def test_concordance_class(self):
        """Test the ConcordanceCorrelationCoefficient metric class."""
        m = ConcordanceCorrelationCoefficient(axis=-1, dtype="float32")
        m.update_state(self.y_true[:2], self.y_pred[:2])
        self.assertAllClose(m.result(), np.mean(self.ccc_expected[:2]))
        m.update_state(self.y_true[2:], self.y_pred[2:])
        self.assertAllClose(m.result(), np.mean(self.ccc_expected))

    def test_pearson_config(self):
        """Test the get_config method for PearsonCorrelationCoefficient."""
        m = PearsonCorrelationCoefficient(axis=-1, dtype="float16")
        config = m.get_config()
        self.assertEqual(config["axis"], -1)
        self.assertEqual(config["dtype"], "float16")
        self.assertEqual(config["name"], "pearson_correlation_coefficient")

    def test_concordance_config(self):
        """Test the get_config method for ConcordanceCorrelationCoefficient."""
        m = ConcordanceCorrelationCoefficient(axis=-1, dtype="float32")
        config = m.get_config()
        self.assertEqual(config["axis"], -1)
        self.assertEqual(config["dtype"], "float32")
        self.assertEqual(config["name"], "concordance_correlation_coefficient")
