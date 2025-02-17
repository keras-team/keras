import numpy as np
from scipy.stats import pearsonr

from keras.src import testing
from keras.src.metrics import ConcordanceCorrelation
from keras.src.metrics import PearsonCorrelation
from keras.src.metrics import correlation_metrics


class CorrelationsTest(testing.TestCase):
    def _get_data(self):
        # Sample data for testing
        y_true = np.array(
            [[0, 1, 0.5], [1, 1, 0.2], [1, 1, 0.1], [0.1, 0.7, 0.0]],
            dtype="float32",
        )
        y_pred = np.array(
            [[0.1, 0.9, 0.5], [1, 0.9, 0.2], [0.2, 0.8, 0], [0.3, 0.3, 0.9]],
            dtype="float32",
        )

        ccc_expected = np.array(
            [0.97560976, 0.98765432, 0.46511628, -0.46376812]
        )
        # pcc_expected = np.array([1, 0.99339927, 0.69337525, -0.60999428])
        pcc_expected = np.array(
            [pearsonr(yt, yp).statistic for yt, yp in zip(y_true, y_pred)]
        )
        return y_true, y_pred, ccc_expected, pcc_expected

    def test_pearson_function(self):
        """Test the functional API for Pearson Correlation Coefficient."""
        y_true, y_pred, _, pcc_expected = self._get_data()
        result = correlation_metrics.pearson_correlation(
            y_true, y_pred, axis=-1
        )
        self.assertAllClose(result, pcc_expected)

    def test_concordance_function(self):
        """Test the functional API for Concordance Correlation Coefficient."""
        y_true, y_pred, ccc_expected, _ = self._get_data()
        result = correlation_metrics.concordance_correlation(
            y_true, y_pred, axis=-1
        )
        self.assertAllClose(result, ccc_expected)

    def test_pearson_class(self):
        """Test the PearsonCorrelation metric class."""
        y_true, y_pred, _, pcc_expected = self._get_data()
        m = PearsonCorrelation(axis=-1, dtype="float32")
        m.update_state(y_true[:2], y_pred[:2])
        self.assertAllClose(m.result(), np.mean(pcc_expected[:2]))
        m.update_state(y_true[2:], y_pred[2:])
        self.assertAllClose(m.result(), np.mean(pcc_expected))

    def test_concordance_class(self):
        """Test the ConcordanceCorrelation metric class."""
        y_true, y_pred, ccc_expected, _ = self._get_data()
        m = ConcordanceCorrelation(axis=-1, dtype="float32")
        m.update_state(y_true[:2], y_pred[:2])
        self.assertAllClose(m.result(), np.mean(ccc_expected[:2]))
        m.update_state(y_true[2:], y_pred[2:])
        self.assertAllClose(m.result(), np.mean(ccc_expected))

    def test_pearson_config(self):
        """Test the get_config method for PearsonCorrelation."""
        m = PearsonCorrelation(axis=-1, dtype="float16")
        config = m.get_config()
        self.assertEqual(config["axis"], -1)
        self.assertEqual(config["dtype"], "float16")
        self.assertEqual(config["name"], "pearson_correlation")

    def test_concordance_config(self):
        """Test the get_config method for ConcordanceCorrelation."""
        m = ConcordanceCorrelation(axis=-1, dtype="float32")
        config = m.get_config()
        self.assertEqual(config["axis"], -1)
        self.assertEqual(config["dtype"], "float32")
        self.assertEqual(config["name"], "concordance_correlation")
