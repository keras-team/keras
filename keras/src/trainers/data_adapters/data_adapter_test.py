from keras.src import testing
from keras.src.trainers.data_adapters import data_adapter_utils


class DataAdapterTest(testing.TestCase):
    """Base test class providing common assertions for DataAdapter tests."""

    def verify_super_batched_iterator(
        self,
        iterator,
        expected_super_batches,
        has_partial_batch=False,
        super_batch=2,
        batch_size=16,
        feature_dim=4,
        target_dim=2,
    ):
        """Verifies structure and shapes yielded by get_jax_iterator."""
        batches = list(iterator)
        expected_total = (
            expected_super_batches + 1
            if has_partial_batch
            else expected_super_batches
        )
        self.assertEqual(len(batches), expected_total)

        for batch in batches[:expected_super_batches]:
            batch_x, batch_y = batch
            self.assertEqual(
                batch_x.shape, (super_batch, batch_size, feature_dim)
            )
            self.assertEqual(
                batch_y.shape, (super_batch, batch_size, target_dim)
            )

        if has_partial_batch:
            remainder = batches[expected_super_batches]
            if isinstance(remainder, data_adapter_utils.PartialBatchList):
                self.assertEqual(len(remainder), 1)
                rem_x, rem_y = remainder[0]
                self.assertEqual(rem_x.shape, (batch_size, feature_dim))
                self.assertEqual(rem_y.shape, (batch_size, target_dim))
            else:
                rem_x, rem_y = remainder
                self.assertEqual(rem_x.shape, (1, batch_size, feature_dim))
                self.assertEqual(rem_y.shape, (1, batch_size, target_dim))
