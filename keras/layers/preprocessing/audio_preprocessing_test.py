import numpy as np
import pytest
from absl.testing import parameterized

from keras import layers
from keras import testing


class MelSpectrogramTest(testing.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_mel_spectrogram_basics(self):
        self.run_layer_test(
            layers.MelSpectrogram,
            init_kwargs={
                "num_mel_bins": 80,
                "sampling_rate": 8000,
                "sequence_stride": 128,
                "fft_length": 2048,
                "dtype": "float32",
            },
            input_shape=(2, 16000),
            expected_output_shape=(2, 80, 126),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.MelSpectrogram,
            init_kwargs={
                "num_mel_bins": 80,
                "sampling_rate": 8000,
                "sequence_stride": 128,
                "fft_length": 2048,
            },
            input_shape=(16000,),
            expected_output_shape=(80, 126),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.parameters(
        [
            ((2, 16000), 80, 128, 2048, 8000),
            ((16000,), 80, 128, 2048, 8000),
            ((2, 16001), 80, 128, 2048, 16000),
            ((16001,), 80, 128, 2048, 8000),
            ((2, 16000), 128, 64, 512, 32000),
            ((16000,), 128, 64, 512, 32000),
        ]
    )
    def test_output_shape(
        self,
        input_shape,
        num_mel_bins,
        sequence_stride,
        fft_length,
        sampling_rate,
    ):
        audios = np.random.random(input_shape)
        out = layers.MelSpectrogram(
            num_mel_bins=num_mel_bins,
            sequence_stride=sequence_stride,
            fft_length=fft_length,
            sampling_rate=sampling_rate,
        )(audios)
        if len(input_shape) == 1:
            ref_shape = (
                num_mel_bins,
                (input_shape[0] + sequence_stride + 1) // sequence_stride,
            )
        else:
            ref_shape = (
                input_shape[0],
                num_mel_bins,
                (input_shape[1] + sequence_stride + 1) // sequence_stride,
            )
        self.assertEqual(tuple(out.shape), ref_shape)
