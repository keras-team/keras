import numpy as np
from .. import backend as K

try:
    import librosa
except ImportError:
    librosa = None


TAGS = ['rock', 'pop', 'alternative', 'indie', 'electronic',
        'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
        'beautiful', 'metal', 'chillout', 'male vocalists',
        'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica',
        '80s', 'folk', '90s', 'chill', 'instrumental', 'punk',
        'oldies', 'blues', 'hard rock', 'ambient', 'acoustic',
        'experimental', 'female vocalist', 'guitar', 'Hip-Hop',
        '70s', 'party', 'country', 'easy listening',
        'sexy', 'catchy', 'funk', 'electro', 'heavy metal',
        'Progressive rock', '60s', 'rnb', 'indie pop',
        'sad', 'House', 'happy']


def preprocess_input(audio_path, dim_ordering='default'):
    """Reads an audio file and outputs a Mel-spectrogram.

    # Arguments
        audio_path: path to the target audio file.
        dim_ordering: data format for the output spectrogram image.

    # Returns
        3D Numpy tensor encoding the Mel-spectrogram.

    # Raises
        ImportError: if librosa is not available.
    """
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if librosa is None:
        raise ImportError('Librosa is required to process audio files. '
                          'Install it via `pip install librosa` or visit '
                          'http://librosa.github.io/librosa/ for details.')

    # mel-spectrogram parameters
    sr = 12000
    n_fft = 512
    n_mels = 96
    hop_length = 256
    duration = 29.12

    src, sr = librosa.load(audio_path, sr=sr)
    n_sample = src.shape[0]
    n_sample_wanted = int(duration * sr)

    # trim the signal at the center
    if n_sample < n_sample_wanted:  # if too short
        src = np.hstack((src, np.zeros((int(duration * sr) - n_sample,))))
    elif n_sample > n_sample_wanted:  # if too long
        src = src[(n_sample - n_sample_wanted) // 2:
                  (n_sample + n_sample_wanted) // 2]

    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    x = logam(melgram(y=src, sr=sr, hop_length=hop_length,
                      n_fft=n_fft, n_mels=n_mels) ** 2,
              ref_power=1.0)

    if dim_ordering == 'th':
        x = np.expand_dims(x, axis=0)
    elif dim_ordering == 'tf':
        x = np.expand_dims(x, axis=3)
    return x


def decode_predictions(preds, top_n=5):
    """Decode the output of a music tagger model.

    # Arguments
        preds: 2-dimensional numpy array
        top_n: integer in [0, 50], number of items to show.

    # Returns
        Decoded output.
    """
    assert len(preds.shape) == 2 and preds.shape[1] == 50
    results = []
    for pred in preds:
        result = zip(TAGS, pred)
        result = sorted(result, key=lambda x: x[1], reverse=True)
        results.append(result[:top_n])
    return results
