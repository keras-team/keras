import numpy as np

def pad_sequences(seqs, maxlen=None, dtype='int32'):
    """
        Pad each sequence to the same lenght: 
        the lenght of the longuest sequence.

        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen.
    """
    lengths = [len(s) for s in seqs]

    nb_samples = len(seqs)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = np.zeros((nb_samples, maxlen)).astype(dtype)
    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx]] = s[:maxlen]

    return x