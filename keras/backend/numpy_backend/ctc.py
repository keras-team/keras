import numpy as np


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    num_samples = y_pred.shape[0]
    num_classes = y_pred.shape[-1]
    log_prob = np.zeros((num_samples, 1))
    decoded_dense = -np.ones_like(y_pred[..., 0])
    decoded_length = np.zeros((num_samples,), dtype=np.int)
    if greedy:
        for i in range(num_samples):
            prob = y_pred[i]
            length = input_length[i]
            decoded = np.argmax(prob[:length], axis=-1)
            log_prob[i] = -np.sum(np.log(prob[np.arange(length), decoded]))
            decoded = _remove_repeats(decoded)
            decoded = _remove_blanks(decoded, num_classes)
            decoded_length[i] = len(decoded)
            decoded_dense[i, :len(decoded)] = decoded
        return decoded_dense[:, :np.max(decoded_length)], log_prob
    else:
        raise "not supported yet"


def _remove_repeats(inds):
    is_not_repeat = np.insert(np.diff(inds).astype(np.bool), 0, True)
    return inds[is_not_repeat]


def _remove_blanks(inds, num_classes):
    return inds[inds < (num_classes - 1)]
