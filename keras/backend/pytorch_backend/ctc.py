def ctc_label_dense_to_sparse(labels, label_lengths):
    raise NotImplementedError


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    raise NotImplementedError


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100,
               top_paths=1):
    raise NotImplementedError
