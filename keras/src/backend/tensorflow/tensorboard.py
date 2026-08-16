from keras.src.utils.module_utils import tensorflow as tf


def start_trace(logdir):
    tf.profiler.experimental.start(logdir=logdir)


def stop_trace(save):
    tf.profiler.experimental.stop(save=save)


def start_batch_trace(batch):
    batch_trace_context = tf.profiler.experimental.Trace(
        "Profiled batch", step_num=batch
    )
    batch_trace_context.__enter__()
    return batch_trace_context


def stop_batch_trace(batch_trace_context):
    batch_trace_context.__exit__(None, None, None)
