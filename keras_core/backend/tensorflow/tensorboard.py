import tensorflow as tf


def start_trace(logdir):
    tf.profiler.experimental.start(logdir=logdir)


def stop_trace(save):
    tf.profiler.experimental.stop(save=save)
