from keras.src.utils.module_utils import jax


def start_trace(logdir):
    if logdir:
        jax.profiler.start_trace(logdir)


def stop_trace(save):
    if save:
        jax.profiler.stop_trace()


def start_batch_trace(batch):
    batch_trace_context = jax.profiler.TraceAnnotation(
        f"Profiled batch {batch}"
    )
    batch_trace_context.__enter__()
    return batch_trace_context


def stop_batch_trace(batch_trace_context):
    batch_trace_context.__exit__(None, None, None)
