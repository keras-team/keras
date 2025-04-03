# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Typing information for the constructs from our .c files."""

from typing import Any, Dict

from coverage.types import TFileDisposition, TTraceData, TTraceFn, Tracer

class CFileDisposition(TFileDisposition):
    """CFileDisposition is in ctracer/filedisp.c"""
    canonical_filename: Any
    file_tracer: Any
    has_dynamic_filename: Any
    original_filename: Any
    reason: Any
    source_filename: Any
    trace: Any
    def __init__(self) -> None: ...

class CTracer(Tracer):
    """CTracer is in ctracer/tracer.c"""
    check_include: Any
    concur_id_func: Any
    data: TTraceData
    disable_plugin: Any
    file_tracers: Any
    should_start_context: Any
    should_trace: Any
    should_trace_cache: Any
    switch_context: Any
    lock_data: Any
    unlock_data: Any
    trace_arcs: Any
    warn: Any
    def __init__(self) -> None: ...
    def activity(self) -> bool: ...
    def get_stats(self) -> Dict[str, int]: ...
    def reset_activity(self) -> Any: ...
    def start(self) -> TTraceFn: ...
    def stop(self) -> None: ...
