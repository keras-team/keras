/*
    Copyright (c) 2005-2021 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_tbb_H
#define __TBB_tbb_H

/**
    This header bulk-includes declarations or definitions of all the functionality
    provided by TBB (save for tbbmalloc and 3rd party dependent headers).

    If you use only a few TBB constructs, consider including specific headers only.
    Any header listed below can be included independently of others.
**/

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/blocked_range2d.h"
#include "oneapi/tbb/blocked_range3d.h"
#if TBB_PREVIEW_BLOCKED_RANGE_ND
#include "tbb/blocked_rangeNd.h"
#endif
#include "oneapi/tbb/cache_aligned_allocator.h"
#include "oneapi/tbb/combinable.h"
#include "oneapi/tbb/concurrent_hash_map.h"
#if TBB_PREVIEW_CONCURRENT_LRU_CACHE
#include "tbb/concurrent_lru_cache.h"
#endif
#include "oneapi/tbb/concurrent_priority_queue.h"
#include "oneapi/tbb/concurrent_queue.h"
#include "oneapi/tbb/concurrent_unordered_map.h"
#include "oneapi/tbb/concurrent_unordered_set.h"
#include "oneapi/tbb/concurrent_map.h"
#include "oneapi/tbb/concurrent_set.h"
#include "oneapi/tbb/concurrent_vector.h"
#include "oneapi/tbb/enumerable_thread_specific.h"
#include "oneapi/tbb/flow_graph.h"
#include "oneapi/tbb/global_control.h"
#include "oneapi/tbb/info.h"
#include "oneapi/tbb/null_mutex.h"
#include "oneapi/tbb/null_rw_mutex.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/parallel_for_each.h"
#include "oneapi/tbb/parallel_invoke.h"
#include "oneapi/tbb/parallel_pipeline.h"
#include "oneapi/tbb/parallel_reduce.h"
#include "oneapi/tbb/parallel_scan.h"
#include "oneapi/tbb/parallel_sort.h"
#include "oneapi/tbb/partitioner.h"
#include "oneapi/tbb/queuing_mutex.h"
#include "oneapi/tbb/queuing_rw_mutex.h"
#include "oneapi/tbb/spin_mutex.h"
#include "oneapi/tbb/spin_rw_mutex.h"
#include "oneapi/tbb/task.h"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"
#include "oneapi/tbb/task_scheduler_observer.h"
#include "oneapi/tbb/tbb_allocator.h"
#include "oneapi/tbb/tick_count.h"
#include "oneapi/tbb/version.h"

#endif /* __TBB_tbb_H */
