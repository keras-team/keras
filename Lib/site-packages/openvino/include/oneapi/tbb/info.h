/*
    Copyright (c) 2019-2021 Intel Corporation

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

#ifndef __TBB_info_H
#define __TBB_info_H

#include "detail/_config.h"
#include "detail/_namespace_injection.h"

#if __TBB_ARENA_BINDING
#include <vector>

namespace tbb {
namespace detail {

namespace d1{

using numa_node_id = int;
using core_type_id = int;

// TODO: consider version approach to resolve backward compatibility potential issues.
struct constraints {
#if !__TBB_CPP20_PRESENT
    constraints(numa_node_id id = -1, int maximal_concurrency = -1)
        : numa_id(id)
        , max_concurrency(maximal_concurrency)
#if __TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION_PRESENT
        , core_type(-1)
        , max_threads_per_core(-1)
#endif
    {}
#endif /*!__TBB_CPP20_PRESENT*/

    constraints& set_numa_id(numa_node_id id) {
        numa_id = id;
        return *this;
    }
    constraints& set_max_concurrency(int maximal_concurrency) {
        max_concurrency = maximal_concurrency;
        return *this;
    }
#if __TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION_PRESENT
    constraints& set_core_type(core_type_id id) {
        core_type = id;
        return *this;
    }
    constraints& set_max_threads_per_core(int threads_number) {
        max_threads_per_core = threads_number;
        return *this;
    }
#endif

    numa_node_id numa_id = -1;
    int max_concurrency = -1;
#if __TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION_PRESENT
    core_type_id core_type = -1;
    int max_threads_per_core = -1;
#endif
};

} // namespace d1

namespace r1 {
unsigned __TBB_EXPORTED_FUNC numa_node_count();
void __TBB_EXPORTED_FUNC fill_numa_indices(int* index_array);
int __TBB_EXPORTED_FUNC numa_default_concurrency(int numa_id);

// Reserved fields are required to save binary backward compatibility in case of future changes.
// They must be defined to 0 at this moment.
unsigned __TBB_EXPORTED_FUNC core_type_count(intptr_t reserved = 0);
void __TBB_EXPORTED_FUNC fill_core_type_indices(int* index_array, intptr_t reserved = 0);

int __TBB_EXPORTED_FUNC constraints_default_concurrency(const d1::constraints& c, intptr_t reserved = 0);
int __TBB_EXPORTED_FUNC constraints_threads_per_core(const d1::constraints& c, intptr_t reserved = 0);
} // namespace r1

namespace d1 {

inline std::vector<numa_node_id> numa_nodes() {
    std::vector<numa_node_id> node_indices(r1::numa_node_count());
    r1::fill_numa_indices(node_indices.data());
    return node_indices;
}

inline int default_concurrency(numa_node_id id = -1) {
    return r1::numa_default_concurrency(id);
}

#if __TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION_PRESENT
inline std::vector<core_type_id> core_types() {
    std::vector<int> core_type_indexes(r1::core_type_count());
    r1::fill_core_type_indices(core_type_indexes.data());
    return core_type_indexes;
}

inline int default_concurrency(constraints c) {
    return r1::constraints_default_concurrency(c);
}
#endif /*__TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION_PRESENT*/

} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::numa_node_id;
#if __TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION_PRESENT
using detail::d1::core_type_id;
#endif

namespace info {
using detail::d1::numa_nodes;
#if __TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION_PRESENT
using detail::d1::core_types;
#endif

using detail::d1::default_concurrency;
} // namespace info
} // namespace v1

} // namespace tbb

#endif /*__TBB_ARENA_BINDING*/

#endif /*__TBB_info_H*/
