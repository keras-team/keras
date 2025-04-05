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

#ifndef __TBB_task_H
#define __TBB_task_H

#include "detail/_config.h"
#include "detail/_namespace_injection.h"
#include "detail/_task.h"

namespace tbb {
inline namespace v1 {
namespace task {
#if __TBB_RESUMABLE_TASKS
    using detail::d1::suspend_point;
    using detail::d1::resume;
    using detail::d1::suspend;
#endif /* __TBB_RESUMABLE_TASKS */
    using detail::d1::current_context;
} // namespace task
} // namespace v1
} // namespace tbb

#endif /* __TBB_task_H */
