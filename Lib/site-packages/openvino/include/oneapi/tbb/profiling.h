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

#ifndef __TBB_profiling_H
#define __TBB_profiling_H

#include "detail/_config.h"
#include <cstdint>

#include <string>

namespace tbb {
namespace detail {
inline namespace d0 {
    // include list of index names
    #define TBB_STRING_RESOURCE(index_name,str) index_name,
    enum string_resource_index : std::uintptr_t {
        #include "detail/_string_resource.h"
        NUM_STRINGS
    };
    #undef TBB_STRING_RESOURCE

    enum itt_relation
    {
    __itt_relation_is_unknown = 0,
    __itt_relation_is_dependent_on,         /**< "A is dependent on B" means that A cannot start until B completes */
    __itt_relation_is_sibling_of,           /**< "A is sibling of B" means that A and B were created as a group */
    __itt_relation_is_parent_of,            /**< "A is parent of B" means that A created B */
    __itt_relation_is_continuation_of,      /**< "A is continuation of B" means that A assumes the dependencies of B */
    __itt_relation_is_child_of,             /**< "A is child of B" means that A was created by B (inverse of is_parent_of) */
    __itt_relation_is_continued_by,         /**< "A is continued by B" means that B assumes the dependencies of A (inverse of is_continuation_of) */
    __itt_relation_is_predecessor_to        /**< "A is predecessor to B" means that B cannot start until A completes (inverse of is_dependent_on) */
    };

//! Unicode support
#if (_WIN32||_WIN64) && !__MINGW32__
    //! Unicode character type. Always wchar_t on Windows.
    using tchar = wchar_t;
#else /* !WIN */
    using tchar = char;
#endif /* !WIN */

} // namespace d0
} // namespace detail
} // namespace tbb

#include <atomic>
#if _WIN32||_WIN64
#include <stdlib.h>  /* mbstowcs_s */
#endif
// Need these to work regardless of tools support
namespace tbb {
namespace detail {
namespace d1 {
    enum notify_type {prepare=0, cancel, acquired, releasing, destroy};
    enum itt_domain_enum { ITT_DOMAIN_FLOW=0, ITT_DOMAIN_MAIN=1, ITT_DOMAIN_ALGO=2, ITT_NUM_DOMAINS };
} // namespace d1

namespace r1 {
    void __TBB_EXPORTED_FUNC call_itt_notify(int t, void* ptr);
    void __TBB_EXPORTED_FUNC create_itt_sync(void* ptr, const tchar* objtype, const tchar* objname);
    void __TBB_EXPORTED_FUNC itt_make_task_group(d1::itt_domain_enum domain, void* group, unsigned long long group_extra,
        void* parent, unsigned long long parent_extra, string_resource_index name_index);
    void __TBB_EXPORTED_FUNC itt_task_begin(d1::itt_domain_enum domain, void* task, unsigned long long task_extra,
        void* parent, unsigned long long parent_extra, string_resource_index name_index);
    void __TBB_EXPORTED_FUNC itt_task_end(d1::itt_domain_enum domain);
    void __TBB_EXPORTED_FUNC itt_set_sync_name(void* obj, const tchar* name);
    void __TBB_EXPORTED_FUNC itt_metadata_str_add(d1::itt_domain_enum domain, void* addr, unsigned long long addr_extra,
        string_resource_index key, const char* value);
    void __TBB_EXPORTED_FUNC itt_metadata_ptr_add(d1::itt_domain_enum domain, void* addr, unsigned long long addr_extra,
        string_resource_index key, void* value);
    void __TBB_EXPORTED_FUNC itt_relation_add(d1::itt_domain_enum domain, void* addr0, unsigned long long addr0_extra,
        itt_relation relation, void* addr1, unsigned long long addr1_extra);
    void __TBB_EXPORTED_FUNC itt_region_begin(d1::itt_domain_enum domain, void* region, unsigned long long region_extra,
        void* parent, unsigned long long parent_extra, string_resource_index /* name_index */);
    void __TBB_EXPORTED_FUNC itt_region_end(d1::itt_domain_enum domain, void* region, unsigned long long region_extra);
} // namespace r1

namespace d1 {
#if TBB_USE_PROFILING_TOOLS && (_WIN32||_WIN64) && !__MINGW32__
    inline std::size_t multibyte_to_widechar(wchar_t* wcs, const char* mbs, std::size_t bufsize) {
        std::size_t len;
        mbstowcs_s(&len, wcs, bufsize, mbs, _TRUNCATE);
        return len;   // mbstowcs_s counts null terminator
    }
#endif

#if TBB_USE_PROFILING_TOOLS
    inline void create_itt_sync(void *ptr, const char *objtype, const char *objname) {
#if (_WIN32||_WIN64) && !__MINGW32__
        std::size_t len_type = multibyte_to_widechar(nullptr, objtype, 0);
        wchar_t *type = new wchar_t[len_type];
        multibyte_to_widechar(type, objtype, len_type);
        std::size_t len_name = multibyte_to_widechar(nullptr, objname, 0);
        wchar_t *name = new wchar_t[len_name];
        multibyte_to_widechar(name, objname, len_name);
#else // WIN
        const char *type = objtype;
        const char *name = objname;
#endif
        r1::create_itt_sync(ptr, type, name);

#if (_WIN32||_WIN64) && !__MINGW32__
        delete[] type;
        delete[] name;
#endif // WIN
    }

// Distinguish notifications on task for reducing overheads
#if TBB_USE_PROFILING_TOOLS == 2
    inline void call_itt_task_notify(d1::notify_type t, void *ptr) {
        r1::call_itt_notify((int)t, ptr);
    }
#else
    inline void call_itt_task_notify(d1::notify_type, void *) {}
#endif // TBB_USE_PROFILING_TOOLS

    inline void call_itt_notify(d1::notify_type t, void *ptr) {
        r1::call_itt_notify((int)t, ptr);
    }

#if (_WIN32||_WIN64) && !__MINGW32__
    inline void itt_set_sync_name(void* obj, const wchar_t* name) {
        r1::itt_set_sync_name(obj, name);
    }
    inline void itt_set_sync_name(void* obj, const char* name) {
        std::size_t len_name = multibyte_to_widechar(nullptr, name, 0);
        wchar_t *obj_name = new wchar_t[len_name];
        multibyte_to_widechar(obj_name, name, len_name);
        r1::itt_set_sync_name(obj, obj_name);
        delete[] obj_name;
    }
#else
    inline void itt_set_sync_name( void* obj, const char* name) {
        r1::itt_set_sync_name(obj, name);
    }
#endif //WIN

    inline void itt_make_task_group(itt_domain_enum domain, void* group, unsigned long long group_extra,
        void* parent, unsigned long long parent_extra, string_resource_index name_index) {
        r1::itt_make_task_group(domain, group, group_extra, parent, parent_extra, name_index);
    }

    inline void itt_metadata_str_add( itt_domain_enum domain, void *addr, unsigned long long addr_extra,
                                        string_resource_index key, const char *value ) {
        r1::itt_metadata_str_add( domain, addr, addr_extra, key, value );
    }

    inline void register_node_addr(itt_domain_enum domain, void *addr, unsigned long long addr_extra,
        string_resource_index key, void *value) {
        r1::itt_metadata_ptr_add(domain, addr, addr_extra, key, value);
    }

    inline void itt_relation_add( itt_domain_enum domain, void *addr0, unsigned long long addr0_extra,
                                    itt_relation relation, void *addr1, unsigned long long addr1_extra ) {
        r1::itt_relation_add( domain, addr0, addr0_extra, relation, addr1, addr1_extra );
    }

    inline void itt_task_begin( itt_domain_enum domain, void *task, unsigned long long task_extra,
                                                    void *parent, unsigned long long parent_extra, string_resource_index name_index ) {
        r1::itt_task_begin( domain, task, task_extra, parent, parent_extra, name_index );
    }

    inline void itt_task_end( itt_domain_enum domain ) {
        r1::itt_task_end( domain );
    }

    inline void itt_region_begin( itt_domain_enum domain, void *region, unsigned long long region_extra,
                                    void *parent, unsigned long long parent_extra, string_resource_index name_index ) {
        r1::itt_region_begin( domain, region, region_extra, parent, parent_extra, name_index );
    }

    inline void itt_region_end( itt_domain_enum domain, void *region, unsigned long long region_extra  ) {
        r1::itt_region_end( domain, region, region_extra );
    }
#else
    inline void create_itt_sync(void* /*ptr*/, const char* /*objtype*/, const char* /*objname*/) {}

    inline void call_itt_notify(notify_type /*t*/, void* /*ptr*/) {}

    inline void call_itt_task_notify(notify_type /*t*/, void* /*ptr*/) {}
#endif // TBB_USE_PROFILING_TOOLS

#if TBB_USE_PROFILING_TOOLS && !(TBB_USE_PROFILING_TOOLS == 2)
class event {
/** This class supports user event traces through itt.
    Common use-case is tagging data flow graph tasks (data-id)
    and visualization by Intel Advisor Flow Graph Analyzer (FGA)  **/
//  TODO: Replace implementation by itt user event api.

    const std::string my_name;

    static void emit_trace(const std::string &input) {
        itt_metadata_str_add( ITT_DOMAIN_FLOW, NULL, FLOW_NULL, USER_EVENT, ( "FGA::DATAID::" + input ).c_str() );
    }

public:
    event(const std::string &input)
              : my_name( input )
    { }

    void emit() {
        emit_trace(my_name);
    }

    static void emit(const std::string &description) {
        emit_trace(description);
    }

};
#else // TBB_USE_PROFILING_TOOLS && !(TBB_USE_PROFILING_TOOLS == 2)
// Using empty struct if user event tracing is disabled:
struct event {
    event(const std::string &) { }

    void emit() { }

    static void emit(const std::string &) { }
};
#endif // TBB_USE_PROFILING_TOOLS && !(TBB_USE_PROFILING_TOOLS == 2)
} // namespace d1
} // namespace detail

namespace profiling {
    using detail::d1::event;
}
} // namespace tbb


#endif /* __TBB_profiling_H */
