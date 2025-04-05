/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef GRPC_CORE_LIB_PROFILING_TIMERS_H
#define GRPC_CORE_LIB_PROFILING_TIMERS_H

void gpr_timers_global_init(void);
void gpr_timers_global_destroy(void);

void gpr_timer_add_mark(const char* tagstr, int important, const char* file,
                        int line);
void gpr_timer_begin(const char* tagstr, int important, const char* file,
                     int line);
void gpr_timer_end(const char* tagstr, int important, const char* file,
                   int line);

void gpr_timers_set_log_filename(const char* filename);

void gpr_timer_set_enabled(int enabled);

#if !(defined(GRPC_STAP_PROFILER) + defined(GRPC_BASIC_PROFILER) + \
      defined(GRPC_CUSTOM_PROFILER))
/* No profiling. No-op all the things. */
#define GPR_TIMER_MARK(tag, important) \
  do {                                 \
  } while (0)

#define GPR_TIMER_SCOPE(tag, important) \
  do {                                  \
  } while (0)

#else /* at least one profiler requested... */
/* ... hopefully only one. */
#if defined(GRPC_STAP_PROFILER) && defined(GRPC_BASIC_PROFILER)
#error "GRPC_STAP_PROFILER and GRPC_BASIC_PROFILER are mutually exclusive."
#endif
#if defined(GRPC_STAP_PROFILER) && defined(GRPC_CUSTOM_PROFILER)
#error "GRPC_STAP_PROFILER and GRPC_CUSTOM_PROFILER are mutually exclusive."
#endif
#if defined(GRPC_CUSTOM_PROFILER) && defined(GRPC_BASIC_PROFILER)
#error "GRPC_CUSTOM_PROFILER and GRPC_BASIC_PROFILER are mutually exclusive."
#endif

/* Generic profiling interface. */
#define GPR_TIMER_MARK(tag, important) \
  gpr_timer_add_mark(tag, important, __FILE__, __LINE__);

#ifdef GRPC_STAP_PROFILER
/* Empty placeholder for now. */
#endif /* GRPC_STAP_PROFILER */

#ifdef GRPC_BASIC_PROFILER
/* Empty placeholder for now. */
#endif /* GRPC_BASIC_PROFILER */

namespace grpc {
class ProfileScope {
 public:
  ProfileScope(const char* desc, bool important, const char* file, int line)
      : desc_(desc) {
    gpr_timer_begin(desc_, important ? 1 : 0, file, line);
  }
  ~ProfileScope() { gpr_timer_end(desc_, 0, "n/a", 0); }

 private:
  const char* const desc_;
};
}  // namespace grpc

#define GPR_TIMER_SCOPE_NAME_INTERNAL(prefix, line) prefix##line
#define GPR_TIMER_SCOPE_NAME(prefix, line) \
  GPR_TIMER_SCOPE_NAME_INTERNAL(prefix, line)
#define GPR_TIMER_SCOPE(tag, important)                                 \
  ::grpc::ProfileScope GPR_TIMER_SCOPE_NAME(_profile_scope_, __LINE__)( \
      (tag), (important), __FILE__, __LINE__)

#endif /* at least one profiler requested. */

#endif /* GRPC_CORE_LIB_PROFILING_TIMERS_H */
