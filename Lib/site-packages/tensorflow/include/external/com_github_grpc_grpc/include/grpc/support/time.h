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

#ifndef GRPC_SUPPORT_TIME_H
#define GRPC_SUPPORT_TIME_H

#include <grpc/support/port_platform.h>

#include <grpc/impl/codegen/gpr_types.h>

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Time constants. */
GPRAPI gpr_timespec
gpr_time_0(gpr_clock_type type); /** The zero time interval. */
GPRAPI gpr_timespec gpr_inf_future(gpr_clock_type type); /** The far future */
GPRAPI gpr_timespec gpr_inf_past(gpr_clock_type type);   /** The far past. */

#define GPR_MS_PER_SEC 1000
#define GPR_US_PER_SEC 1000000
#define GPR_NS_PER_SEC 1000000000
#define GPR_NS_PER_MS 1000000
#define GPR_NS_PER_US 1000
#define GPR_US_PER_MS 1000

/** initialize time subsystem */
GPRAPI void gpr_time_init(void);

/** Return the current time measured from the given clocks epoch. */
GPRAPI gpr_timespec gpr_now(gpr_clock_type clock);

/** Convert a timespec from one clock to another */
GPRAPI gpr_timespec gpr_convert_clock_type(gpr_timespec t,
                                           gpr_clock_type target_clock);

/** Return -ve, 0, or +ve according to whether a < b, a == b, or a > b
   respectively.  */
GPRAPI int gpr_time_cmp(gpr_timespec a, gpr_timespec b);

GPRAPI gpr_timespec gpr_time_max(gpr_timespec a, gpr_timespec b);
GPRAPI gpr_timespec gpr_time_min(gpr_timespec a, gpr_timespec b);

/** Add and subtract times.  Calculations saturate at infinities. */
GPRAPI gpr_timespec gpr_time_add(gpr_timespec a, gpr_timespec b);
GPRAPI gpr_timespec gpr_time_sub(gpr_timespec a, gpr_timespec b);

/** Return a timespec representing a given number of time units. INT64_MIN is
   interpreted as gpr_inf_past, and INT64_MAX as gpr_inf_future.  */
GPRAPI gpr_timespec gpr_time_from_micros(int64_t x, gpr_clock_type clock_type);
GPRAPI gpr_timespec gpr_time_from_nanos(int64_t x, gpr_clock_type clock_type);
GPRAPI gpr_timespec gpr_time_from_millis(int64_t x, gpr_clock_type clock_type);
GPRAPI gpr_timespec gpr_time_from_seconds(int64_t x, gpr_clock_type clock_type);
GPRAPI gpr_timespec gpr_time_from_minutes(int64_t x, gpr_clock_type clock_type);
GPRAPI gpr_timespec gpr_time_from_hours(int64_t x, gpr_clock_type clock_type);

GPRAPI int32_t gpr_time_to_millis(gpr_timespec timespec);

/** Return 1 if two times are equal or within threshold of each other,
   0 otherwise */
GPRAPI int gpr_time_similar(gpr_timespec a, gpr_timespec b,
                            gpr_timespec threshold);

/** Sleep until at least 'until' - an absolute timeout */
GPRAPI void gpr_sleep_until(gpr_timespec until);

GPRAPI double gpr_timespec_to_micros(gpr_timespec t);

#ifdef __cplusplus
}
#endif

#endif /* GRPC_SUPPORT_TIME_H */
