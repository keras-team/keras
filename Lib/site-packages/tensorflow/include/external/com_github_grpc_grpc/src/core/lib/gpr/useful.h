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

#ifndef GRPC_CORE_LIB_GPR_USEFUL_H
#define GRPC_CORE_LIB_GPR_USEFUL_H

/** useful macros that don't belong anywhere else */

#define GPR_MIN(a, b) ((a) < (b) ? (a) : (b))
#define GPR_MAX(a, b) ((a) > (b) ? (a) : (b))
#define GPR_CLAMP(a, min, max) ((a) < (min) ? (min) : (a) > (max) ? (max) : (a))
/** rotl, rotr assume x is unsigned */
#define GPR_ROTL(x, n) (((x) << (n)) | ((x) >> (sizeof(x) * 8 - (n))))
#define GPR_ROTR(x, n) (((x) >> (n)) | ((x) << (sizeof(x) * 8 - (n))))

#define GPR_ARRAY_SIZE(array) (sizeof(array) / sizeof(*(array)))

#define GPR_SWAP(type, a, b) \
  do {                       \
    type x = a;              \
    a = b;                   \
    b = x;                   \
  } while (0)

/** Set the \a n-th bit of \a i (a mutable pointer). */
#define GPR_BITSET(i, n) ((*(i)) |= (1u << (n)))

/** Clear the \a n-th bit of \a i (a mutable pointer). */
#define GPR_BITCLEAR(i, n) ((*(i)) &= ~(1u << (n)))

/** Get the \a n-th bit of \a i */
#define GPR_BITGET(i, n) (((i) & (1u << (n))) != 0)

#define GPR_INTERNAL_HEXDIGIT_BITCOUNT(x)                        \
  ((x) - (((x) >> 1) & 0x77777777) - (((x) >> 2) & 0x33333333) - \
   (((x) >> 3) & 0x11111111))

/** Returns number of bits set in bitset \a i */
#define GPR_BITCOUNT(i)                          \
  (((GPR_INTERNAL_HEXDIGIT_BITCOUNT(i) +         \
     (GPR_INTERNAL_HEXDIGIT_BITCOUNT(i) >> 4)) & \
    0x0f0f0f0f) %                                \
   255)

#define GPR_ICMP(a, b) ((a) < (b) ? -1 : ((a) > (b) ? 1 : 0))

#define GPR_HASH_POINTER(x, range) \
  ((((size_t)x) >> 4) ^ (((size_t)x) >> 9) ^ (((size_t)x) >> 14)) % (range)

#endif /* GRPC_CORE_LIB_GPR_USEFUL_H */
