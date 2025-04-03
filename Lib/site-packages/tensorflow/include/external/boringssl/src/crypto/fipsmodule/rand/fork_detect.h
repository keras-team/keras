/* Copyright (c) 2020, Google Inc.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
 * OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#ifndef OPENSSL_HEADER_CRYPTO_FORK_DETECT_H
#define OPENSSL_HEADER_CRYPTO_FORK_DETECT_H

#include <openssl/base.h>

#if defined(__cplusplus)
extern "C" {
#endif


// crypto_get_fork_generation returns the fork generation number for the current
// process, or zero if not supported on the platform. The fork generation number
// is a non-zero, strictly-monotonic counter with the property that, if queried
// in an address space and then again in a subsequently forked copy, the forked
// address space will observe a greater value.
//
// This function may be used to clear cached values across a fork. When
// initializing a cache, record the fork generation. Before using the cache,
// check if the fork generation has changed. If so, drop the cache and update
// the save fork generation. Note this logic transparently handles platforms
// which always return zero.
//
// This is not reliably supported on all platforms which implement |fork|, so it
// should only be used as a hardening measure.
OPENSSL_EXPORT uint64_t CRYPTO_get_fork_generation(void);

// CRYPTO_fork_detect_ignore_madv_wipeonfork_for_testing is an internal detail
// used for testing purposes.
OPENSSL_EXPORT void CRYPTO_fork_detect_ignore_madv_wipeonfork_for_testing(void);

#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_FORK_DETECT_H
