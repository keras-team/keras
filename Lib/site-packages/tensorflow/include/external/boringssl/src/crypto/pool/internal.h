/* Copyright (c) 2016, Google Inc.
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

#ifndef OPENSSL_HEADER_POOL_INTERNAL_H
#define OPENSSL_HEADER_POOL_INTERNAL_H

#include <openssl/lhash.h>
#include <openssl/thread.h>

#include "../lhash/internal.h"


#if defined(__cplusplus)
extern "C" {
#endif


DEFINE_LHASH_OF(CRYPTO_BUFFER)

struct crypto_buffer_st {
  CRYPTO_BUFFER_POOL *pool;
  uint8_t *data;
  size_t len;
  CRYPTO_refcount_t references;
  int data_is_static;
};

struct crypto_buffer_pool_st {
  LHASH_OF(CRYPTO_BUFFER) *bufs;
  CRYPTO_MUTEX lock;
  const uint64_t hash_key[2];
};


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_POOL_INTERNAL_H
