/* ====================================================================
 * Copyright (c) 1999-2007 The OpenSSL Project.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. All advertising materials mentioning features or use of this
 *    software must display the following acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit. (http://www.OpenSSL.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    licensing@OpenSSL.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.OpenSSL.org/)"
 *
 * THIS SOFTWARE IS PROVIDED BY THE OpenSSL PROJECT ``AS IS'' AND ANY
 * EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE OpenSSL PROJECT OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * ==================================================================== */

#ifndef OPENSSL_HEADER_DIGEST_MD32_COMMON_H
#define OPENSSL_HEADER_DIGEST_MD32_COMMON_H

#include <openssl/base.h>

#include <assert.h>

#include "../../internal.h"

#if defined(__cplusplus)
extern "C" {
#endif


// This is a generic 32-bit "collector" for message digest algorithms. It
// collects input character stream into chunks of 32-bit values and invokes the
// block function that performs the actual hash calculations.
//
// To make use of this mechanism, the hash context should be defined with the
// following parameters.
//
//     typedef struct <name>_state_st {
//       uint32_t h[<chaining length> / sizeof(uint32_t)];
//       uint32_t Nl, Nh;
//       uint8_t data[<block size>];
//       unsigned num;
//       ...
//     } <NAME>_CTX;
//
// <chaining length> is the output length of the hash in bytes, before
// any truncation (e.g. 64 for SHA-224 and SHA-256, 128 for SHA-384 and
// SHA-512).
//
// |h| is the hash state and is updated by a function of type
// |crypto_md32_block_func|. |data| is the partial unprocessed block and has
// |num| bytes. |Nl| and |Nh| maintain the number of bits processed so far.

// A crypto_md32_block_func should incorporate |num_blocks| of input from |data|
// into |state|. It is assumed the caller has sized |state| and |data| for the
// hash function.
typedef void (*crypto_md32_block_func)(uint32_t *state, const uint8_t *data,
                                       size_t num_blocks);

// crypto_md32_update adds |len| bytes from |in| to the digest. |data| must be a
// buffer of length |block_size| with the first |*num| bytes containing a
// partial block. This function combines the partial block with |in| and
// incorporates any complete blocks into the digest state |h|. It then updates
// |data| and |*num| with the new partial block and updates |*Nh| and |*Nl| with
// the data consumed.
static inline void crypto_md32_update(crypto_md32_block_func block_func,
                                      uint32_t *h, uint8_t *data,
                                      size_t block_size, unsigned *num,
                                      uint32_t *Nh, uint32_t *Nl,
                                      const uint8_t *in, size_t len) {
  if (len == 0) {
    return;
  }

  uint32_t l = *Nl + (((uint32_t)len) << 3);
  if (l < *Nl) {
    // Handle carries.
    (*Nh)++;
  }
  *Nh += (uint32_t)(len >> 29);
  *Nl = l;

  size_t n = *num;
  if (n != 0) {
    if (len >= block_size || len + n >= block_size) {
      OPENSSL_memcpy(data + n, in, block_size - n);
      block_func(h, data, 1);
      n = block_size - n;
      in += n;
      len -= n;
      *num = 0;
      // Keep |data| zeroed when unused.
      OPENSSL_memset(data, 0, block_size);
    } else {
      OPENSSL_memcpy(data + n, in, len);
      *num += (unsigned)len;
      return;
    }
  }

  n = len / block_size;
  if (n > 0) {
    block_func(h, in, n);
    n *= block_size;
    in += n;
    len -= n;
  }

  if (len != 0) {
    *num = (unsigned)len;
    OPENSSL_memcpy(data, in, len);
  }
}

// crypto_md32_final incorporates the partial block and trailing length into the
// digest state |h|. The trailing length is encoded in little-endian if
// |is_big_endian| is zero and big-endian otherwise. |data| must be a buffer of
// length |block_size| with the first |*num| bytes containing a partial block.
// |Nh| and |Nl| contain the total number of bits processed. On return, this
// function clears the partial block in |data| and
// |*num|.
//
// This function does not serialize |h| into a final digest. This is the
// responsibility of the caller.
static inline void crypto_md32_final(crypto_md32_block_func block_func,
                                     uint32_t *h, uint8_t *data,
                                     size_t block_size, unsigned *num,
                                     uint32_t Nh, uint32_t Nl,
                                     int is_big_endian) {
  // |data| always has room for at least one byte. A full block would have
  // been consumed.
  size_t n = *num;
  assert(n < block_size);
  data[n] = 0x80;
  n++;

  // Fill the block with zeros if there isn't room for a 64-bit length.
  if (n > block_size - 8) {
    OPENSSL_memset(data + n, 0, block_size - n);
    n = 0;
    block_func(h, data, 1);
  }
  OPENSSL_memset(data + n, 0, block_size - 8 - n);

  // Append a 64-bit length to the block and process it.
  if (is_big_endian) {
    CRYPTO_store_u32_be(data + block_size - 8, Nh);
    CRYPTO_store_u32_be(data + block_size - 4, Nl);
  } else {
    CRYPTO_store_u32_le(data + block_size - 8, Nl);
    CRYPTO_store_u32_le(data + block_size - 4, Nh);
  }
  block_func(h, data, 1);
  *num = 0;
  OPENSSL_memset(data, 0, block_size);
}


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_DIGEST_MD32_COMMON_H
