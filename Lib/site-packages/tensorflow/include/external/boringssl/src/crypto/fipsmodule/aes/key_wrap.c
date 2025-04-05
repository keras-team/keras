/* ====================================================================
 * Copyright (c) 2001-2011 The OpenSSL Project.  All rights reserved.
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
 *    for use in the OpenSSL Toolkit. (http://www.openssl.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    openssl-core@openssl.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.openssl.org/)"
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

#include <openssl/aes.h>

#include <assert.h>
#include <limits.h>
#include <string.h>

#include <openssl/mem.h>

#include "../../internal.h"
#include "../service_indicator/internal.h"


// kDefaultIV is the default IV value given in RFC 3394, 2.2.3.1.
static const uint8_t kDefaultIV[] = {
    0xa6, 0xa6, 0xa6, 0xa6, 0xa6, 0xa6, 0xa6, 0xa6,
};

static const unsigned kBound = 6;

int AES_wrap_key(const AES_KEY *key, const uint8_t *iv, uint8_t *out,
                 const uint8_t *in, size_t in_len) {
  // See RFC 3394, section 2.2.1. Additionally, note that section 2 requires the
  // plaintext be at least two 8-byte blocks.

  if (in_len > INT_MAX - 8 || in_len < 16 || in_len % 8 != 0) {
    return -1;
  }

  if (iv == NULL) {
    iv = kDefaultIV;
  }

  OPENSSL_memmove(out + 8, in, in_len);
  uint8_t A[AES_BLOCK_SIZE];
  OPENSSL_memcpy(A, iv, 8);

  size_t n = in_len / 8;

  for (unsigned j = 0; j < kBound; j++) {
    for (size_t i = 1; i <= n; i++) {
      OPENSSL_memcpy(A + 8, out + 8 * i, 8);
      AES_encrypt(A, A, key);

      uint32_t t = (uint32_t)(n * j + i);
      A[7] ^= t & 0xff;
      A[6] ^= (t >> 8) & 0xff;
      A[5] ^= (t >> 16) & 0xff;
      A[4] ^= (t >> 24) & 0xff;
      OPENSSL_memcpy(out + 8 * i, A + 8, 8);
    }
  }

  OPENSSL_memcpy(out, A, 8);
  FIPS_service_indicator_update_state();
  return (int)in_len + 8;
}

// aes_unwrap_key_inner performs steps one and two from
// https://tools.ietf.org/html/rfc3394#section-2.2.2
static int aes_unwrap_key_inner(const AES_KEY *key, uint8_t *out,
                                uint8_t out_iv[8], const uint8_t *in,
                                size_t in_len) {
  // See RFC 3394, section 2.2.2. Additionally, note that section 2 requires the
  // plaintext be at least two 8-byte blocks, so the ciphertext must be at least
  // three blocks.

  if (in_len > INT_MAX || in_len < 24 || in_len % 8 != 0) {
    return 0;
  }

  uint8_t A[AES_BLOCK_SIZE];
  OPENSSL_memcpy(A, in, 8);
  OPENSSL_memmove(out, in + 8, in_len - 8);

  size_t n = (in_len / 8) - 1;

  for (unsigned j = kBound - 1; j < kBound; j--) {
    for (size_t i = n; i > 0; i--) {
      uint32_t t = (uint32_t)(n * j + i);
      A[7] ^= t & 0xff;
      A[6] ^= (t >> 8) & 0xff;
      A[5] ^= (t >> 16) & 0xff;
      A[4] ^= (t >> 24) & 0xff;
      OPENSSL_memcpy(A + 8, out + 8 * (i - 1), 8);
      AES_decrypt(A, A, key);
      OPENSSL_memcpy(out + 8 * (i - 1), A + 8, 8);
    }
  }

  memcpy(out_iv, A, 8);
  return 1;
}

int AES_unwrap_key(const AES_KEY *key, const uint8_t *iv, uint8_t *out,
                   const uint8_t *in, size_t in_len) {
  uint8_t calculated_iv[8];
  if (!aes_unwrap_key_inner(key, out, calculated_iv, in, in_len)) {
    return -1;
  }

  if (iv == NULL) {
    iv = kDefaultIV;
  }
  if (CRYPTO_memcmp(calculated_iv, iv, 8) != 0) {
    return -1;
  }

  FIPS_service_indicator_update_state();
  return (int)in_len - 8;
}

// kPaddingConstant is used in Key Wrap with Padding. See
// https://tools.ietf.org/html/rfc5649#section-3
static const uint8_t kPaddingConstant[4] = {0xa6, 0x59, 0x59, 0xa6};

int AES_wrap_key_padded(const AES_KEY *key, uint8_t *out, size_t *out_len,
                        size_t max_out, const uint8_t *in, size_t in_len) {
  // See https://tools.ietf.org/html/rfc5649#section-4.1
  const uint64_t in_len64 = in_len;
  const size_t padded_len = (in_len + 7) & ~7;
  *out_len = 0;
  if (in_len == 0 || in_len64 > 0xffffffffu || in_len + 7 < in_len ||
      padded_len + 8 < padded_len || max_out < padded_len + 8) {
    return 0;
  }

  uint8_t block[AES_BLOCK_SIZE];
  memcpy(block, kPaddingConstant, sizeof(kPaddingConstant));
  CRYPTO_store_u32_be(block + 4, (uint32_t)in_len);

  if (in_len <= 8) {
    memset(block + 8, 0, 8);
    memcpy(block + 8, in, in_len);
    AES_encrypt(block, out, key);
    *out_len = AES_BLOCK_SIZE;
    return 1;
  }

  uint8_t *padded_in = OPENSSL_malloc(padded_len);
  if (padded_in == NULL) {
    return 0;
  }
  assert(padded_len >= 8);
  memset(padded_in + padded_len - 8, 0, 8);
  memcpy(padded_in, in, in_len);
  FIPS_service_indicator_lock_state();
  const int ret = AES_wrap_key(key, block, out, padded_in, padded_len);
  FIPS_service_indicator_unlock_state();
  OPENSSL_free(padded_in);
  if (ret < 0) {
    return 0;
  }
  *out_len = ret;
  FIPS_service_indicator_update_state();
  return 1;
}

int AES_unwrap_key_padded(const AES_KEY *key, uint8_t *out, size_t *out_len,
                          size_t max_out, const uint8_t *in, size_t in_len) {
  *out_len = 0;
  if (in_len < AES_BLOCK_SIZE || max_out < in_len - 8) {
    return 0;
  }

  uint8_t iv[8];
  if (in_len == AES_BLOCK_SIZE) {
    uint8_t block[AES_BLOCK_SIZE];
    AES_decrypt(in, block, key);
    memcpy(iv, block, sizeof(iv));
    memcpy(out, block + 8, 8);
  } else if (!aes_unwrap_key_inner(key, out, iv, in, in_len)) {
    return 0;
  }
  assert(in_len % 8 == 0);

  crypto_word_t ok = constant_time_eq_int(
      CRYPTO_memcmp(iv, kPaddingConstant, sizeof(kPaddingConstant)), 0);

  const size_t claimed_len = CRYPTO_load_u32_be(iv + 4);
  ok &= ~constant_time_is_zero_w(claimed_len);
  ok &= constant_time_eq_w((claimed_len - 1) >> 3, (in_len - 9) >> 3);

  // Check that padding bytes are all zero.
  for (size_t i = in_len - 15; i < in_len - 8; i++) {
    ok &= constant_time_is_zero_w(constant_time_ge_8(i, claimed_len) & out[i]);
  }

  *out_len = constant_time_select_w(ok, claimed_len, 0);
  const int ret = ok & 1;
  if (ret) {
    FIPS_service_indicator_update_state();
  }
  return ret;
}
