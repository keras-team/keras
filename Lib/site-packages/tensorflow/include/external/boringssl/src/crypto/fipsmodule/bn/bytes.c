/* Copyright (C) 1995-1998 Eric Young (eay@cryptsoft.com)
 * All rights reserved.
 *
 * This package is an SSL implementation written
 * by Eric Young (eay@cryptsoft.com).
 * The implementation was written so as to conform with Netscapes SSL.
 *
 * This library is free for commercial and non-commercial use as long as
 * the following conditions are aheared to.  The following conditions
 * apply to all code found in this distribution, be it the RC4, RSA,
 * lhash, DES, etc., code; not just the SSL code.  The SSL documentation
 * included with this distribution is covered by the same copyright terms
 * except that the holder is Tim Hudson (tjh@cryptsoft.com).
 *
 * Copyright remains Eric Young's, and as such any Copyright notices in
 * the code are not to be removed.
 * If this package is used in a product, Eric Young should be given attribution
 * as the author of the parts of the library used.
 * This can be in the form of a textual message at program startup or
 * in documentation (online or textual) provided with the package.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    "This product includes cryptographic software written by
 *     Eric Young (eay@cryptsoft.com)"
 *    The word 'cryptographic' can be left out if the rouines from the library
 *    being used are not cryptographic related :-).
 * 4. If you include any Windows specific code (or a derivative thereof) from
 *    the apps directory (application code) you must include an acknowledgement:
 *    "This product includes software written by Tim Hudson (tjh@cryptsoft.com)"
 *
 * THIS SOFTWARE IS PROVIDED BY ERIC YOUNG ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * The licence and distribution terms for any publically available version or
 * derivative of this code cannot be changed.  i.e. this code cannot simply be
 * copied and put under another distribution licence
 * [including the GNU Public Licence.] */

#include <openssl/bn.h>

#include <assert.h>
#include <limits.h>

#include "internal.h"

void bn_big_endian_to_words(BN_ULONG *out, size_t out_len, const uint8_t *in,
                            size_t in_len) {
  for (size_t i = 0; i < out_len; i++) {
    if (in_len < sizeof(BN_ULONG)) {
      // Load the last partial word.
      BN_ULONG word = 0;
      for (size_t j = 0; j < in_len; j++) {
        word = (word << 8) | in[j];
      }
      in_len = 0;
      out[i] = word;
      // Fill the remainder with zeros.
      OPENSSL_memset(out + i + 1, 0, (out_len - i - 1) * sizeof(BN_ULONG));
      break;
    }

    in_len -= sizeof(BN_ULONG);
    out[i] = CRYPTO_load_word_be(in + in_len);
  }

  // The caller should have sized the output to avoid truncation.
  assert(in_len == 0);
}

BIGNUM *BN_bin2bn(const uint8_t *in, size_t len, BIGNUM *ret) {
  BIGNUM *bn = NULL;
  if (ret == NULL) {
    bn = BN_new();
    if (bn == NULL) {
      return NULL;
    }
    ret = bn;
  }

  if (len == 0) {
    ret->width = 0;
    return ret;
  }

  size_t num_words = ((len - 1) / BN_BYTES) + 1;
  if (!bn_wexpand(ret, num_words)) {
    BN_free(bn);
    return NULL;
  }

  // |bn_wexpand| must check bounds on |num_words| to write it into
  // |ret->dmax|.
  assert(num_words <= INT_MAX);
  ret->width = (int)num_words;
  ret->neg = 0;

  bn_big_endian_to_words(ret->d, ret->width, in, len);
  return ret;
}

BIGNUM *BN_le2bn(const uint8_t *in, size_t len, BIGNUM *ret) {
  BIGNUM *bn = NULL;
  if (ret == NULL) {
    bn = BN_new();
    if (bn == NULL) {
      return NULL;
    }
    ret = bn;
  }

  if (len == 0) {
    ret->width = 0;
    ret->neg = 0;
    return ret;
  }

  // Reserve enough space in |ret|.
  size_t num_words = ((len - 1) / BN_BYTES) + 1;
  if (!bn_wexpand(ret, num_words)) {
    BN_free(bn);
    return NULL;
  }
  ret->width = (int)num_words;

  // Make sure the top bytes will be zeroed.
  ret->d[num_words - 1] = 0;

  // We only support little-endian platforms, so we can simply memcpy the
  // internal representation.
  OPENSSL_memcpy(ret->d, in, len);
  return ret;
}

// fits_in_bytes returns one if the |num_words| words in |words| can be
// represented in |num_bytes| bytes.
static int fits_in_bytes(const BN_ULONG *words, size_t num_words,
                         size_t num_bytes) {
  const uint8_t *bytes = (const uint8_t *)words;
  size_t tot_bytes = num_words * sizeof(BN_ULONG);
  uint8_t mask = 0;
  for (size_t i = num_bytes; i < tot_bytes; i++) {
    mask |= bytes[i];
  }
  return mask == 0;
}

void bn_assert_fits_in_bytes(const BIGNUM *bn, size_t num) {
  const uint8_t *bytes = (const uint8_t *)bn->d;
  size_t tot_bytes = bn->width * sizeof(BN_ULONG);
  if (tot_bytes > num) {
    CONSTTIME_DECLASSIFY(bytes + num, tot_bytes - num);
    for (size_t i = num; i < tot_bytes; i++) {
      assert(bytes[i] == 0);
    }
    (void)bytes;
  }
}

void bn_words_to_big_endian(uint8_t *out, size_t out_len, const BN_ULONG *in,
                            size_t in_len) {
  // The caller should have selected an output length without truncation.
  assert(fits_in_bytes(in, in_len, out_len));

  // We only support little-endian platforms, so the internal representation is
  // also little-endian as bytes. We can simply copy it in reverse.
  const uint8_t *bytes = (const uint8_t *)in;
  size_t num_bytes = in_len * sizeof(BN_ULONG);
  if (out_len < num_bytes) {
    num_bytes = out_len;
  }

  for (size_t i = 0; i < num_bytes; i++) {
    out[out_len - i - 1] = bytes[i];
  }
  // Pad out the rest of the buffer with zeroes.
  OPENSSL_memset(out, 0, out_len - num_bytes);
}

size_t BN_bn2bin(const BIGNUM *in, uint8_t *out) {
  size_t n = BN_num_bytes(in);
  bn_words_to_big_endian(out, n, in->d, in->width);
  return n;
}

int BN_bn2le_padded(uint8_t *out, size_t len, const BIGNUM *in) {
  if (!fits_in_bytes(in->d, in->width, len)) {
    return 0;
  }

  // We only support little-endian platforms, so we can simply memcpy into the
  // internal representation.
  const uint8_t *bytes = (const uint8_t *)in->d;
  size_t num_bytes = in->width * BN_BYTES;
  if (len < num_bytes) {
    num_bytes = len;
  }

  OPENSSL_memcpy(out, bytes, num_bytes);
  // Pad out the rest of the buffer with zeroes.
  OPENSSL_memset(out + num_bytes, 0, len - num_bytes);
  return 1;
}

int BN_bn2bin_padded(uint8_t *out, size_t len, const BIGNUM *in) {
  if (!fits_in_bytes(in->d, in->width, len)) {
    return 0;
  }

  bn_words_to_big_endian(out, len, in->d, in->width);
  return 1;
}

BN_ULONG BN_get_word(const BIGNUM *bn) {
  switch (bn_minimal_width(bn)) {
    case 0:
      return 0;
    case 1:
      return bn->d[0];
    default:
      return BN_MASK2;
  }
}

int BN_get_u64(const BIGNUM *bn, uint64_t *out) {
  switch (bn_minimal_width(bn)) {
    case 0:
      *out = 0;
      return 1;
    case 1:
      *out = bn->d[0];
      return 1;
#if defined(OPENSSL_32_BIT)
    case 2:
      *out = (uint64_t) bn->d[0] | (((uint64_t) bn->d[1]) << 32);
      return 1;
#endif
    default:
      return 0;
  }
}
