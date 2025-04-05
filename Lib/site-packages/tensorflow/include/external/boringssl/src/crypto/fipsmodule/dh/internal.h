/* Copyright (c) 2022, Google Inc.
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

#ifndef OPENSSL_HEADER_CRYPTO_FIPSMODULE_DH_INTERNAL_H
#define OPENSSL_HEADER_CRYPTO_FIPSMODULE_DH_INTERNAL_H

#include <openssl/base.h>

#include <openssl/thread.h>

#if defined(__cplusplus)
extern "C" {
#endif


struct dh_st {
  BIGNUM *p;
  BIGNUM *g;
  BIGNUM *q;
  BIGNUM *pub_key;   // g^x mod p
  BIGNUM *priv_key;  // x

  // priv_length contains the length, in bits, of the private value. If zero,
  // the private value will be the same length as |p|.
  unsigned priv_length;

  CRYPTO_MUTEX method_mont_p_lock;
  BN_MONT_CTX *method_mont_p;

  int flags;
  CRYPTO_refcount_t references;
};

// dh_compute_key_padded_no_self_test does the same as |DH_compute_key_padded|,
// but doesn't try to run the self-test first. This is for use in the self tests
// themselves, to prevent an infinite loop.
int dh_compute_key_padded_no_self_test(unsigned char *out,
                                       const BIGNUM *peers_key, DH *dh);


#if defined(__cplusplus)
}
#endif

#endif  // OPENSSL_HEADER_CRYPTO_FIPSMODULE_DH_INTERNAL_H
