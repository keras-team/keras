/* Copyright (c) 2021, Google Inc.
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

#ifndef OPENSSL_HEADER_CRYPTO_FIPSMODULE_ECDSA_INTERNAL_H
#define OPENSSL_HEADER_CRYPTO_FIPSMODULE_ECDSA_INTERNAL_H

#include <openssl/base.h>

#if defined(__cplusplus)
extern "C" {
#endif


// ecdsa_sign_with_nonce_for_known_answer_test behaves like |ECDSA_do_sign| but
// takes a fixed nonce. This function is used as part of known-answer tests in
// the FIPS module.
ECDSA_SIG *ecdsa_sign_with_nonce_for_known_answer_test(const uint8_t *digest,
                                                       size_t digest_len,
                                                       const EC_KEY *eckey,
                                                       const uint8_t *nonce,
                                                       size_t nonce_len);

// ecdsa_do_verify_no_self_test does the same as |ECDSA_do_verify|, but doesn't
// try to run the self-test first. This is for use in the self tests themselves,
// to prevent an infinite loop.
int ecdsa_do_verify_no_self_test(const uint8_t *digest, size_t digest_len,
                                 const ECDSA_SIG *sig, const EC_KEY *eckey);


#if defined(__cplusplus)
}
#endif

#endif  // OPENSSL_HEADER_CRYPTO_FIPSMODULE_ECDSA_INTERNAL_H
