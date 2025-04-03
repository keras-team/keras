/* Copyright (c) 2018, Google Inc.
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

#ifndef OPENSSL_HEADER_CRYPTO_FIPSMODULE_TLS_INTERNAL_H
#define OPENSSL_HEADER_CRYPTO_FIPSMODULE_TLS_INTERNAL_H

#include <openssl/base.h>

#if defined(__cplusplus)
extern "C" {
#endif


// tls1_prf calculates |out_len| bytes of the TLS PDF, using |digest|, and
// writes them to |out|. It returns one on success and zero on error.
OPENSSL_EXPORT int CRYPTO_tls1_prf(const EVP_MD *digest,
                                   uint8_t *out, size_t out_len,
                                   const uint8_t *secret, size_t secret_len,
                                   const char *label, size_t label_len,
                                   const uint8_t *seed1, size_t seed1_len,
                                   const uint8_t *seed2, size_t seed2_len);


#if defined(__cplusplus)
}
#endif

#endif  // OPENSSL_HEADER_CRYPTO_FIPSMODULE_TLS_INTERNAL_H
