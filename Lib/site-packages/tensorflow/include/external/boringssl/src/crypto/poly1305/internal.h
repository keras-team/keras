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

#ifndef OPENSSL_HEADER_POLY1305_INTERNAL_H
#define OPENSSL_HEADER_POLY1305_INTERNAL_H

#include <openssl/base.h>
#include <openssl/poly1305.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(OPENSSL_ARM) && !defined(OPENSSL_NO_ASM) && !defined(OPENSSL_APPLE)
#define OPENSSL_POLY1305_NEON

void CRYPTO_poly1305_init_neon(poly1305_state *state, const uint8_t key[32]);

void CRYPTO_poly1305_update_neon(poly1305_state *state, const uint8_t *in,
                                 size_t in_len);

void CRYPTO_poly1305_finish_neon(poly1305_state *state, uint8_t mac[16]);
#endif


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_POLY1305_INTERNAL_H
