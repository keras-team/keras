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

#include <openssl/base.h>

#include <assert.h>
#include <string.h>

#include "internal.h"
#include "../../internal.h"


// byte_reverse reverses the order of the bytes in |b->c|.
static void byte_reverse(polyval_block *b) {
  const uint64_t t = CRYPTO_bswap8(b->u[0]);
  b->u[0] = CRYPTO_bswap8(b->u[1]);
  b->u[1] = t;
}

// reverse_and_mulX_ghash interprets the bytes |b->c| as a reversed element of
// the GHASH field, multiplies that by 'x' and serialises the result back into
// |b|, but with GHASH's backwards bit ordering.
static void reverse_and_mulX_ghash(polyval_block *b) {
  uint64_t hi = b->u[0];
  uint64_t lo = b->u[1];
  const crypto_word_t carry = constant_time_eq_w(hi & 1, 1);
  hi >>= 1;
  hi |= lo << 63;
  lo >>= 1;
  lo ^= ((uint64_t) constant_time_select_w(carry, 0xe1, 0)) << 56;

  b->u[0] = CRYPTO_bswap8(lo);
  b->u[1] = CRYPTO_bswap8(hi);
}

// POLYVAL(H, X_1, ..., X_n) =
// ByteReverse(GHASH(mulX_GHASH(ByteReverse(H)), ByteReverse(X_1), ...,
// ByteReverse(X_n))).
//
// See https://tools.ietf.org/html/draft-irtf-cfrg-gcmsiv-02#appendix-A.

void CRYPTO_POLYVAL_init(struct polyval_ctx *ctx, const uint8_t key[16]) {
  polyval_block H;
  OPENSSL_memcpy(H.c, key, 16);
  reverse_and_mulX_ghash(&H);

  int is_avx;
  CRYPTO_ghash_init(&ctx->gmult, &ctx->ghash, &ctx->H, ctx->Htable, &is_avx,
                    H.c);
  OPENSSL_memset(&ctx->S, 0, sizeof(ctx->S));
}

void CRYPTO_POLYVAL_update_blocks(struct polyval_ctx *ctx, const uint8_t *in,
                                  size_t in_len) {
  assert((in_len & 15) == 0);
  polyval_block reversed[32];

  while (in_len > 0) {
    size_t todo = in_len;
    if (todo > sizeof(reversed)) {
      todo = sizeof(reversed);
    }
    OPENSSL_memcpy(reversed, in, todo);
    in += todo;
    in_len -= todo;

    size_t blocks = todo / sizeof(polyval_block);
    for (size_t i = 0; i < blocks; i++) {
      byte_reverse(&reversed[i]);
    }

    ctx->ghash(ctx->S.u, ctx->Htable, (const uint8_t *) reversed, todo);
  }
}

void CRYPTO_POLYVAL_finish(const struct polyval_ctx *ctx, uint8_t out[16]) {
  polyval_block S = ctx->S;
  byte_reverse(&S);
  OPENSSL_memcpy(out, &S.c, sizeof(polyval_block));
}
