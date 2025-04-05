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

#include <openssl/md4.h>

#include <stdlib.h>
#include <string.h>

#include "../../internal.h"
#include "../digest/md32_common.h"


uint8_t *MD4(const uint8_t *data, size_t len, uint8_t out[MD4_DIGEST_LENGTH]) {
  MD4_CTX ctx;
  MD4_Init(&ctx);
  MD4_Update(&ctx, data, len);
  MD4_Final(out, &ctx);

  return out;
}

// Implemented from RFC 1186 The MD4 Message-Digest Algorithm.

int MD4_Init(MD4_CTX *md4) {
  OPENSSL_memset(md4, 0, sizeof(MD4_CTX));
  md4->h[0] = 0x67452301UL;
  md4->h[1] = 0xefcdab89UL;
  md4->h[2] = 0x98badcfeUL;
  md4->h[3] = 0x10325476UL;
  return 1;
}

void md4_block_data_order(uint32_t *state, const uint8_t *data, size_t num);

void MD4_Transform(MD4_CTX *c, const uint8_t data[MD4_CBLOCK]) {
  md4_block_data_order(c->h, data, 1);
}

int MD4_Update(MD4_CTX *c, const void *data, size_t len) {
  crypto_md32_update(&md4_block_data_order, c->h, c->data, MD4_CBLOCK, &c->num,
                     &c->Nh, &c->Nl, data, len);
  return 1;
}

int MD4_Final(uint8_t out[MD4_DIGEST_LENGTH], MD4_CTX *c) {
  crypto_md32_final(&md4_block_data_order, c->h, c->data, MD4_CBLOCK, &c->num,
                    c->Nh, c->Nl, /*is_big_endian=*/0);

  CRYPTO_store_u32_le(out, c->h[0]);
  CRYPTO_store_u32_le(out + 4, c->h[1]);
  CRYPTO_store_u32_le(out + 8, c->h[2]);
  CRYPTO_store_u32_le(out + 12, c->h[3]);
  return 1;
}

// As pointed out by Wei Dai <weidai@eskimo.com>, the above can be
// simplified to the code below.  Wei attributes these optimizations
// to Peter Gutmann's SHS code, and he attributes it to Rich Schroeppel.
#define F(b, c, d) ((((c) ^ (d)) & (b)) ^ (d))
#define G(b, c, d) (((b) & (c)) | ((b) & (d)) | ((c) & (d)))
#define H(b, c, d) ((b) ^ (c) ^ (d))

#define R0(a, b, c, d, k, s, t)            \
  do {                                     \
    (a) += ((k) + (t) + F((b), (c), (d))); \
    (a) = CRYPTO_rotl_u32(a, s);           \
  } while (0)

#define R1(a, b, c, d, k, s, t)            \
  do {                                     \
    (a) += ((k) + (t) + G((b), (c), (d))); \
    (a) = CRYPTO_rotl_u32(a, s);           \
  } while (0)

#define R2(a, b, c, d, k, s, t)            \
  do {                                     \
    (a) += ((k) + (t) + H((b), (c), (d))); \
    (a) = CRYPTO_rotl_u32(a, s);           \
  } while (0)

void md4_block_data_order(uint32_t *state, const uint8_t *data, size_t num) {
  uint32_t A, B, C, D;
  uint32_t X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15;

  A = state[0];
  B = state[1];
  C = state[2];
  D = state[3];

  for (; num--;) {
    X0 = CRYPTO_load_u32_le(data);
    data += 4;
    X1 = CRYPTO_load_u32_le(data);
    data += 4;
    // Round 0
    R0(A, B, C, D, X0, 3, 0);
    X2 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(D, A, B, C, X1, 7, 0);
    X3 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(C, D, A, B, X2, 11, 0);
    X4 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(B, C, D, A, X3, 19, 0);
    X5 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(A, B, C, D, X4, 3, 0);
    X6 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(D, A, B, C, X5, 7, 0);
    X7 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(C, D, A, B, X6, 11, 0);
    X8 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(B, C, D, A, X7, 19, 0);
    X9 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(A, B, C, D, X8, 3, 0);
    X10 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(D, A, B, C, X9, 7, 0);
    X11 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(C, D, A, B, X10, 11, 0);
    X12 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(B, C, D, A, X11, 19, 0);
    X13 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(A, B, C, D, X12, 3, 0);
    X14 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(D, A, B, C, X13, 7, 0);
    X15 = CRYPTO_load_u32_le(data);
    data += 4;
    R0(C, D, A, B, X14, 11, 0);
    R0(B, C, D, A, X15, 19, 0);
    // Round 1
    R1(A, B, C, D, X0, 3, 0x5A827999L);
    R1(D, A, B, C, X4, 5, 0x5A827999L);
    R1(C, D, A, B, X8, 9, 0x5A827999L);
    R1(B, C, D, A, X12, 13, 0x5A827999L);
    R1(A, B, C, D, X1, 3, 0x5A827999L);
    R1(D, A, B, C, X5, 5, 0x5A827999L);
    R1(C, D, A, B, X9, 9, 0x5A827999L);
    R1(B, C, D, A, X13, 13, 0x5A827999L);
    R1(A, B, C, D, X2, 3, 0x5A827999L);
    R1(D, A, B, C, X6, 5, 0x5A827999L);
    R1(C, D, A, B, X10, 9, 0x5A827999L);
    R1(B, C, D, A, X14, 13, 0x5A827999L);
    R1(A, B, C, D, X3, 3, 0x5A827999L);
    R1(D, A, B, C, X7, 5, 0x5A827999L);
    R1(C, D, A, B, X11, 9, 0x5A827999L);
    R1(B, C, D, A, X15, 13, 0x5A827999L);
    // Round 2
    R2(A, B, C, D, X0, 3, 0x6ED9EBA1L);
    R2(D, A, B, C, X8, 9, 0x6ED9EBA1L);
    R2(C, D, A, B, X4, 11, 0x6ED9EBA1L);
    R2(B, C, D, A, X12, 15, 0x6ED9EBA1L);
    R2(A, B, C, D, X2, 3, 0x6ED9EBA1L);
    R2(D, A, B, C, X10, 9, 0x6ED9EBA1L);
    R2(C, D, A, B, X6, 11, 0x6ED9EBA1L);
    R2(B, C, D, A, X14, 15, 0x6ED9EBA1L);
    R2(A, B, C, D, X1, 3, 0x6ED9EBA1L);
    R2(D, A, B, C, X9, 9, 0x6ED9EBA1L);
    R2(C, D, A, B, X5, 11, 0x6ED9EBA1L);
    R2(B, C, D, A, X13, 15, 0x6ED9EBA1L);
    R2(A, B, C, D, X3, 3, 0x6ED9EBA1L);
    R2(D, A, B, C, X11, 9, 0x6ED9EBA1L);
    R2(C, D, A, B, X7, 11, 0x6ED9EBA1L);
    R2(B, C, D, A, X15, 15, 0x6ED9EBA1L);

    A = state[0] += A;
    B = state[1] += B;
    C = state[2] += C;
    D = state[3] += D;
  }
}

#undef F
#undef G
#undef H
#undef R0
#undef R1
#undef R2
