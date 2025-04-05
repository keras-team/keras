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

#include <openssl/md5.h>

#include <string.h>

#include <openssl/mem.h>

#include "../../internal.h"
#include "../digest/md32_common.h"
#include "internal.h"


uint8_t *MD5(const uint8_t *data, size_t len, uint8_t out[MD5_DIGEST_LENGTH]) {
  MD5_CTX ctx;
  MD5_Init(&ctx);
  MD5_Update(&ctx, data, len);
  MD5_Final(out, &ctx);

  return out;
}

int MD5_Init(MD5_CTX *md5) {
  OPENSSL_memset(md5, 0, sizeof(MD5_CTX));
  md5->h[0] = 0x67452301UL;
  md5->h[1] = 0xefcdab89UL;
  md5->h[2] = 0x98badcfeUL;
  md5->h[3] = 0x10325476UL;
  return 1;
}

#if defined(MD5_ASM)
#define md5_block_data_order md5_block_asm_data_order
#else
static void md5_block_data_order(uint32_t *state, const uint8_t *data,
                                 size_t num);
#endif

void MD5_Transform(MD5_CTX *c, const uint8_t data[MD5_CBLOCK]) {
  md5_block_data_order(c->h, data, 1);
}

int MD5_Update(MD5_CTX *c, const void *data, size_t len) {
  crypto_md32_update(&md5_block_data_order, c->h, c->data, MD5_CBLOCK, &c->num,
                     &c->Nh, &c->Nl, data, len);
  return 1;
}

int MD5_Final(uint8_t out[MD5_DIGEST_LENGTH], MD5_CTX *c) {
  crypto_md32_final(&md5_block_data_order, c->h, c->data, MD5_CBLOCK, &c->num,
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
#define G(b, c, d) ((((b) ^ (c)) & (d)) ^ (c))
#define H(b, c, d) ((b) ^ (c) ^ (d))
#define I(b, c, d) (((~(d)) | (b)) ^ (c))

#define R0(a, b, c, d, k, s, t)            \
  do {                                     \
    (a) += ((k) + (t) + F((b), (c), (d))); \
    (a) = CRYPTO_rotl_u32(a, s);           \
    (a) += (b);                            \
  } while (0)

#define R1(a, b, c, d, k, s, t)            \
  do {                                     \
    (a) += ((k) + (t) + G((b), (c), (d))); \
    (a) = CRYPTO_rotl_u32(a, s);           \
    (a) += (b);                            \
  } while (0)

#define R2(a, b, c, d, k, s, t)            \
  do {                                     \
    (a) += ((k) + (t) + H((b), (c), (d))); \
    (a) = CRYPTO_rotl_u32(a, s);           \
    (a) += (b);                            \
  } while (0)

#define R3(a, b, c, d, k, s, t)            \
  do {                                     \
    (a) += ((k) + (t) + I((b), (c), (d))); \
    (a) = CRYPTO_rotl_u32(a, s);           \
    (a) += (b);                            \
  } while (0)

#ifndef MD5_ASM
#ifdef X
#undef X
#endif
static void md5_block_data_order(uint32_t *state, const uint8_t *data,
                                 size_t num) {
  uint32_t A, B, C, D;
  uint32_t XX0, XX1, XX2, XX3, XX4, XX5, XX6, XX7, XX8, XX9, XX10, XX11, XX12,
      XX13, XX14, XX15;
#define X(i) XX##i

  A = state[0];
  B = state[1];
  C = state[2];
  D = state[3];

  for (; num--;) {
    X(0) = CRYPTO_load_u32_le(data);
    data += 4;
    X(1) = CRYPTO_load_u32_le(data);
    data += 4;
    // Round 0
    R0(A, B, C, D, X(0), 7, 0xd76aa478L);
    X(2) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(D, A, B, C, X(1), 12, 0xe8c7b756L);
    X(3) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(C, D, A, B, X(2), 17, 0x242070dbL);
    X(4) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(B, C, D, A, X(3), 22, 0xc1bdceeeL);
    X(5) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(A, B, C, D, X(4), 7, 0xf57c0fafL);
    X(6) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(D, A, B, C, X(5), 12, 0x4787c62aL);
    X(7) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(C, D, A, B, X(6), 17, 0xa8304613L);
    X(8) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(B, C, D, A, X(7), 22, 0xfd469501L);
    X(9) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(A, B, C, D, X(8), 7, 0x698098d8L);
    X(10) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(D, A, B, C, X(9), 12, 0x8b44f7afL);
    X(11) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(C, D, A, B, X(10), 17, 0xffff5bb1L);
    X(12) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(B, C, D, A, X(11), 22, 0x895cd7beL);
    X(13) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(A, B, C, D, X(12), 7, 0x6b901122L);
    X(14) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(D, A, B, C, X(13), 12, 0xfd987193L);
    X(15) = CRYPTO_load_u32_le(data);
    data += 4;
    R0(C, D, A, B, X(14), 17, 0xa679438eL);
    R0(B, C, D, A, X(15), 22, 0x49b40821L);
    // Round 1
    R1(A, B, C, D, X(1), 5, 0xf61e2562L);
    R1(D, A, B, C, X(6), 9, 0xc040b340L);
    R1(C, D, A, B, X(11), 14, 0x265e5a51L);
    R1(B, C, D, A, X(0), 20, 0xe9b6c7aaL);
    R1(A, B, C, D, X(5), 5, 0xd62f105dL);
    R1(D, A, B, C, X(10), 9, 0x02441453L);
    R1(C, D, A, B, X(15), 14, 0xd8a1e681L);
    R1(B, C, D, A, X(4), 20, 0xe7d3fbc8L);
    R1(A, B, C, D, X(9), 5, 0x21e1cde6L);
    R1(D, A, B, C, X(14), 9, 0xc33707d6L);
    R1(C, D, A, B, X(3), 14, 0xf4d50d87L);
    R1(B, C, D, A, X(8), 20, 0x455a14edL);
    R1(A, B, C, D, X(13), 5, 0xa9e3e905L);
    R1(D, A, B, C, X(2), 9, 0xfcefa3f8L);
    R1(C, D, A, B, X(7), 14, 0x676f02d9L);
    R1(B, C, D, A, X(12), 20, 0x8d2a4c8aL);
    // Round 2
    R2(A, B, C, D, X(5), 4, 0xfffa3942L);
    R2(D, A, B, C, X(8), 11, 0x8771f681L);
    R2(C, D, A, B, X(11), 16, 0x6d9d6122L);
    R2(B, C, D, A, X(14), 23, 0xfde5380cL);
    R2(A, B, C, D, X(1), 4, 0xa4beea44L);
    R2(D, A, B, C, X(4), 11, 0x4bdecfa9L);
    R2(C, D, A, B, X(7), 16, 0xf6bb4b60L);
    R2(B, C, D, A, X(10), 23, 0xbebfbc70L);
    R2(A, B, C, D, X(13), 4, 0x289b7ec6L);
    R2(D, A, B, C, X(0), 11, 0xeaa127faL);
    R2(C, D, A, B, X(3), 16, 0xd4ef3085L);
    R2(B, C, D, A, X(6), 23, 0x04881d05L);
    R2(A, B, C, D, X(9), 4, 0xd9d4d039L);
    R2(D, A, B, C, X(12), 11, 0xe6db99e5L);
    R2(C, D, A, B, X(15), 16, 0x1fa27cf8L);
    R2(B, C, D, A, X(2), 23, 0xc4ac5665L);
    // Round 3
    R3(A, B, C, D, X(0), 6, 0xf4292244L);
    R3(D, A, B, C, X(7), 10, 0x432aff97L);
    R3(C, D, A, B, X(14), 15, 0xab9423a7L);
    R3(B, C, D, A, X(5), 21, 0xfc93a039L);
    R3(A, B, C, D, X(12), 6, 0x655b59c3L);
    R3(D, A, B, C, X(3), 10, 0x8f0ccc92L);
    R3(C, D, A, B, X(10), 15, 0xffeff47dL);
    R3(B, C, D, A, X(1), 21, 0x85845dd1L);
    R3(A, B, C, D, X(8), 6, 0x6fa87e4fL);
    R3(D, A, B, C, X(15), 10, 0xfe2ce6e0L);
    R3(C, D, A, B, X(6), 15, 0xa3014314L);
    R3(B, C, D, A, X(13), 21, 0x4e0811a1L);
    R3(A, B, C, D, X(4), 6, 0xf7537e82L);
    R3(D, A, B, C, X(11), 10, 0xbd3af235L);
    R3(C, D, A, B, X(2), 15, 0x2ad7d2bbL);
    R3(B, C, D, A, X(9), 21, 0xeb86d391L);

    A = state[0] += A;
    B = state[1] += B;
    C = state[2] += C;
    D = state[3] += D;
  }
}
#undef X
#endif

#undef F
#undef G
#undef H
#undef I
#undef R0
#undef R1
#undef R2
#undef R3
