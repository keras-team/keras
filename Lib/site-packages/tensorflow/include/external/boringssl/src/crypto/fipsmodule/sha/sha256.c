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

#include <openssl/sha.h>

#include <string.h>

#include <openssl/mem.h>

#include "../../internal.h"
#include "../digest/md32_common.h"
#include "../service_indicator/internal.h"
#include "internal.h"


int SHA224_Init(SHA256_CTX *sha) {
  OPENSSL_memset(sha, 0, sizeof(SHA256_CTX));
  sha->h[0] = 0xc1059ed8UL;
  sha->h[1] = 0x367cd507UL;
  sha->h[2] = 0x3070dd17UL;
  sha->h[3] = 0xf70e5939UL;
  sha->h[4] = 0xffc00b31UL;
  sha->h[5] = 0x68581511UL;
  sha->h[6] = 0x64f98fa7UL;
  sha->h[7] = 0xbefa4fa4UL;
  sha->md_len = SHA224_DIGEST_LENGTH;
  return 1;
}

int SHA256_Init(SHA256_CTX *sha) {
  OPENSSL_memset(sha, 0, sizeof(SHA256_CTX));
  sha->h[0] = 0x6a09e667UL;
  sha->h[1] = 0xbb67ae85UL;
  sha->h[2] = 0x3c6ef372UL;
  sha->h[3] = 0xa54ff53aUL;
  sha->h[4] = 0x510e527fUL;
  sha->h[5] = 0x9b05688cUL;
  sha->h[6] = 0x1f83d9abUL;
  sha->h[7] = 0x5be0cd19UL;
  sha->md_len = SHA256_DIGEST_LENGTH;
  return 1;
}

uint8_t *SHA224(const uint8_t *data, size_t len,
                uint8_t out[SHA224_DIGEST_LENGTH]) {
  SHA256_CTX ctx;
  SHA224_Init(&ctx);
  SHA224_Update(&ctx, data, len);
  SHA224_Final(out, &ctx);
  OPENSSL_cleanse(&ctx, sizeof(ctx));
  return out;
}

uint8_t *SHA256(const uint8_t *data, size_t len,
                uint8_t out[SHA256_DIGEST_LENGTH]) {
  SHA256_CTX ctx;
  SHA256_Init(&ctx);
  SHA256_Update(&ctx, data, len);
  SHA256_Final(out, &ctx);
  OPENSSL_cleanse(&ctx, sizeof(ctx));
  return out;
}

#ifndef SHA256_ASM
static void sha256_block_data_order(uint32_t *state, const uint8_t *in,
                                    size_t num);
#endif

void SHA256_Transform(SHA256_CTX *c, const uint8_t data[SHA256_CBLOCK]) {
  sha256_block_data_order(c->h, data, 1);
}

int SHA256_Update(SHA256_CTX *c, const void *data, size_t len) {
  crypto_md32_update(&sha256_block_data_order, c->h, c->data, SHA256_CBLOCK,
                     &c->num, &c->Nh, &c->Nl, data, len);
  return 1;
}

int SHA224_Update(SHA256_CTX *ctx, const void *data, size_t len) {
  return SHA256_Update(ctx, data, len);
}

static int sha256_final_impl(uint8_t *out, size_t md_len, SHA256_CTX *c) {
  crypto_md32_final(&sha256_block_data_order, c->h, c->data, SHA256_CBLOCK,
                    &c->num, c->Nh, c->Nl, /*is_big_endian=*/1);

  // TODO(davidben): This overflow check one of the few places a low-level hash
  // 'final' function can fail. SHA-512 does not have a corresponding check.
  // These functions already misbehave if the caller arbitrarily mutates |c|, so
  // can we assume one of |SHA256_Init| or |SHA224_Init| was used?
  if (md_len > SHA256_DIGEST_LENGTH) {
    return 0;
  }

  assert(md_len % 4 == 0);
  const size_t out_words = md_len / 4;
  for (size_t i = 0; i < out_words; i++) {
    CRYPTO_store_u32_be(out, c->h[i]);
    out += 4;
  }

  FIPS_service_indicator_update_state();
  return 1;
}

int SHA256_Final(uint8_t out[SHA256_DIGEST_LENGTH], SHA256_CTX *c) {
  // Ideally we would assert |sha->md_len| is |SHA256_DIGEST_LENGTH| to match
  // the size hint, but calling code often pairs |SHA224_Init| with
  // |SHA256_Final| and expects |sha->md_len| to carry the size over.
  //
  // TODO(davidben): Add an assert and fix code to match them up.
  return sha256_final_impl(out, c->md_len, c);
}

int SHA224_Final(uint8_t out[SHA224_DIGEST_LENGTH], SHA256_CTX *ctx) {
  // This function must be paired with |SHA224_Init|, which sets |ctx->md_len|
  // to |SHA224_DIGEST_LENGTH|.
  assert(ctx->md_len == SHA224_DIGEST_LENGTH);
  return sha256_final_impl(out, SHA224_DIGEST_LENGTH, ctx);
}

#ifndef SHA256_ASM
static const uint32_t K256[64] = {
    0x428a2f98UL, 0x71374491UL, 0xb5c0fbcfUL, 0xe9b5dba5UL, 0x3956c25bUL,
    0x59f111f1UL, 0x923f82a4UL, 0xab1c5ed5UL, 0xd807aa98UL, 0x12835b01UL,
    0x243185beUL, 0x550c7dc3UL, 0x72be5d74UL, 0x80deb1feUL, 0x9bdc06a7UL,
    0xc19bf174UL, 0xe49b69c1UL, 0xefbe4786UL, 0x0fc19dc6UL, 0x240ca1ccUL,
    0x2de92c6fUL, 0x4a7484aaUL, 0x5cb0a9dcUL, 0x76f988daUL, 0x983e5152UL,
    0xa831c66dUL, 0xb00327c8UL, 0xbf597fc7UL, 0xc6e00bf3UL, 0xd5a79147UL,
    0x06ca6351UL, 0x14292967UL, 0x27b70a85UL, 0x2e1b2138UL, 0x4d2c6dfcUL,
    0x53380d13UL, 0x650a7354UL, 0x766a0abbUL, 0x81c2c92eUL, 0x92722c85UL,
    0xa2bfe8a1UL, 0xa81a664bUL, 0xc24b8b70UL, 0xc76c51a3UL, 0xd192e819UL,
    0xd6990624UL, 0xf40e3585UL, 0x106aa070UL, 0x19a4c116UL, 0x1e376c08UL,
    0x2748774cUL, 0x34b0bcb5UL, 0x391c0cb3UL, 0x4ed8aa4aUL, 0x5b9cca4fUL,
    0x682e6ff3UL, 0x748f82eeUL, 0x78a5636fUL, 0x84c87814UL, 0x8cc70208UL,
    0x90befffaUL, 0xa4506cebUL, 0xbef9a3f7UL, 0xc67178f2UL};

// See FIPS 180-4, section 4.1.2.
#define Sigma0(x)                                       \
  (CRYPTO_rotr_u32((x), 2) ^ CRYPTO_rotr_u32((x), 13) ^ \
   CRYPTO_rotr_u32((x), 22))
#define Sigma1(x)                                       \
  (CRYPTO_rotr_u32((x), 6) ^ CRYPTO_rotr_u32((x), 11) ^ \
   CRYPTO_rotr_u32((x), 25))
#define sigma0(x) \
  (CRYPTO_rotr_u32((x), 7) ^ CRYPTO_rotr_u32((x), 18) ^ ((x) >> 3))
#define sigma1(x) \
  (CRYPTO_rotr_u32((x), 17) ^ CRYPTO_rotr_u32((x), 19) ^ ((x) >> 10))

#define Ch(x, y, z) (((x) & (y)) ^ ((~(x)) & (z)))
#define Maj(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

#define ROUND_00_15(i, a, b, c, d, e, f, g, h)   \
  do {                                           \
    T1 += h + Sigma1(e) + Ch(e, f, g) + K256[i]; \
    h = Sigma0(a) + Maj(a, b, c);                \
    d += T1;                                     \
    h += T1;                                     \
  } while (0)

#define ROUND_16_63(i, a, b, c, d, e, f, g, h, X)      \
  do {                                                 \
    s0 = X[(i + 1) & 0x0f];                            \
    s0 = sigma0(s0);                                   \
    s1 = X[(i + 14) & 0x0f];                           \
    s1 = sigma1(s1);                                   \
    T1 = X[(i) & 0x0f] += s0 + s1 + X[(i + 9) & 0x0f]; \
    ROUND_00_15(i, a, b, c, d, e, f, g, h);            \
  } while (0)

static void sha256_block_data_order(uint32_t *state, const uint8_t *data,
                                    size_t num) {
  uint32_t a, b, c, d, e, f, g, h, s0, s1, T1;
  uint32_t X[16];
  int i;

  while (num--) {
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    T1 = X[0] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(0, a, b, c, d, e, f, g, h);
    T1 = X[1] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(1, h, a, b, c, d, e, f, g);
    T1 = X[2] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(2, g, h, a, b, c, d, e, f);
    T1 = X[3] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(3, f, g, h, a, b, c, d, e);
    T1 = X[4] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(4, e, f, g, h, a, b, c, d);
    T1 = X[5] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(5, d, e, f, g, h, a, b, c);
    T1 = X[6] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(6, c, d, e, f, g, h, a, b);
    T1 = X[7] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(7, b, c, d, e, f, g, h, a);
    T1 = X[8] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(8, a, b, c, d, e, f, g, h);
    T1 = X[9] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(9, h, a, b, c, d, e, f, g);
    T1 = X[10] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(10, g, h, a, b, c, d, e, f);
    T1 = X[11] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(11, f, g, h, a, b, c, d, e);
    T1 = X[12] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(12, e, f, g, h, a, b, c, d);
    T1 = X[13] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(13, d, e, f, g, h, a, b, c);
    T1 = X[14] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(14, c, d, e, f, g, h, a, b);
    T1 = X[15] = CRYPTO_load_u32_be(data);
    data += 4;
    ROUND_00_15(15, b, c, d, e, f, g, h, a);

    for (i = 16; i < 64; i += 8) {
      ROUND_16_63(i + 0, a, b, c, d, e, f, g, h, X);
      ROUND_16_63(i + 1, h, a, b, c, d, e, f, g, X);
      ROUND_16_63(i + 2, g, h, a, b, c, d, e, f, X);
      ROUND_16_63(i + 3, f, g, h, a, b, c, d, e, X);
      ROUND_16_63(i + 4, e, f, g, h, a, b, c, d, X);
      ROUND_16_63(i + 5, d, e, f, g, h, a, b, c, X);
      ROUND_16_63(i + 6, c, d, e, f, g, h, a, b, X);
      ROUND_16_63(i + 7, b, c, d, e, f, g, h, a, X);
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
}

#endif  // !SHA256_ASM

void SHA256_TransformBlocks(uint32_t state[8], const uint8_t *data,
                            size_t num_blocks) {
  sha256_block_data_order(state, data, num_blocks);
}

#undef Sigma0
#undef Sigma1
#undef sigma0
#undef sigma1
#undef Ch
#undef Maj
#undef ROUND_00_15
#undef ROUND_16_63
