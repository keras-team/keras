/* ====================================================================
 * Copyright (c) 1998-2007 The OpenSSL Project.  All rights reserved.
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
 * ====================================================================
 *
 * This product includes cryptographic software written by Eric Young
 * (eay@cryptsoft.com).  This product includes software written by Tim
 * Hudson (tjh@cryptsoft.com). */

#include <assert.h>

#include <openssl/digest.h>
#include <openssl/hmac.h>
#include <openssl/mem.h>

#include "internal.h"
#include "../../internal.h"
#include "../service_indicator/internal.h"


// tls1_P_hash computes the TLS P_<hash> function as described in RFC 5246,
// section 5. It XORs |out_len| bytes to |out|, using |md| as the hash and
// |secret| as the secret. |label|, |seed1|, and |seed2| are concatenated to
// form the seed parameter. It returns true on success and false on failure.
static int tls1_P_hash(uint8_t *out, size_t out_len,
                       const EVP_MD *md,
                       const uint8_t *secret, size_t secret_len,
                       const char *label, size_t label_len,
                       const uint8_t *seed1, size_t seed1_len,
                       const uint8_t *seed2, size_t seed2_len) {
  HMAC_CTX ctx, ctx_tmp, ctx_init;
  uint8_t A1[EVP_MAX_MD_SIZE];
  unsigned A1_len;
  int ret = 0;

  const size_t chunk = EVP_MD_size(md);
  HMAC_CTX_init(&ctx);
  HMAC_CTX_init(&ctx_tmp);
  HMAC_CTX_init(&ctx_init);

  if (!HMAC_Init_ex(&ctx_init, secret, secret_len, md, NULL) ||
      !HMAC_CTX_copy_ex(&ctx, &ctx_init) ||
      !HMAC_Update(&ctx, (const uint8_t *) label, label_len) ||
      !HMAC_Update(&ctx, seed1, seed1_len) ||
      !HMAC_Update(&ctx, seed2, seed2_len) ||
      !HMAC_Final(&ctx, A1, &A1_len)) {
    goto err;
  }

  for (;;) {
    unsigned len_u;
    uint8_t hmac[EVP_MAX_MD_SIZE];
    if (!HMAC_CTX_copy_ex(&ctx, &ctx_init) ||
        !HMAC_Update(&ctx, A1, A1_len) ||
        // Save a copy of |ctx| to compute the next A1 value below.
        (out_len > chunk && !HMAC_CTX_copy_ex(&ctx_tmp, &ctx)) ||
        !HMAC_Update(&ctx, (const uint8_t *) label, label_len) ||
        !HMAC_Update(&ctx, seed1, seed1_len) ||
        !HMAC_Update(&ctx, seed2, seed2_len) ||
        !HMAC_Final(&ctx, hmac, &len_u)) {
      goto err;
    }
    size_t len = len_u;
    assert(len == chunk);

    // XOR the result into |out|.
    if (len > out_len) {
      len = out_len;
    }
    for (size_t i = 0; i < len; i++) {
      out[i] ^= hmac[i];
    }
    out += len;
    out_len -= len;

    if (out_len == 0) {
      break;
    }

    // Calculate the next A1 value.
    if (!HMAC_Final(&ctx_tmp, A1, &A1_len)) {
      goto err;
    }
  }

  ret = 1;

err:
  OPENSSL_cleanse(A1, sizeof(A1));
  HMAC_CTX_cleanup(&ctx);
  HMAC_CTX_cleanup(&ctx_tmp);
  HMAC_CTX_cleanup(&ctx_init);
  return ret;
}

int CRYPTO_tls1_prf(const EVP_MD *digest,
                    uint8_t *out, size_t out_len,
                    const uint8_t *secret, size_t secret_len,
                    const char *label, size_t label_len,
                    const uint8_t *seed1, size_t seed1_len,
                    const uint8_t *seed2, size_t seed2_len) {
  if (out_len == 0) {
    return 1;
  }

  OPENSSL_memset(out, 0, out_len);

  const EVP_MD *const original_digest = digest;
  FIPS_service_indicator_lock_state();
  int ret = 0;

  if (digest == EVP_md5_sha1()) {
    // If using the MD5/SHA1 PRF, |secret| is partitioned between MD5 and SHA-1.
    size_t secret_half = secret_len - (secret_len / 2);
    if (!tls1_P_hash(out, out_len, EVP_md5(), secret, secret_half, label,
                     label_len, seed1, seed1_len, seed2, seed2_len)) {
      goto end;
    }

    // Note that, if |secret_len| is odd, the two halves share a byte.
    secret += secret_len - secret_half;
    secret_len = secret_half;
    digest = EVP_sha1();
  }

  ret = tls1_P_hash(out, out_len, digest, secret, secret_len, label, label_len,
                    seed1, seed1_len, seed2, seed2_len);

end:
  FIPS_service_indicator_unlock_state();
  if (ret) {
    TLSKDF_verify_service_indicator(original_digest);
  }
  return ret;
}
