/* Written by Dr Stephen N Henson (steve@openssl.org) for the OpenSSL
 * project 2005.
 */
/* ====================================================================
 * Copyright (c) 2005 The OpenSSL Project.  All rights reserved.
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
 *    for use in the OpenSSL Toolkit. (http://www.OpenSSL.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    licensing@OpenSSL.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.OpenSSL.org/)"
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

#include <openssl/rsa.h>

#include <assert.h>
#include <limits.h>
#include <string.h>

#include <openssl/bn.h>
#include <openssl/digest.h>
#include <openssl/err.h>
#include <openssl/mem.h>
#include <openssl/rand.h>
#include <openssl/sha.h>

#include "internal.h"
#include "../service_indicator/internal.h"
#include "../../internal.h"


#define RSA_PKCS1_PADDING_SIZE 11

int RSA_padding_add_PKCS1_type_1(uint8_t *to, size_t to_len,
                                 const uint8_t *from, size_t from_len) {
  // See RFC 8017, section 9.2.
  if (to_len < RSA_PKCS1_PADDING_SIZE) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_KEY_SIZE_TOO_SMALL);
    return 0;
  }

  if (from_len > to_len - RSA_PKCS1_PADDING_SIZE) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_DIGEST_TOO_BIG_FOR_RSA_KEY);
    return 0;
  }

  to[0] = 0;
  to[1] = 1;
  OPENSSL_memset(to + 2, 0xff, to_len - 3 - from_len);
  to[to_len - from_len - 1] = 0;
  OPENSSL_memcpy(to + to_len - from_len, from, from_len);
  return 1;
}

int RSA_padding_check_PKCS1_type_1(uint8_t *out, size_t *out_len,
                                   size_t max_out, const uint8_t *from,
                                   size_t from_len) {
  // See RFC 8017, section 9.2. This is part of signature verification and thus
  // does not need to run in constant-time.
  if (from_len < 2) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_DATA_TOO_SMALL);
    return 0;
  }

  // Check the header.
  if (from[0] != 0 || from[1] != 1) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_BLOCK_TYPE_IS_NOT_01);
    return 0;
  }

  // Scan over padded data, looking for the 00.
  size_t pad;
  for (pad = 2 /* header */; pad < from_len; pad++) {
    if (from[pad] == 0x00) {
      break;
    }

    if (from[pad] != 0xff) {
      OPENSSL_PUT_ERROR(RSA, RSA_R_BAD_FIXED_HEADER_DECRYPT);
      return 0;
    }
  }

  if (pad == from_len) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_NULL_BEFORE_BLOCK_MISSING);
    return 0;
  }

  if (pad < 2 /* header */ + 8) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_BAD_PAD_BYTE_COUNT);
    return 0;
  }

  // Skip over the 00.
  pad++;

  if (from_len - pad > max_out) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_DATA_TOO_LARGE);
    return 0;
  }

  OPENSSL_memcpy(out, from + pad, from_len - pad);
  *out_len = from_len - pad;
  return 1;
}

static void rand_nonzero(uint8_t *out, size_t len) {
  FIPS_service_indicator_lock_state();
  RAND_bytes(out, len);

  for (size_t i = 0; i < len; i++) {
    while (out[i] == 0) {
      RAND_bytes(out + i, 1);
    }
  }

  FIPS_service_indicator_unlock_state();
}

int RSA_padding_add_PKCS1_type_2(uint8_t *to, size_t to_len,
                                 const uint8_t *from, size_t from_len) {
  // See RFC 8017, section 7.2.1.
  if (to_len < RSA_PKCS1_PADDING_SIZE) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_KEY_SIZE_TOO_SMALL);
    return 0;
  }

  if (from_len > to_len - RSA_PKCS1_PADDING_SIZE) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_DATA_TOO_LARGE_FOR_KEY_SIZE);
    return 0;
  }

  to[0] = 0;
  to[1] = 2;

  size_t padding_len = to_len - 3 - from_len;
  rand_nonzero(to + 2, padding_len);
  to[2 + padding_len] = 0;
  OPENSSL_memcpy(to + to_len - from_len, from, from_len);
  return 1;
}

int RSA_padding_check_PKCS1_type_2(uint8_t *out, size_t *out_len,
                                   size_t max_out, const uint8_t *from,
                                   size_t from_len) {
  if (from_len == 0) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_EMPTY_PUBLIC_KEY);
    return 0;
  }

  // PKCS#1 v1.5 decryption. See "PKCS #1 v2.2: RSA Cryptography
  // Standard", section 7.2.2.
  if (from_len < RSA_PKCS1_PADDING_SIZE) {
    // |from| is zero-padded to the size of the RSA modulus, a public value, so
    // this can be rejected in non-constant time.
    OPENSSL_PUT_ERROR(RSA, RSA_R_KEY_SIZE_TOO_SMALL);
    return 0;
  }

  crypto_word_t first_byte_is_zero = constant_time_eq_w(from[0], 0);
  crypto_word_t second_byte_is_two = constant_time_eq_w(from[1], 2);

  crypto_word_t zero_index = 0, looking_for_index = CONSTTIME_TRUE_W;
  for (size_t i = 2; i < from_len; i++) {
    crypto_word_t equals0 = constant_time_is_zero_w(from[i]);
    zero_index =
        constant_time_select_w(looking_for_index & equals0, i, zero_index);
    looking_for_index = constant_time_select_w(equals0, 0, looking_for_index);
  }

  // The input must begin with 00 02.
  crypto_word_t valid_index = first_byte_is_zero;
  valid_index &= second_byte_is_two;

  // We must have found the end of PS.
  valid_index &= ~looking_for_index;

  // PS must be at least 8 bytes long, and it starts two bytes into |from|.
  valid_index &= constant_time_ge_w(zero_index, 2 + 8);

  // Skip the zero byte.
  zero_index++;

  // NOTE: Although this logic attempts to be constant time, the API contracts
  // of this function and |RSA_decrypt| with |RSA_PKCS1_PADDING| make it
  // impossible to completely avoid Bleichenbacher's attack. Consumers should
  // use |RSA_PADDING_NONE| and perform the padding check in constant-time
  // combined with a swap to a random session key or other mitigation.
  CONSTTIME_DECLASSIFY(&valid_index, sizeof(valid_index));
  CONSTTIME_DECLASSIFY(&zero_index, sizeof(zero_index));

  if (!valid_index) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_PKCS_DECODING_ERROR);
    return 0;
  }

  const size_t msg_len = from_len - zero_index;
  if (msg_len > max_out) {
    // This shouldn't happen because this function is always called with
    // |max_out| as the key size and |from_len| is bounded by the key size.
    OPENSSL_PUT_ERROR(RSA, RSA_R_PKCS_DECODING_ERROR);
    return 0;
  }

  OPENSSL_memcpy(out, &from[zero_index], msg_len);
  *out_len = msg_len;
  return 1;
}

int RSA_padding_add_none(uint8_t *to, size_t to_len, const uint8_t *from,
                         size_t from_len) {
  if (from_len > to_len) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_DATA_TOO_LARGE_FOR_KEY_SIZE);
    return 0;
  }

  if (from_len < to_len) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_DATA_TOO_SMALL);
    return 0;
  }

  OPENSSL_memcpy(to, from, from_len);
  return 1;
}

static int PKCS1_MGF1(uint8_t *out, size_t len, const uint8_t *seed,
                      size_t seed_len, const EVP_MD *md) {
  int ret = 0;
  EVP_MD_CTX ctx;
  EVP_MD_CTX_init(&ctx);
  FIPS_service_indicator_lock_state();

  size_t md_len = EVP_MD_size(md);

  for (uint32_t i = 0; len > 0; i++) {
    uint8_t counter[4];
    counter[0] = (uint8_t)(i >> 24);
    counter[1] = (uint8_t)(i >> 16);
    counter[2] = (uint8_t)(i >> 8);
    counter[3] = (uint8_t)i;
    if (!EVP_DigestInit_ex(&ctx, md, NULL) ||
        !EVP_DigestUpdate(&ctx, seed, seed_len) ||
        !EVP_DigestUpdate(&ctx, counter, sizeof(counter))) {
      goto err;
    }

    if (md_len <= len) {
      if (!EVP_DigestFinal_ex(&ctx, out, NULL)) {
        goto err;
      }
      out += md_len;
      len -= md_len;
    } else {
      uint8_t digest[EVP_MAX_MD_SIZE];
      if (!EVP_DigestFinal_ex(&ctx, digest, NULL)) {
        goto err;
      }
      OPENSSL_memcpy(out, digest, len);
      len = 0;
    }
  }

  ret = 1;

err:
  EVP_MD_CTX_cleanup(&ctx);
  FIPS_service_indicator_unlock_state();
  return ret;
}

int RSA_padding_add_PKCS1_OAEP_mgf1(uint8_t *to, size_t to_len,
                                    const uint8_t *from, size_t from_len,
                                    const uint8_t *param, size_t param_len,
                                    const EVP_MD *md, const EVP_MD *mgf1md) {
  if (md == NULL) {
    md = EVP_sha1();
  }
  if (mgf1md == NULL) {
    mgf1md = md;
  }

  size_t mdlen = EVP_MD_size(md);

  if (to_len < 2 * mdlen + 2) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_KEY_SIZE_TOO_SMALL);
    return 0;
  }

  size_t emlen = to_len - 1;
  if (from_len > emlen - 2 * mdlen - 1) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_DATA_TOO_LARGE_FOR_KEY_SIZE);
    return 0;
  }

  if (emlen < 2 * mdlen + 1) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_KEY_SIZE_TOO_SMALL);
    return 0;
  }

  to[0] = 0;
  uint8_t *seed = to + 1;
  uint8_t *db = to + mdlen + 1;

  uint8_t *dbmask = NULL;
  int ret = 0;
  FIPS_service_indicator_lock_state();
  if (!EVP_Digest(param, param_len, db, NULL, md, NULL)) {
    goto out;
  }
  OPENSSL_memset(db + mdlen, 0, emlen - from_len - 2 * mdlen - 1);
  db[emlen - from_len - mdlen - 1] = 0x01;
  OPENSSL_memcpy(db + emlen - from_len - mdlen, from, from_len);
  if (!RAND_bytes(seed, mdlen)) {
    goto out;
  }

  dbmask = OPENSSL_malloc(emlen - mdlen);
  if (dbmask == NULL) {
    goto out;
  }

  if (!PKCS1_MGF1(dbmask, emlen - mdlen, seed, mdlen, mgf1md)) {
    goto out;
  }
  for (size_t i = 0; i < emlen - mdlen; i++) {
    db[i] ^= dbmask[i];
  }

  uint8_t seedmask[EVP_MAX_MD_SIZE];
  if (!PKCS1_MGF1(seedmask, mdlen, db, emlen - mdlen, mgf1md)) {
    goto out;
  }
  for (size_t i = 0; i < mdlen; i++) {
    seed[i] ^= seedmask[i];
  }
  ret = 1;

out:
  OPENSSL_free(dbmask);
  FIPS_service_indicator_unlock_state();
  return ret;
}

int RSA_padding_check_PKCS1_OAEP_mgf1(uint8_t *out, size_t *out_len,
                                      size_t max_out, const uint8_t *from,
                                      size_t from_len, const uint8_t *param,
                                      size_t param_len, const EVP_MD *md,
                                      const EVP_MD *mgf1md) {
  uint8_t *db = NULL;

  if (md == NULL) {
    md = EVP_sha1();
  }
  if (mgf1md == NULL) {
    mgf1md = md;
  }

  size_t mdlen = EVP_MD_size(md);

  // The encoded message is one byte smaller than the modulus to ensure that it
  // doesn't end up greater than the modulus. Thus there's an extra "+1" here
  // compared to https://tools.ietf.org/html/rfc2437#section-9.1.1.2.
  if (from_len < 1 + 2*mdlen + 1) {
    // 'from_len' is the length of the modulus, i.e. does not depend on the
    // particular ciphertext.
    goto decoding_err;
  }

  size_t dblen = from_len - mdlen - 1;
  FIPS_service_indicator_lock_state();
  db = OPENSSL_malloc(dblen);
  if (db == NULL) {
    goto err;
  }

  const uint8_t *maskedseed = from + 1;
  const uint8_t *maskeddb = from + 1 + mdlen;

  uint8_t seed[EVP_MAX_MD_SIZE];
  if (!PKCS1_MGF1(seed, mdlen, maskeddb, dblen, mgf1md)) {
    goto err;
  }
  for (size_t i = 0; i < mdlen; i++) {
    seed[i] ^= maskedseed[i];
  }

  if (!PKCS1_MGF1(db, dblen, seed, mdlen, mgf1md)) {
    goto err;
  }
  for (size_t i = 0; i < dblen; i++) {
    db[i] ^= maskeddb[i];
  }

  uint8_t phash[EVP_MAX_MD_SIZE];
  if (!EVP_Digest(param, param_len, phash, NULL, md, NULL)) {
    goto err;
  }

  crypto_word_t bad = ~constant_time_is_zero_w(CRYPTO_memcmp(db, phash, mdlen));
  bad |= ~constant_time_is_zero_w(from[0]);

  crypto_word_t looking_for_one_byte = CONSTTIME_TRUE_W;
  size_t one_index = 0;
  for (size_t i = mdlen; i < dblen; i++) {
    crypto_word_t equals1 = constant_time_eq_w(db[i], 1);
    crypto_word_t equals0 = constant_time_eq_w(db[i], 0);
    one_index =
        constant_time_select_w(looking_for_one_byte & equals1, i, one_index);
    looking_for_one_byte =
        constant_time_select_w(equals1, 0, looking_for_one_byte);
    bad |= looking_for_one_byte & ~equals0;
  }

  bad |= looking_for_one_byte;

  // Whether the overall padding was valid or not in OAEP is public.
  if (constant_time_declassify_w(bad)) {
    goto decoding_err;
  }

  // Once the padding is known to be valid, the output length is also public.
  static_assert(sizeof(size_t) <= sizeof(crypto_word_t),
                "size_t does not fit in crypto_word_t");
  one_index = constant_time_declassify_w(one_index);

  one_index++;
  size_t mlen = dblen - one_index;
  if (max_out < mlen) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_DATA_TOO_LARGE);
    goto err;
  }

  OPENSSL_memcpy(out, db + one_index, mlen);
  *out_len = mlen;
  OPENSSL_free(db);
  FIPS_service_indicator_unlock_state();
  return 1;

decoding_err:
  // To avoid chosen ciphertext attacks, the error message should not reveal
  // which kind of decoding error happened.
  OPENSSL_PUT_ERROR(RSA, RSA_R_OAEP_DECODING_ERROR);
 err:
  OPENSSL_free(db);
  FIPS_service_indicator_unlock_state();
  return 0;
}

static const uint8_t kPSSZeroes[] = {0, 0, 0, 0, 0, 0, 0, 0};

int RSA_verify_PKCS1_PSS_mgf1(const RSA *rsa, const uint8_t *mHash,
                              const EVP_MD *Hash, const EVP_MD *mgf1Hash,
                              const uint8_t *EM, int sLen) {
  if (mgf1Hash == NULL) {
    mgf1Hash = Hash;
  }

  int ret = 0;
  uint8_t *DB = NULL;
  EVP_MD_CTX ctx;
  EVP_MD_CTX_init(&ctx);
  FIPS_service_indicator_lock_state();

  // Negative sLen has special meanings:
  //	-1	sLen == hLen
  //	-2	salt length is autorecovered from signature
  //	-N	reserved
  size_t hLen = EVP_MD_size(Hash);
  if (sLen == -1) {
    sLen = (int)hLen;
  } else if (sLen == -2) {
    sLen = -2;
  } else if (sLen < -2) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_SLEN_CHECK_FAILED);
    goto err;
  }

  unsigned MSBits = (BN_num_bits(rsa->n) - 1) & 0x7;
  size_t emLen = RSA_size(rsa);
  if (EM[0] & (0xFF << MSBits)) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_FIRST_OCTET_INVALID);
    goto err;
  }
  if (MSBits == 0) {
    EM++;
    emLen--;
  }
  // |sLen| may be -2 for the non-standard salt length recovery mode.
  if (emLen < hLen + 2 ||
      (sLen >= 0 && emLen < hLen + (size_t)sLen + 2)) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_DATA_TOO_LARGE);
    goto err;
  }
  if (EM[emLen - 1] != 0xbc) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_LAST_OCTET_INVALID);
    goto err;
  }
  size_t maskedDBLen = emLen - hLen - 1;
  const uint8_t *H = EM + maskedDBLen;
  DB = OPENSSL_malloc(maskedDBLen);
  if (!DB) {
    goto err;
  }
  if (!PKCS1_MGF1(DB, maskedDBLen, H, hLen, mgf1Hash)) {
    goto err;
  }
  for (size_t i = 0; i < maskedDBLen; i++) {
    DB[i] ^= EM[i];
  }
  if (MSBits) {
    DB[0] &= 0xFF >> (8 - MSBits);
  }
  // This step differs slightly from EMSA-PSS-VERIFY (RFC 8017) step 10 because
  // it accepts a non-standard salt recovery flow. DB should be some number of
  // zeros, a one, then the salt.
  size_t salt_start;
  for (salt_start = 0; DB[salt_start] == 0 && salt_start < maskedDBLen - 1;
       salt_start++) {
    ;
  }
  if (DB[salt_start] != 0x1) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_SLEN_RECOVERY_FAILED);
    goto err;
  }
  salt_start++;
  // If a salt length was specified, check it matches.
  if (sLen >= 0 && maskedDBLen - salt_start != (size_t)sLen) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_SLEN_CHECK_FAILED);
    goto err;
  }
  uint8_t H_[EVP_MAX_MD_SIZE];
  if (!EVP_DigestInit_ex(&ctx, Hash, NULL) ||
      !EVP_DigestUpdate(&ctx, kPSSZeroes, sizeof(kPSSZeroes)) ||
      !EVP_DigestUpdate(&ctx, mHash, hLen) ||
      !EVP_DigestUpdate(&ctx, DB + salt_start, maskedDBLen - salt_start) ||
      !EVP_DigestFinal_ex(&ctx, H_, NULL)) {
    goto err;
  }
  if (OPENSSL_memcmp(H_, H, hLen) != 0) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_BAD_SIGNATURE);
    goto err;
  }

  ret = 1;

err:
  OPENSSL_free(DB);
  EVP_MD_CTX_cleanup(&ctx);
  FIPS_service_indicator_unlock_state();
  return ret;
}

int RSA_padding_add_PKCS1_PSS_mgf1(const RSA *rsa, unsigned char *EM,
                                   const unsigned char *mHash,
                                   const EVP_MD *Hash, const EVP_MD *mgf1Hash,
                                   int sLenRequested) {
  int ret = 0;
  size_t maskedDBLen, MSBits, emLen;
  size_t hLen;
  unsigned char *H, *salt = NULL, *p;

  if (mgf1Hash == NULL) {
    mgf1Hash = Hash;
  }

  FIPS_service_indicator_lock_state();
  hLen = EVP_MD_size(Hash);

  if (BN_is_zero(rsa->n)) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_EMPTY_PUBLIC_KEY);
    goto err;
  }

  MSBits = (BN_num_bits(rsa->n) - 1) & 0x7;
  emLen = RSA_size(rsa);
  if (MSBits == 0) {
    assert(emLen >= 1);
    *EM++ = 0;
    emLen--;
  }

  if (emLen < hLen + 2) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_DATA_TOO_LARGE_FOR_KEY_SIZE);
    goto err;
  }

  // Negative sLenRequested has special meanings:
  //   -1  sLen == hLen
  //   -2  salt length is maximized
  //   -N  reserved
  size_t sLen;
  if (sLenRequested == -1) {
    sLen = hLen;
  } else if (sLenRequested == -2) {
    sLen = emLen - hLen - 2;
  } else if (sLenRequested < 0) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_SLEN_CHECK_FAILED);
    goto err;
  } else {
    sLen = (size_t)sLenRequested;
  }

  if (emLen - hLen - 2 < sLen) {
    OPENSSL_PUT_ERROR(RSA, RSA_R_DATA_TOO_LARGE_FOR_KEY_SIZE);
    goto err;
  }

  if (sLen > 0) {
    salt = OPENSSL_malloc(sLen);
    if (!salt) {
      goto err;
    }
    if (!RAND_bytes(salt, sLen)) {
      goto err;
    }
  }
  maskedDBLen = emLen - hLen - 1;
  H = EM + maskedDBLen;

  EVP_MD_CTX ctx;
  EVP_MD_CTX_init(&ctx);
  int digest_ok = EVP_DigestInit_ex(&ctx, Hash, NULL) &&
                  EVP_DigestUpdate(&ctx, kPSSZeroes, sizeof(kPSSZeroes)) &&
                  EVP_DigestUpdate(&ctx, mHash, hLen) &&
                  EVP_DigestUpdate(&ctx, salt, sLen) &&
                  EVP_DigestFinal_ex(&ctx, H, NULL);
  EVP_MD_CTX_cleanup(&ctx);
  if (!digest_ok) {
    goto err;
  }

  // Generate dbMask in place then perform XOR on it
  if (!PKCS1_MGF1(EM, maskedDBLen, H, hLen, mgf1Hash)) {
    goto err;
  }

  p = EM;

  // Initial PS XORs with all zeroes which is a NOP so just update
  // pointer. Note from a test above this value is guaranteed to
  // be non-negative.
  p += emLen - sLen - hLen - 2;
  *p++ ^= 0x1;
  if (sLen > 0) {
    for (size_t i = 0; i < sLen; i++) {
      *p++ ^= salt[i];
    }
  }
  if (MSBits) {
    EM[0] &= 0xFF >> (8 - MSBits);
  }

  // H is already in place so just set final 0xbc

  EM[emLen - 1] = 0xbc;

  ret = 1;

err:
  OPENSSL_free(salt);
  FIPS_service_indicator_unlock_state();

  return ret;
}
