/* Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

#ifndef OPENSSL_HEADER_SERVICE_INDICATOR_INTERNAL_H
#define OPENSSL_HEADER_SERVICE_INDICATOR_INTERNAL_H

#include <openssl/base.h>
#include <openssl/service_indicator.h>

#if defined(BORINGSSL_FIPS)

// FIPS_service_indicator_update_state records that an approved service has been
// invoked.
void FIPS_service_indicator_update_state(void);

// FIPS_service_indicator_lock_state and |FIPS_service_indicator_unlock_state|
// stop |FIPS_service_indicator_update_state| from actually updating the service
// indicator. This is used when a primitive calls a potentially approved
// primitive to avoid false positives. For example, just because a key
// generation calls |RAND_bytes| (and thus the approved DRBG) doesn't mean that
// the key generation operation itself is approved.
//
// This lock nests: i.e. locking twice is fine so long as each lock is paired
// with an unlock. If the (64-bit) counter overflows, the process aborts.
void FIPS_service_indicator_lock_state(void);
void FIPS_service_indicator_unlock_state(void);

// The following functions may call |FIPS_service_indicator_update_state| if
// their parameter specifies an approved operation.

void AEAD_GCM_verify_service_indicator(const EVP_AEAD_CTX *ctx);
void AEAD_CCM_verify_service_indicator(const EVP_AEAD_CTX *ctx);
void EC_KEY_keygen_verify_service_indicator(const EC_KEY *eckey);
void ECDH_verify_service_indicator(const EC_KEY *ec_key);
void EVP_Cipher_verify_service_indicator(const EVP_CIPHER_CTX *ctx);
void EVP_DigestSign_verify_service_indicator(const EVP_MD_CTX *ctx);
void EVP_DigestVerify_verify_service_indicator(const EVP_MD_CTX *ctx);
void HMAC_verify_service_indicator(const EVP_MD *evp_md);
void TLSKDF_verify_service_indicator(const EVP_MD *dgst);

#else

// Service indicator functions are no-ops in non-FIPS builds.

OPENSSL_INLINE void FIPS_service_indicator_update_state(void) {}
OPENSSL_INLINE void FIPS_service_indicator_lock_state(void) {}
OPENSSL_INLINE void FIPS_service_indicator_unlock_state(void) {}

OPENSSL_INLINE void AEAD_GCM_verify_service_indicator(
    OPENSSL_UNUSED const EVP_AEAD_CTX *ctx) {}

OPENSSL_INLINE void AEAD_CCM_verify_service_indicator(
    OPENSSL_UNUSED const EVP_AEAD_CTX *ctx) {}

OPENSSL_INLINE void EC_KEY_keygen_verify_service_indicator(
    OPENSSL_UNUSED const EC_KEY *eckey) {}

OPENSSL_INLINE void ECDH_verify_service_indicator(
    OPENSSL_UNUSED const EC_KEY *ec_key) {}

OPENSSL_INLINE void EVP_Cipher_verify_service_indicator(
    OPENSSL_UNUSED const EVP_CIPHER_CTX *ctx) {}

OPENSSL_INLINE void EVP_DigestSign_verify_service_indicator(
    OPENSSL_UNUSED const EVP_MD_CTX *ctx) {}

OPENSSL_INLINE void EVP_DigestVerify_verify_service_indicator(
    OPENSSL_UNUSED const EVP_MD_CTX *ctx) {}

OPENSSL_INLINE void HMAC_verify_service_indicator(
    OPENSSL_UNUSED const EVP_MD *evp_md) {}

OPENSSL_INLINE void TLSKDF_verify_service_indicator(
    OPENSSL_UNUSED const EVP_MD *dgst) {}

#endif  // BORINGSSL_FIPS

#endif  // OPENSSL_HEADER_SERVICE_INDICATOR_INTERNAL_H
