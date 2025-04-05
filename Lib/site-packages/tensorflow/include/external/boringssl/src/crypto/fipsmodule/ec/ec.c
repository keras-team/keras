/* Originally written by Bodo Moeller for the OpenSSL project.
 * ====================================================================
 * Copyright (c) 1998-2005 The OpenSSL Project.  All rights reserved.
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
 * Hudson (tjh@cryptsoft.com).
 *
 */
/* ====================================================================
 * Copyright 2002 Sun Microsystems, Inc. ALL RIGHTS RESERVED.
 *
 * Portions of the attached software ("Contribution") are developed by
 * SUN MICROSYSTEMS, INC., and are contributed to the OpenSSL project.
 *
 * The Contribution is licensed pursuant to the OpenSSL open source
 * license provided above.
 *
 * The elliptic curve binary polynomial software is originally written by
 * Sheueling Chang Shantz and Douglas Stebila of Sun Microsystems
 * Laboratories. */

#include <openssl/ec.h>

#include <assert.h>
#include <string.h>

#include <openssl/bn.h>
#include <openssl/err.h>
#include <openssl/mem.h>
#include <openssl/nid.h>

#include "internal.h"
#include "../../internal.h"
#include "../bn/internal.h"
#include "../delocate.h"


static void ec_point_free(EC_POINT *point, int free_group);

static const uint8_t kP224Params[6 * 28] = {
    // p = 2^224 - 2^96 + 1
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01,
    // a
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFE,
    // b
    0xB4, 0x05, 0x0A, 0x85, 0x0C, 0x04, 0xB3, 0xAB, 0xF5, 0x41, 0x32, 0x56,
    0x50, 0x44, 0xB0, 0xB7, 0xD7, 0xBF, 0xD8, 0xBA, 0x27, 0x0B, 0x39, 0x43,
    0x23, 0x55, 0xFF, 0xB4,
    // x
    0xB7, 0x0E, 0x0C, 0xBD, 0x6B, 0xB4, 0xBF, 0x7F, 0x32, 0x13, 0x90, 0xB9,
    0x4A, 0x03, 0xC1, 0xD3, 0x56, 0xC2, 0x11, 0x22, 0x34, 0x32, 0x80, 0xD6,
    0x11, 0x5C, 0x1D, 0x21,
    // y
    0xbd, 0x37, 0x63, 0x88, 0xb5, 0xf7, 0x23, 0xfb, 0x4c, 0x22, 0xdf, 0xe6,
    0xcd, 0x43, 0x75, 0xa0, 0x5a, 0x07, 0x47, 0x64, 0x44, 0xd5, 0x81, 0x99,
    0x85, 0x00, 0x7e, 0x34,
    // order
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0x16, 0xA2, 0xE0, 0xB8, 0xF0, 0x3E, 0x13, 0xDD, 0x29, 0x45,
    0x5C, 0x5C, 0x2A, 0x3D,
};

static const uint8_t kP256Params[6 * 32] = {
    // p = 2^256 - 2^224 + 2^192 + 2^96 - 1
    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    // a
    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC,
    // b
    0x5A, 0xC6, 0x35, 0xD8, 0xAA, 0x3A, 0x93, 0xE7, 0xB3, 0xEB, 0xBD, 0x55,
    0x76, 0x98, 0x86, 0xBC, 0x65, 0x1D, 0x06, 0xB0, 0xCC, 0x53, 0xB0, 0xF6,
    0x3B, 0xCE, 0x3C, 0x3E, 0x27, 0xD2, 0x60, 0x4B,
    // x
    0x6B, 0x17, 0xD1, 0xF2, 0xE1, 0x2C, 0x42, 0x47, 0xF8, 0xBC, 0xE6, 0xE5,
    0x63, 0xA4, 0x40, 0xF2, 0x77, 0x03, 0x7D, 0x81, 0x2D, 0xEB, 0x33, 0xA0,
    0xF4, 0xA1, 0x39, 0x45, 0xD8, 0x98, 0xC2, 0x96,
    // y
    0x4f, 0xe3, 0x42, 0xe2, 0xfe, 0x1a, 0x7f, 0x9b, 0x8e, 0xe7, 0xeb, 0x4a,
    0x7c, 0x0f, 0x9e, 0x16, 0x2b, 0xce, 0x33, 0x57, 0x6b, 0x31, 0x5e, 0xce,
    0xcb, 0xb6, 0x40, 0x68, 0x37, 0xbf, 0x51, 0xf5,
    // order
    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xBC, 0xE6, 0xFA, 0xAD, 0xA7, 0x17, 0x9E, 0x84,
    0xF3, 0xB9, 0xCA, 0xC2, 0xFC, 0x63, 0x25, 0x51,
};

static const uint8_t kP384Params[6 * 48] = {
    // p = 2^384 - 2^128 - 2^96 + 2^32 - 1
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF,
    // a
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFC,
    // b
    0xB3, 0x31, 0x2F, 0xA7, 0xE2, 0x3E, 0xE7, 0xE4, 0x98, 0x8E, 0x05, 0x6B,
    0xE3, 0xF8, 0x2D, 0x19, 0x18, 0x1D, 0x9C, 0x6E, 0xFE, 0x81, 0x41, 0x12,
    0x03, 0x14, 0x08, 0x8F, 0x50, 0x13, 0x87, 0x5A, 0xC6, 0x56, 0x39, 0x8D,
    0x8A, 0x2E, 0xD1, 0x9D, 0x2A, 0x85, 0xC8, 0xED, 0xD3, 0xEC, 0x2A, 0xEF,
    // x
    0xAA, 0x87, 0xCA, 0x22, 0xBE, 0x8B, 0x05, 0x37, 0x8E, 0xB1, 0xC7, 0x1E,
    0xF3, 0x20, 0xAD, 0x74, 0x6E, 0x1D, 0x3B, 0x62, 0x8B, 0xA7, 0x9B, 0x98,
    0x59, 0xF7, 0x41, 0xE0, 0x82, 0x54, 0x2A, 0x38, 0x55, 0x02, 0xF2, 0x5D,
    0xBF, 0x55, 0x29, 0x6C, 0x3A, 0x54, 0x5E, 0x38, 0x72, 0x76, 0x0A, 0xB7,
    // y
    0x36, 0x17, 0xde, 0x4a, 0x96, 0x26, 0x2c, 0x6f, 0x5d, 0x9e, 0x98, 0xbf,
    0x92, 0x92, 0xdc, 0x29, 0xf8, 0xf4, 0x1d, 0xbd, 0x28, 0x9a, 0x14, 0x7c,
    0xe9, 0xda, 0x31, 0x13, 0xb5, 0xf0, 0xb8, 0xc0, 0x0a, 0x60, 0xb1, 0xce,
    0x1d, 0x7e, 0x81, 0x9d, 0x7a, 0x43, 0x1d, 0x7c, 0x90, 0xea, 0x0e, 0x5f,
    // order
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xC7, 0x63, 0x4D, 0x81, 0xF4, 0x37, 0x2D, 0xDF, 0x58, 0x1A, 0x0D, 0xB2,
    0x48, 0xB0, 0xA7, 0x7A, 0xEC, 0xEC, 0x19, 0x6A, 0xCC, 0xC5, 0x29, 0x73,
};

static const uint8_t kP521Params[6 * 66] = {
    // p = 2^521 - 1
    0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    // a
    0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC,
    // b
    0x00, 0x51, 0x95, 0x3E, 0xB9, 0x61, 0x8E, 0x1C, 0x9A, 0x1F, 0x92, 0x9A,
    0x21, 0xA0, 0xB6, 0x85, 0x40, 0xEE, 0xA2, 0xDA, 0x72, 0x5B, 0x99, 0xB3,
    0x15, 0xF3, 0xB8, 0xB4, 0x89, 0x91, 0x8E, 0xF1, 0x09, 0xE1, 0x56, 0x19,
    0x39, 0x51, 0xEC, 0x7E, 0x93, 0x7B, 0x16, 0x52, 0xC0, 0xBD, 0x3B, 0xB1,
    0xBF, 0x07, 0x35, 0x73, 0xDF, 0x88, 0x3D, 0x2C, 0x34, 0xF1, 0xEF, 0x45,
    0x1F, 0xD4, 0x6B, 0x50, 0x3F, 0x00,
    // x
    0x00, 0xC6, 0x85, 0x8E, 0x06, 0xB7, 0x04, 0x04, 0xE9, 0xCD, 0x9E, 0x3E,
    0xCB, 0x66, 0x23, 0x95, 0xB4, 0x42, 0x9C, 0x64, 0x81, 0x39, 0x05, 0x3F,
    0xB5, 0x21, 0xF8, 0x28, 0xAF, 0x60, 0x6B, 0x4D, 0x3D, 0xBA, 0xA1, 0x4B,
    0x5E, 0x77, 0xEF, 0xE7, 0x59, 0x28, 0xFE, 0x1D, 0xC1, 0x27, 0xA2, 0xFF,
    0xA8, 0xDE, 0x33, 0x48, 0xB3, 0xC1, 0x85, 0x6A, 0x42, 0x9B, 0xF9, 0x7E,
    0x7E, 0x31, 0xC2, 0xE5, 0xBD, 0x66,
    // y
    0x01, 0x18, 0x39, 0x29, 0x6a, 0x78, 0x9a, 0x3b, 0xc0, 0x04, 0x5c, 0x8a,
    0x5f, 0xb4, 0x2c, 0x7d, 0x1b, 0xd9, 0x98, 0xf5, 0x44, 0x49, 0x57, 0x9b,
    0x44, 0x68, 0x17, 0xaf, 0xbd, 0x17, 0x27, 0x3e, 0x66, 0x2c, 0x97, 0xee,
    0x72, 0x99, 0x5e, 0xf4, 0x26, 0x40, 0xc5, 0x50, 0xb9, 0x01, 0x3f, 0xad,
    0x07, 0x61, 0x35, 0x3c, 0x70, 0x86, 0xa2, 0x72, 0xc2, 0x40, 0x88, 0xbe,
    0x94, 0x76, 0x9f, 0xd1, 0x66, 0x50,
    // order
    0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFA, 0x51, 0x86,
    0x87, 0x83, 0xBF, 0x2F, 0x96, 0x6B, 0x7F, 0xCC, 0x01, 0x48, 0xF7, 0x09,
    0xA5, 0xD0, 0x3B, 0xB5, 0xC9, 0xB8, 0x89, 0x9C, 0x47, 0xAE, 0xBB, 0x6F,
    0xB7, 0x1E, 0x91, 0x38, 0x64, 0x09,
};

DEFINE_METHOD_FUNCTION(struct built_in_curves, OPENSSL_built_in_curves) {
  // 1.3.132.0.35
  static const uint8_t kOIDP521[] = {0x2b, 0x81, 0x04, 0x00, 0x23};
  out->curves[0].nid = NID_secp521r1;
  out->curves[0].oid = kOIDP521;
  out->curves[0].oid_len = sizeof(kOIDP521);
  out->curves[0].comment = "NIST P-521";
  out->curves[0].param_len = 66;
  out->curves[0].params = kP521Params;
  out->curves[0].method = EC_GFp_mont_method();

  // 1.3.132.0.34
  static const uint8_t kOIDP384[] = {0x2b, 0x81, 0x04, 0x00, 0x22};
  out->curves[1].nid = NID_secp384r1;
  out->curves[1].oid = kOIDP384;
  out->curves[1].oid_len = sizeof(kOIDP384);
  out->curves[1].comment = "NIST P-384";
  out->curves[1].param_len = 48;
  out->curves[1].params = kP384Params;
  out->curves[1].method = EC_GFp_mont_method();

  // 1.2.840.10045.3.1.7
  static const uint8_t kOIDP256[] = {0x2a, 0x86, 0x48, 0xce,
                                     0x3d, 0x03, 0x01, 0x07};
  out->curves[2].nid = NID_X9_62_prime256v1;
  out->curves[2].oid = kOIDP256;
  out->curves[2].oid_len = sizeof(kOIDP256);
  out->curves[2].comment = "NIST P-256";
  out->curves[2].param_len = 32;
  out->curves[2].params = kP256Params;
  out->curves[2].method =
#if !defined(OPENSSL_NO_ASM) && \
    (defined(OPENSSL_X86_64) || defined(OPENSSL_AARCH64)) &&   \
    !defined(OPENSSL_SMALL)
      EC_GFp_nistz256_method();
#else
      EC_GFp_nistp256_method();
#endif

  // 1.3.132.0.33
  static const uint8_t kOIDP224[] = {0x2b, 0x81, 0x04, 0x00, 0x21};
  out->curves[3].nid = NID_secp224r1;
  out->curves[3].oid = kOIDP224;
  out->curves[3].oid_len = sizeof(kOIDP224);
  out->curves[3].comment = "NIST P-224";
  out->curves[3].param_len = 28;
  out->curves[3].params = kP224Params;
  out->curves[3].method =
#if defined(BORINGSSL_HAS_UINT128) && !defined(OPENSSL_SMALL)
      EC_GFp_nistp224_method();
#else
      EC_GFp_mont_method();
#endif
}

EC_GROUP *ec_group_new(const EC_METHOD *meth) {
  EC_GROUP *ret;

  if (meth == NULL) {
    OPENSSL_PUT_ERROR(EC, EC_R_SLOT_FULL);
    return NULL;
  }

  if (meth->group_init == 0) {
    OPENSSL_PUT_ERROR(EC, ERR_R_SHOULD_NOT_HAVE_BEEN_CALLED);
    return NULL;
  }

  ret = OPENSSL_malloc(sizeof(EC_GROUP));
  if (ret == NULL) {
    return NULL;
  }
  OPENSSL_memset(ret, 0, sizeof(EC_GROUP));

  ret->references = 1;
  ret->meth = meth;
  BN_init(&ret->order);

  if (!meth->group_init(ret)) {
    OPENSSL_free(ret);
    return NULL;
  }

  return ret;
}

static int ec_group_set_generator(EC_GROUP *group, const EC_AFFINE *generator,
                                  const BIGNUM *order) {
  assert(group->generator == NULL);

  if (!BN_copy(&group->order, order)) {
    return 0;
  }
  // Store the order in minimal form, so it can be used with |BN_ULONG| arrays.
  bn_set_minimal_width(&group->order);

  BN_MONT_CTX_free(group->order_mont);
  group->order_mont = BN_MONT_CTX_new_for_modulus(&group->order, NULL);
  if (group->order_mont == NULL) {
    return 0;
  }

  group->field_greater_than_order = BN_cmp(&group->field, order) > 0;
  if (group->field_greater_than_order) {
    BIGNUM tmp;
    BN_init(&tmp);
    int ok =
        BN_sub(&tmp, &group->field, order) &&
        bn_copy_words(group->field_minus_order.words, group->field.width, &tmp);
    BN_free(&tmp);
    if (!ok) {
      return 0;
    }
  }

  group->generator = EC_POINT_new(group);
  if (group->generator == NULL) {
    return 0;
  }
  ec_affine_to_jacobian(group, &group->generator->raw, generator);
  assert(ec_felem_equal(group, &group->one, &group->generator->raw.Z));

  // Avoid a reference cycle. |group->generator| does not maintain an owning
  // pointer to |group|.
  int is_zero = CRYPTO_refcount_dec_and_test_zero(&group->references);

  assert(!is_zero);
  (void)is_zero;
  return 1;
}

EC_GROUP *EC_GROUP_new_curve_GFp(const BIGNUM *p, const BIGNUM *a,
                                 const BIGNUM *b, BN_CTX *ctx) {
  if (BN_num_bytes(p) > EC_MAX_BYTES) {
    OPENSSL_PUT_ERROR(EC, EC_R_INVALID_FIELD);
    return NULL;
  }

  BN_CTX *new_ctx = NULL;
  if (ctx == NULL) {
    ctx = new_ctx = BN_CTX_new();
    if (ctx == NULL) {
      return NULL;
    }
  }

  // Historically, |a| and |b| were not required to be fully reduced.
  // TODO(davidben): Can this be removed?
  EC_GROUP *ret = NULL;
  BN_CTX_start(ctx);
  BIGNUM *a_reduced = BN_CTX_get(ctx);
  BIGNUM *b_reduced = BN_CTX_get(ctx);
  if (a_reduced == NULL || b_reduced == NULL ||
      !BN_nnmod(a_reduced, a, p, ctx) ||
      !BN_nnmod(b_reduced, b, p, ctx)) {
    goto err;
  }

  ret = ec_group_new(EC_GFp_mont_method());
  if (ret == NULL ||
      !ret->meth->group_set_curve(ret, p, a_reduced, b_reduced, ctx)) {
    EC_GROUP_free(ret);
    ret = NULL;
    goto err;
  }

err:
  BN_CTX_end(ctx);
  BN_CTX_free(new_ctx);
  return ret;
}

int EC_GROUP_set_generator(EC_GROUP *group, const EC_POINT *generator,
                           const BIGNUM *order, const BIGNUM *cofactor) {
  if (group->curve_name != NID_undef || group->generator != NULL ||
      generator->group != group) {
    // |EC_GROUP_set_generator| may only be used with |EC_GROUP|s returned by
    // |EC_GROUP_new_curve_GFp| and may only used once on each group.
    // |generator| must have been created from |EC_GROUP_new_curve_GFp|, not a
    // copy, so that |generator->group->generator| is set correctly.
    OPENSSL_PUT_ERROR(EC, ERR_R_SHOULD_NOT_HAVE_BEEN_CALLED);
    return 0;
  }

  if (BN_num_bytes(order) > EC_MAX_BYTES) {
    OPENSSL_PUT_ERROR(EC, EC_R_INVALID_GROUP_ORDER);
    return 0;
  }

  // Require a cofactor of one for custom curves, which implies prime order.
  if (!BN_is_one(cofactor)) {
    OPENSSL_PUT_ERROR(EC, EC_R_INVALID_COFACTOR);
    return 0;
  }

  // Require that p < 2×order. This simplifies some ECDSA operations.
  //
  // Note any curve which did not satisfy this must have been invalid or use a
  // tiny prime (less than 17). See the proof in |field_element_to_scalar| in
  // the ECDSA implementation.
  int ret = 0;
  BIGNUM *tmp = BN_new();
  if (tmp == NULL ||
      !BN_lshift1(tmp, order)) {
    goto err;
  }
  if (BN_cmp(tmp, &group->field) <= 0) {
    OPENSSL_PUT_ERROR(EC, EC_R_INVALID_GROUP_ORDER);
    goto err;
  }

  EC_AFFINE affine;
  if (!ec_jacobian_to_affine(group, &affine, &generator->raw) ||
      !ec_group_set_generator(group, &affine, order)) {
    goto err;
  }

  ret = 1;

err:
  BN_free(tmp);
  return ret;
}

static EC_GROUP *ec_group_new_from_data(const struct built_in_curve *curve) {
  EC_GROUP *group = NULL;
  BIGNUM *p = NULL, *a = NULL, *b = NULL, *order = NULL;
  int ok = 0;

  BN_CTX *ctx = BN_CTX_new();
  if (ctx == NULL) {
    goto err;
  }

  const unsigned param_len = curve->param_len;
  const uint8_t *params = curve->params;

  if (!(p = BN_bin2bn(params + 0 * param_len, param_len, NULL)) ||
      !(a = BN_bin2bn(params + 1 * param_len, param_len, NULL)) ||
      !(b = BN_bin2bn(params + 2 * param_len, param_len, NULL)) ||
      !(order = BN_bin2bn(params + 5 * param_len, param_len, NULL))) {
    OPENSSL_PUT_ERROR(EC, ERR_R_BN_LIB);
    goto err;
  }

  group = ec_group_new(curve->method);
  if (group == NULL ||
      !group->meth->group_set_curve(group, p, a, b, ctx)) {
    OPENSSL_PUT_ERROR(EC, ERR_R_EC_LIB);
    goto err;
  }

  EC_AFFINE G;
  EC_FELEM x, y;
  if (!ec_felem_from_bytes(group, &x, params + 3 * param_len, param_len) ||
      !ec_felem_from_bytes(group, &y, params + 4 * param_len, param_len) ||
      !ec_point_set_affine_coordinates(group, &G, &x, &y)) {
    goto err;
  }

  if (!ec_group_set_generator(group, &G, order)) {
    goto err;
  }

  ok = 1;

err:
  if (!ok) {
    EC_GROUP_free(group);
    group = NULL;
  }
  BN_CTX_free(ctx);
  BN_free(p);
  BN_free(a);
  BN_free(b);
  BN_free(order);
  return group;
}

// Built-in groups are allocated lazily and static once allocated.
// TODO(davidben): Make these actually static. https://crbug.com/boringssl/20.
struct built_in_groups_st {
  EC_GROUP *groups[OPENSSL_NUM_BUILT_IN_CURVES];
};
DEFINE_BSS_GET(struct built_in_groups_st, built_in_groups)
DEFINE_STATIC_MUTEX(built_in_groups_lock)

EC_GROUP *EC_GROUP_new_by_curve_name(int nid) {
  struct built_in_groups_st *groups = built_in_groups_bss_get();
  EC_GROUP **group_ptr = NULL;
  const struct built_in_curves *const curves = OPENSSL_built_in_curves();
  const struct built_in_curve *curve = NULL;
  for (size_t i = 0; i < OPENSSL_NUM_BUILT_IN_CURVES; i++) {
    if (curves->curves[i].nid == nid) {
      curve = &curves->curves[i];
      group_ptr = &groups->groups[i];
      break;
    }
  }

  if (curve == NULL) {
    OPENSSL_PUT_ERROR(EC, EC_R_UNKNOWN_GROUP);
    return NULL;
  }

  CRYPTO_STATIC_MUTEX_lock_read(built_in_groups_lock_bss_get());
  EC_GROUP *ret = *group_ptr;
  CRYPTO_STATIC_MUTEX_unlock_read(built_in_groups_lock_bss_get());
  if (ret != NULL) {
    return ret;
  }

  ret = ec_group_new_from_data(curve);
  if (ret == NULL) {
    return NULL;
  }

  EC_GROUP *to_free = NULL;
  CRYPTO_STATIC_MUTEX_lock_write(built_in_groups_lock_bss_get());
  if (*group_ptr == NULL) {
    *group_ptr = ret;
    // Filling in |ret->curve_name| makes |EC_GROUP_free| and |EC_GROUP_dup|
    // into no-ops. At this point, |ret| is considered static.
    ret->curve_name = nid;
  } else {
    to_free = ret;
    ret = *group_ptr;
  }
  CRYPTO_STATIC_MUTEX_unlock_write(built_in_groups_lock_bss_get());

  EC_GROUP_free(to_free);
  return ret;
}

void EC_GROUP_free(EC_GROUP *group) {
  if (group == NULL ||
      // Built-in curves are static.
      group->curve_name != NID_undef ||
      !CRYPTO_refcount_dec_and_test_zero(&group->references)) {
    return;
  }

  if (group->meth->group_finish != NULL) {
    group->meth->group_finish(group);
  }

  ec_point_free(group->generator, 0 /* don't free group */);
  BN_free(&group->order);
  BN_MONT_CTX_free(group->order_mont);

  OPENSSL_free(group);
}

EC_GROUP *EC_GROUP_dup(const EC_GROUP *a) {
  if (a == NULL ||
      // Built-in curves are static.
      a->curve_name != NID_undef) {
    return (EC_GROUP *)a;
  }

  // Groups are logically immutable (but for |EC_GROUP_set_generator| which must
  // be called early on), so we simply take a reference.
  EC_GROUP *group = (EC_GROUP *)a;
  CRYPTO_refcount_inc(&group->references);
  return group;
}

int EC_GROUP_cmp(const EC_GROUP *a, const EC_GROUP *b, BN_CTX *ignored) {
  // Note this function returns 0 if equal and non-zero otherwise.
  if (a == b) {
    return 0;
  }
  if (a->curve_name != b->curve_name) {
    return 1;
  }
  if (a->curve_name != NID_undef) {
    // Built-in curves may be compared by curve name alone.
    return 0;
  }

  // |a| and |b| are both custom curves. We compare the entire curve
  // structure. If |a| or |b| is incomplete (due to legacy OpenSSL mistakes,
  // custom curve construction is sadly done in two parts) but otherwise not the
  // same object, we consider them always unequal.
  return a->meth != b->meth ||
         a->generator == NULL ||
         b->generator == NULL ||
         BN_cmp(&a->order, &b->order) != 0 ||
         BN_cmp(&a->field, &b->field) != 0 ||
         !ec_felem_equal(a, &a->a, &b->a) ||
         !ec_felem_equal(a, &a->b, &b->b) ||
         !ec_GFp_simple_points_equal(a, &a->generator->raw, &b->generator->raw);
}

const EC_POINT *EC_GROUP_get0_generator(const EC_GROUP *group) {
  return group->generator;
}

const BIGNUM *EC_GROUP_get0_order(const EC_GROUP *group) {
  assert(!BN_is_zero(&group->order));
  return &group->order;
}

int EC_GROUP_get_order(const EC_GROUP *group, BIGNUM *order, BN_CTX *ctx) {
  if (BN_copy(order, EC_GROUP_get0_order(group)) == NULL) {
    return 0;
  }
  return 1;
}

int EC_GROUP_order_bits(const EC_GROUP *group) {
  return BN_num_bits(&group->order);
}

int EC_GROUP_get_cofactor(const EC_GROUP *group, BIGNUM *cofactor,
                          BN_CTX *ctx) {
  // All |EC_GROUP|s have cofactor 1.
  return BN_set_word(cofactor, 1);
}

int EC_GROUP_get_curve_GFp(const EC_GROUP *group, BIGNUM *out_p, BIGNUM *out_a,
                           BIGNUM *out_b, BN_CTX *ctx) {
  return ec_GFp_simple_group_get_curve(group, out_p, out_a, out_b);
}

int EC_GROUP_get_curve_name(const EC_GROUP *group) { return group->curve_name; }

unsigned EC_GROUP_get_degree(const EC_GROUP *group) {
  return BN_num_bits(&group->field);
}

const char *EC_curve_nid2nist(int nid) {
  switch (nid) {
    case NID_secp224r1:
      return "P-224";
    case NID_X9_62_prime256v1:
      return "P-256";
    case NID_secp384r1:
      return "P-384";
    case NID_secp521r1:
      return "P-521";
  }
  return NULL;
}

int EC_curve_nist2nid(const char *name) {
  if (strcmp(name, "P-224") == 0) {
    return NID_secp224r1;
  }
  if (strcmp(name, "P-256") == 0) {
    return NID_X9_62_prime256v1;
  }
  if (strcmp(name, "P-384") == 0) {
    return NID_secp384r1;
  }
  if (strcmp(name, "P-521") == 0) {
    return NID_secp521r1;
  }
  return NID_undef;
}

EC_POINT *EC_POINT_new(const EC_GROUP *group) {
  if (group == NULL) {
    OPENSSL_PUT_ERROR(EC, ERR_R_PASSED_NULL_PARAMETER);
    return NULL;
  }

  EC_POINT *ret = OPENSSL_malloc(sizeof *ret);
  if (ret == NULL) {
    return NULL;
  }

  ret->group = EC_GROUP_dup(group);
  ec_GFp_simple_point_init(&ret->raw);
  return ret;
}

static void ec_point_free(EC_POINT *point, int free_group) {
  if (!point) {
    return;
  }
  if (free_group) {
    EC_GROUP_free(point->group);
  }
  OPENSSL_free(point);
}

void EC_POINT_free(EC_POINT *point) {
  ec_point_free(point, 1 /* free group */);
}

void EC_POINT_clear_free(EC_POINT *point) { EC_POINT_free(point); }

int EC_POINT_copy(EC_POINT *dest, const EC_POINT *src) {
  if (EC_GROUP_cmp(dest->group, src->group, NULL) != 0) {
    OPENSSL_PUT_ERROR(EC, EC_R_INCOMPATIBLE_OBJECTS);
    return 0;
  }
  if (dest == src) {
    return 1;
  }
  ec_GFp_simple_point_copy(&dest->raw, &src->raw);
  return 1;
}

EC_POINT *EC_POINT_dup(const EC_POINT *a, const EC_GROUP *group) {
  if (a == NULL) {
    return NULL;
  }

  EC_POINT *ret = EC_POINT_new(group);
  if (ret == NULL ||
      !EC_POINT_copy(ret, a)) {
    EC_POINT_free(ret);
    return NULL;
  }

  return ret;
}

int EC_POINT_set_to_infinity(const EC_GROUP *group, EC_POINT *point) {
  if (EC_GROUP_cmp(group, point->group, NULL) != 0) {
    OPENSSL_PUT_ERROR(EC, EC_R_INCOMPATIBLE_OBJECTS);
    return 0;
  }
  ec_GFp_simple_point_set_to_infinity(group, &point->raw);
  return 1;
}

int EC_POINT_is_at_infinity(const EC_GROUP *group, const EC_POINT *point) {
  if (EC_GROUP_cmp(group, point->group, NULL) != 0) {
    OPENSSL_PUT_ERROR(EC, EC_R_INCOMPATIBLE_OBJECTS);
    return 0;
  }
  return ec_GFp_simple_is_at_infinity(group, &point->raw);
}

int EC_POINT_is_on_curve(const EC_GROUP *group, const EC_POINT *point,
                         BN_CTX *ctx) {
  if (EC_GROUP_cmp(group, point->group, NULL) != 0) {
    OPENSSL_PUT_ERROR(EC, EC_R_INCOMPATIBLE_OBJECTS);
    return 0;
  }
  return ec_GFp_simple_is_on_curve(group, &point->raw);
}

int EC_POINT_cmp(const EC_GROUP *group, const EC_POINT *a, const EC_POINT *b,
                 BN_CTX *ctx) {
  if (EC_GROUP_cmp(group, a->group, NULL) != 0 ||
      EC_GROUP_cmp(group, b->group, NULL) != 0) {
    OPENSSL_PUT_ERROR(EC, EC_R_INCOMPATIBLE_OBJECTS);
    return -1;
  }

  // Note |EC_POINT_cmp| returns zero for equality and non-zero for inequality.
  return ec_GFp_simple_points_equal(group, &a->raw, &b->raw) ? 0 : 1;
}

int EC_POINT_get_affine_coordinates_GFp(const EC_GROUP *group,
                                        const EC_POINT *point, BIGNUM *x,
                                        BIGNUM *y, BN_CTX *ctx) {
  if (group->meth->point_get_affine_coordinates == 0) {
    OPENSSL_PUT_ERROR(EC, ERR_R_SHOULD_NOT_HAVE_BEEN_CALLED);
    return 0;
  }
  if (EC_GROUP_cmp(group, point->group, NULL) != 0) {
    OPENSSL_PUT_ERROR(EC, EC_R_INCOMPATIBLE_OBJECTS);
    return 0;
  }
  EC_FELEM x_felem, y_felem;
  if (!group->meth->point_get_affine_coordinates(group, &point->raw,
                                                 x == NULL ? NULL : &x_felem,
                                                 y == NULL ? NULL : &y_felem) ||
      (x != NULL && !ec_felem_to_bignum(group, x, &x_felem)) ||
      (y != NULL && !ec_felem_to_bignum(group, y, &y_felem))) {
    return 0;
  }
  return 1;
}

int EC_POINT_get_affine_coordinates(const EC_GROUP *group,
                                    const EC_POINT *point, BIGNUM *x, BIGNUM *y,
                                    BN_CTX *ctx) {
  return EC_POINT_get_affine_coordinates_GFp(group, point, x, y, ctx);
}

void ec_affine_to_jacobian(const EC_GROUP *group, EC_RAW_POINT *out,
                           const EC_AFFINE *p) {
  out->X = p->X;
  out->Y = p->Y;
  out->Z = group->one;
}

int ec_jacobian_to_affine(const EC_GROUP *group, EC_AFFINE *out,
                          const EC_RAW_POINT *p) {
  return group->meth->point_get_affine_coordinates(group, p, &out->X, &out->Y);
}

int ec_jacobian_to_affine_batch(const EC_GROUP *group, EC_AFFINE *out,
                                const EC_RAW_POINT *in, size_t num) {
  if (group->meth->jacobian_to_affine_batch == NULL) {
    OPENSSL_PUT_ERROR(EC, ERR_R_SHOULD_NOT_HAVE_BEEN_CALLED);
    return 0;
  }
  return group->meth->jacobian_to_affine_batch(group, out, in, num);
}

int ec_point_set_affine_coordinates(const EC_GROUP *group, EC_AFFINE *out,
                                    const EC_FELEM *x, const EC_FELEM *y) {
  void (*const felem_mul)(const EC_GROUP *, EC_FELEM *r, const EC_FELEM *a,
                          const EC_FELEM *b) = group->meth->felem_mul;
  void (*const felem_sqr)(const EC_GROUP *, EC_FELEM *r, const EC_FELEM *a) =
      group->meth->felem_sqr;

  // Check if the point is on the curve.
  EC_FELEM lhs, rhs;
  felem_sqr(group, &lhs, y);                   // lhs = y^2
  felem_sqr(group, &rhs, x);                   // rhs = x^2
  ec_felem_add(group, &rhs, &rhs, &group->a);  // rhs = x^2 + a
  felem_mul(group, &rhs, &rhs, x);             // rhs = x^3 + ax
  ec_felem_add(group, &rhs, &rhs, &group->b);  // rhs = x^3 + ax + b
  if (!ec_felem_equal(group, &lhs, &rhs)) {
    OPENSSL_PUT_ERROR(EC, EC_R_POINT_IS_NOT_ON_CURVE);
    // In the event of an error, defend against the caller not checking the
    // return value by setting a known safe value. Note this may not be possible
    // if the caller is in the process of constructing an arbitrary group and
    // the generator is missing.
    if (group->generator != NULL) {
      assert(ec_felem_equal(group, &group->one, &group->generator->raw.Z));
      out->X = group->generator->raw.X;
      out->Y = group->generator->raw.Y;
    }
    return 0;
  }

  out->X = *x;
  out->Y = *y;
  return 1;
}

int EC_POINT_set_affine_coordinates_GFp(const EC_GROUP *group, EC_POINT *point,
                                        const BIGNUM *x, const BIGNUM *y,
                                        BN_CTX *ctx) {
  if (EC_GROUP_cmp(group, point->group, NULL) != 0) {
    OPENSSL_PUT_ERROR(EC, EC_R_INCOMPATIBLE_OBJECTS);
    return 0;
  }

  if (x == NULL || y == NULL) {
    OPENSSL_PUT_ERROR(EC, ERR_R_PASSED_NULL_PARAMETER);
    return 0;
  }

  EC_FELEM x_felem, y_felem;
  EC_AFFINE affine;
  if (!ec_bignum_to_felem(group, &x_felem, x) ||
      !ec_bignum_to_felem(group, &y_felem, y) ||
      !ec_point_set_affine_coordinates(group, &affine, &x_felem, &y_felem)) {
    // In the event of an error, defend against the caller not checking the
    // return value by setting a known safe value.
    ec_set_to_safe_point(group, &point->raw);
    return 0;
  }

  ec_affine_to_jacobian(group, &point->raw, &affine);
  return 1;
}

int EC_POINT_set_affine_coordinates(const EC_GROUP *group, EC_POINT *point,
                                    const BIGNUM *x, const BIGNUM *y,
                                    BN_CTX *ctx) {
  return EC_POINT_set_affine_coordinates_GFp(group, point, x, y, ctx);
}

int EC_POINT_add(const EC_GROUP *group, EC_POINT *r, const EC_POINT *a,
                 const EC_POINT *b, BN_CTX *ctx) {
  if (EC_GROUP_cmp(group, r->group, NULL) != 0 ||
      EC_GROUP_cmp(group, a->group, NULL) != 0 ||
      EC_GROUP_cmp(group, b->group, NULL) != 0) {
    OPENSSL_PUT_ERROR(EC, EC_R_INCOMPATIBLE_OBJECTS);
    return 0;
  }
  group->meth->add(group, &r->raw, &a->raw, &b->raw);
  return 1;
}

int EC_POINT_dbl(const EC_GROUP *group, EC_POINT *r, const EC_POINT *a,
                 BN_CTX *ctx) {
  if (EC_GROUP_cmp(group, r->group, NULL) != 0 ||
      EC_GROUP_cmp(group, a->group, NULL) != 0) {
    OPENSSL_PUT_ERROR(EC, EC_R_INCOMPATIBLE_OBJECTS);
    return 0;
  }
  group->meth->dbl(group, &r->raw, &a->raw);
  return 1;
}


int EC_POINT_invert(const EC_GROUP *group, EC_POINT *a, BN_CTX *ctx) {
  if (EC_GROUP_cmp(group, a->group, NULL) != 0) {
    OPENSSL_PUT_ERROR(EC, EC_R_INCOMPATIBLE_OBJECTS);
    return 0;
  }
  ec_GFp_simple_invert(group, &a->raw);
  return 1;
}

static int arbitrary_bignum_to_scalar(const EC_GROUP *group, EC_SCALAR *out,
                                      const BIGNUM *in, BN_CTX *ctx) {
  if (ec_bignum_to_scalar(group, out, in)) {
    return 1;
  }

  ERR_clear_error();

  // This is an unusual input, so we do not guarantee constant-time processing.
  const BIGNUM *order = &group->order;
  BN_CTX_start(ctx);
  BIGNUM *tmp = BN_CTX_get(ctx);
  int ok = tmp != NULL &&
           BN_nnmod(tmp, in, order, ctx) &&
           ec_bignum_to_scalar(group, out, tmp);
  BN_CTX_end(ctx);
  return ok;
}

int ec_point_mul_no_self_test(const EC_GROUP *group, EC_POINT *r,
                              const BIGNUM *g_scalar, const EC_POINT *p,
                              const BIGNUM *p_scalar, BN_CTX *ctx) {
  // Previously, this function set |r| to the point at infinity if there was
  // nothing to multiply. But, nobody should be calling this function with
  // nothing to multiply in the first place.
  if ((g_scalar == NULL && p_scalar == NULL) ||
      (p == NULL) != (p_scalar == NULL))  {
    OPENSSL_PUT_ERROR(EC, ERR_R_PASSED_NULL_PARAMETER);
    return 0;
  }

  if (EC_GROUP_cmp(group, r->group, NULL) != 0 ||
      (p != NULL && EC_GROUP_cmp(group, p->group, NULL) != 0)) {
    OPENSSL_PUT_ERROR(EC, EC_R_INCOMPATIBLE_OBJECTS);
    return 0;
  }

  int ret = 0;
  BN_CTX *new_ctx = NULL;
  if (ctx == NULL) {
    new_ctx = BN_CTX_new();
    if (new_ctx == NULL) {
      goto err;
    }
    ctx = new_ctx;
  }

  // If both |g_scalar| and |p_scalar| are non-NULL,
  // |ec_point_mul_scalar_public| would share the doublings between the two
  // products, which would be more efficient. However, we conservatively assume
  // the caller needs a constant-time operation. (ECDSA verification does not
  // use this function.)
  //
  // Previously, the low-level constant-time multiplication function aligned
  // with this function's calling convention, but this was misleading. Curves
  // which combined the two multiplications did not avoid the doubling case
  // in the incomplete addition formula and were not constant-time.

  if (g_scalar != NULL) {
    EC_SCALAR scalar;
    if (!arbitrary_bignum_to_scalar(group, &scalar, g_scalar, ctx) ||
        !ec_point_mul_scalar_base(group, &r->raw, &scalar)) {
      goto err;
    }
  }

  if (p_scalar != NULL) {
    EC_SCALAR scalar;
    EC_RAW_POINT tmp;
    if (!arbitrary_bignum_to_scalar(group, &scalar, p_scalar, ctx) ||
        !ec_point_mul_scalar(group, &tmp, &p->raw, &scalar)) {
      goto err;
    }
    if (g_scalar == NULL) {
      OPENSSL_memcpy(&r->raw, &tmp, sizeof(EC_RAW_POINT));
    } else {
      group->meth->add(group, &r->raw, &r->raw, &tmp);
    }
  }

  ret = 1;

err:
  BN_CTX_free(new_ctx);
  return ret;
}

int EC_POINT_mul(const EC_GROUP *group, EC_POINT *r, const BIGNUM *g_scalar,
                 const EC_POINT *p, const BIGNUM *p_scalar, BN_CTX *ctx) {
  boringssl_ensure_ecc_self_test();

  return ec_point_mul_no_self_test(group, r, g_scalar, p, p_scalar, ctx);
}

int ec_point_mul_scalar_public(const EC_GROUP *group, EC_RAW_POINT *r,
                               const EC_SCALAR *g_scalar, const EC_RAW_POINT *p,
                               const EC_SCALAR *p_scalar) {
  if (g_scalar == NULL || p_scalar == NULL || p == NULL) {
    OPENSSL_PUT_ERROR(EC, ERR_R_PASSED_NULL_PARAMETER);
    return 0;
  }

  if (group->meth->mul_public == NULL) {
    return group->meth->mul_public_batch(group, r, g_scalar, p, p_scalar, 1);
  }

  group->meth->mul_public(group, r, g_scalar, p, p_scalar);
  return 1;
}

int ec_point_mul_scalar_public_batch(const EC_GROUP *group, EC_RAW_POINT *r,
                                     const EC_SCALAR *g_scalar,
                                     const EC_RAW_POINT *points,
                                     const EC_SCALAR *scalars, size_t num) {
  if (group->meth->mul_public_batch == NULL) {
    OPENSSL_PUT_ERROR(EC, ERR_R_SHOULD_NOT_HAVE_BEEN_CALLED);
    return 0;
  }

  return group->meth->mul_public_batch(group, r, g_scalar, points, scalars,
                                       num);
}

int ec_point_mul_scalar(const EC_GROUP *group, EC_RAW_POINT *r,
                        const EC_RAW_POINT *p, const EC_SCALAR *scalar) {
  if (p == NULL || scalar == NULL) {
    OPENSSL_PUT_ERROR(EC, ERR_R_PASSED_NULL_PARAMETER);
    return 0;
  }

  group->meth->mul(group, r, p, scalar);

  // Check the result is on the curve to defend against fault attacks or bugs.
  // This has negligible cost compared to the multiplication.
  if (!ec_GFp_simple_is_on_curve(group, r)) {
    OPENSSL_PUT_ERROR(EC, ERR_R_INTERNAL_ERROR);
    return 0;
  }

  return 1;
}

int ec_point_mul_scalar_base(const EC_GROUP *group, EC_RAW_POINT *r,
                             const EC_SCALAR *scalar) {
  if (scalar == NULL) {
    OPENSSL_PUT_ERROR(EC, ERR_R_PASSED_NULL_PARAMETER);
    return 0;
  }

  group->meth->mul_base(group, r, scalar);

  // Check the result is on the curve to defend against fault attacks or bugs.
  // This has negligible cost compared to the multiplication.
  if (!ec_GFp_simple_is_on_curve(group, r)) {
    OPENSSL_PUT_ERROR(EC, ERR_R_INTERNAL_ERROR);
    return 0;
  }

  return 1;
}

int ec_point_mul_scalar_batch(const EC_GROUP *group, EC_RAW_POINT *r,
                              const EC_RAW_POINT *p0, const EC_SCALAR *scalar0,
                              const EC_RAW_POINT *p1, const EC_SCALAR *scalar1,
                              const EC_RAW_POINT *p2,
                              const EC_SCALAR *scalar2) {
  if (group->meth->mul_batch == NULL) {
    OPENSSL_PUT_ERROR(EC, ERR_R_SHOULD_NOT_HAVE_BEEN_CALLED);
    return 0;
  }

  group->meth->mul_batch(group, r, p0, scalar0, p1, scalar1, p2, scalar2);

  // Check the result is on the curve to defend against fault attacks or bugs.
  // This has negligible cost compared to the multiplication.
  if (!ec_GFp_simple_is_on_curve(group, r)) {
    OPENSSL_PUT_ERROR(EC, ERR_R_INTERNAL_ERROR);
    return 0;
  }

  return 1;
}

int ec_init_precomp(const EC_GROUP *group, EC_PRECOMP *out,
                    const EC_RAW_POINT *p) {
  if (group->meth->init_precomp == NULL) {
    OPENSSL_PUT_ERROR(EC, ERR_R_SHOULD_NOT_HAVE_BEEN_CALLED);
    return 0;
  }

  return group->meth->init_precomp(group, out, p);
}

int ec_point_mul_scalar_precomp(const EC_GROUP *group, EC_RAW_POINT *r,
                                const EC_PRECOMP *p0, const EC_SCALAR *scalar0,
                                const EC_PRECOMP *p1, const EC_SCALAR *scalar1,
                                const EC_PRECOMP *p2,
                                const EC_SCALAR *scalar2) {
  if (group->meth->mul_precomp == NULL) {
    OPENSSL_PUT_ERROR(EC, ERR_R_SHOULD_NOT_HAVE_BEEN_CALLED);
    return 0;
  }

  group->meth->mul_precomp(group, r, p0, scalar0, p1, scalar1, p2, scalar2);

  // Check the result is on the curve to defend against fault attacks or bugs.
  // This has negligible cost compared to the multiplication.
  if (!ec_GFp_simple_is_on_curve(group, r)) {
    OPENSSL_PUT_ERROR(EC, ERR_R_INTERNAL_ERROR);
    return 0;
  }

  return 1;
}

void ec_point_select(const EC_GROUP *group, EC_RAW_POINT *out, BN_ULONG mask,
                      const EC_RAW_POINT *a, const EC_RAW_POINT *b) {
  ec_felem_select(group, &out->X, mask, &a->X, &b->X);
  ec_felem_select(group, &out->Y, mask, &a->Y, &b->Y);
  ec_felem_select(group, &out->Z, mask, &a->Z, &b->Z);
}

void ec_affine_select(const EC_GROUP *group, EC_AFFINE *out, BN_ULONG mask,
                      const EC_AFFINE *a, const EC_AFFINE *b) {
  ec_felem_select(group, &out->X, mask, &a->X, &b->X);
  ec_felem_select(group, &out->Y, mask, &a->Y, &b->Y);
}

void ec_precomp_select(const EC_GROUP *group, EC_PRECOMP *out, BN_ULONG mask,
                       const EC_PRECOMP *a, const EC_PRECOMP *b) {
  static_assert(sizeof(out->comb) == sizeof(*out),
                "out->comb does not span the entire structure");
  for (size_t i = 0; i < OPENSSL_ARRAY_SIZE(out->comb); i++) {
    ec_affine_select(group, &out->comb[i], mask, &a->comb[i], &b->comb[i]);
  }
}

int ec_cmp_x_coordinate(const EC_GROUP *group, const EC_RAW_POINT *p,
                        const EC_SCALAR *r) {
  return group->meth->cmp_x_coordinate(group, p, r);
}

int ec_get_x_coordinate_as_scalar(const EC_GROUP *group, EC_SCALAR *out,
                                  const EC_RAW_POINT *p) {
  uint8_t bytes[EC_MAX_BYTES];
  size_t len;
  if (!ec_get_x_coordinate_as_bytes(group, bytes, &len, sizeof(bytes), p)) {
    return 0;
  }

  // The x-coordinate is bounded by p, but we need a scalar, bounded by the
  // order. These may not have the same size. However, we must have p < 2×order,
  // assuming p is not tiny (p >= 17).
  //
  // Thus |bytes| will fit in |order.width + 1| words, and we can reduce by
  // performing at most one subtraction.
  //
  // Proof: We only work with prime order curves, so the number of points on
  // the curve is the order. Thus Hasse's theorem gives:
  //
  //     |order - (p + 1)| <= 2×sqrt(p)
  //         p + 1 - order <= 2×sqrt(p)
  //     p + 1 - 2×sqrt(p) <= order
  //       p + 1 - 2×(p/4)  < order       (p/4 > sqrt(p) for p >= 17)
  //         p/2 < p/2 + 1  < order
  //                     p  < 2×order
  //
  // Additionally, one can manually check this property for built-in curves. It
  // is enforced for legacy custom curves in |EC_GROUP_set_generator|.
  const BIGNUM *order = &group->order;
  BN_ULONG words[EC_MAX_WORDS + 1] = {0};
  bn_big_endian_to_words(words, order->width + 1, bytes, len);
  bn_reduce_once(out->words, words, /*carry=*/words[order->width], order->d,
                 order->width);
  return 1;
}

int ec_get_x_coordinate_as_bytes(const EC_GROUP *group, uint8_t *out,
                                 size_t *out_len, size_t max_out,
                                 const EC_RAW_POINT *p) {
  size_t len = BN_num_bytes(&group->field);
  assert(len <= EC_MAX_BYTES);
  if (max_out < len) {
    OPENSSL_PUT_ERROR(EC, EC_R_BUFFER_TOO_SMALL);
    return 0;
  }

  EC_FELEM x;
  if (!group->meth->point_get_affine_coordinates(group, p, &x, NULL)) {
    return 0;
  }

  ec_felem_to_bytes(group, out, out_len, &x);
  *out_len = len;
  return 1;
}

void ec_set_to_safe_point(const EC_GROUP *group, EC_RAW_POINT *out) {
  if (group->generator != NULL) {
    ec_GFp_simple_point_copy(out, &group->generator->raw);
  } else {
    // The generator can be missing if the caller is in the process of
    // constructing an arbitrary group. In this case, we give up and use the
    // point at infinity.
    ec_GFp_simple_point_set_to_infinity(group, out);
  }
}

void EC_GROUP_set_asn1_flag(EC_GROUP *group, int flag) {}

int EC_GROUP_get_asn1_flag(const EC_GROUP *group) {
  return OPENSSL_EC_NAMED_CURVE;
}

const EC_METHOD *EC_GROUP_method_of(const EC_GROUP *group) {
  // This function exists purely to give callers a way to call
  // |EC_METHOD_get_field_type|. cryptography.io crashes if |EC_GROUP_method_of|
  // returns NULL, so return some other garbage pointer.
  return (const EC_METHOD *)0x12340000;
}

int EC_METHOD_get_field_type(const EC_METHOD *meth) {
  return NID_X9_62_prime_field;
}

void EC_GROUP_set_point_conversion_form(EC_GROUP *group,
                                        point_conversion_form_t form) {
  if (form != POINT_CONVERSION_UNCOMPRESSED) {
    abort();
  }
}

size_t EC_get_builtin_curves(EC_builtin_curve *out_curves,
                             size_t max_num_curves) {
  const struct built_in_curves *const curves = OPENSSL_built_in_curves();

  for (size_t i = 0; i < max_num_curves && i < OPENSSL_NUM_BUILT_IN_CURVES;
       i++) {
    out_curves[i].comment = curves->curves[i].comment;
    out_curves[i].nid = curves->curves[i].nid;
  }

  return OPENSSL_NUM_BUILT_IN_CURVES;
}
