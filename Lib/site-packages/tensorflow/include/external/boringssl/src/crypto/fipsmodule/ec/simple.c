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

#include <string.h>

#include <openssl/bn.h>
#include <openssl/err.h>
#include <openssl/mem.h>

#include "internal.h"
#include "../../internal.h"


// Most method functions in this file are designed to work with non-trivial
// representations of field elements if necessary (see ecp_mont.c): while
// standard modular addition and subtraction are used, the field_mul and
// field_sqr methods will be used for multiplication, and field_encode and
// field_decode (if defined) will be used for converting between
// representations.
//
// Functions here specifically assume that if a non-trivial representation is
// used, it is a Montgomery representation (i.e. 'encoding' means multiplying
// by some factor R).

int ec_GFp_simple_group_init(EC_GROUP *group) {
  BN_init(&group->field);
  group->a_is_minus3 = 0;
  return 1;
}

void ec_GFp_simple_group_finish(EC_GROUP *group) {
  BN_free(&group->field);
}

int ec_GFp_simple_group_set_curve(EC_GROUP *group, const BIGNUM *p,
                                  const BIGNUM *a, const BIGNUM *b,
                                  BN_CTX *ctx) {
  // p must be a prime > 3
  if (BN_num_bits(p) <= 2 || !BN_is_odd(p)) {
    OPENSSL_PUT_ERROR(EC, EC_R_INVALID_FIELD);
    return 0;
  }

  int ret = 0;
  BN_CTX_start(ctx);
  BIGNUM *tmp = BN_CTX_get(ctx);
  if (tmp == NULL) {
    goto err;
  }

  // group->field
  if (!BN_copy(&group->field, p)) {
    goto err;
  }
  BN_set_negative(&group->field, 0);
  // Store the field in minimal form, so it can be used with |BN_ULONG| arrays.
  bn_set_minimal_width(&group->field);

  if (!ec_bignum_to_felem(group, &group->a, a) ||
      !ec_bignum_to_felem(group, &group->b, b) ||
      !ec_bignum_to_felem(group, &group->one, BN_value_one())) {
    goto err;
  }

  // group->a_is_minus3
  if (!BN_copy(tmp, a) ||
      !BN_add_word(tmp, 3)) {
    goto err;
  }
  group->a_is_minus3 = (0 == BN_cmp(tmp, &group->field));

  ret = 1;

err:
  BN_CTX_end(ctx);
  return ret;
}

int ec_GFp_simple_group_get_curve(const EC_GROUP *group, BIGNUM *p, BIGNUM *a,
                                  BIGNUM *b) {
  if ((p != NULL && !BN_copy(p, &group->field)) ||
      (a != NULL && !ec_felem_to_bignum(group, a, &group->a)) ||
      (b != NULL && !ec_felem_to_bignum(group, b, &group->b))) {
    return 0;
  }
  return 1;
}

void ec_GFp_simple_point_init(EC_RAW_POINT *point) {
  OPENSSL_memset(&point->X, 0, sizeof(EC_FELEM));
  OPENSSL_memset(&point->Y, 0, sizeof(EC_FELEM));
  OPENSSL_memset(&point->Z, 0, sizeof(EC_FELEM));
}

void ec_GFp_simple_point_copy(EC_RAW_POINT *dest, const EC_RAW_POINT *src) {
  OPENSSL_memcpy(&dest->X, &src->X, sizeof(EC_FELEM));
  OPENSSL_memcpy(&dest->Y, &src->Y, sizeof(EC_FELEM));
  OPENSSL_memcpy(&dest->Z, &src->Z, sizeof(EC_FELEM));
}

void ec_GFp_simple_point_set_to_infinity(const EC_GROUP *group,
                                         EC_RAW_POINT *point) {
  // Although it is strictly only necessary to zero Z, we zero the entire point
  // in case |point| was stack-allocated and yet to be initialized.
  ec_GFp_simple_point_init(point);
}

void ec_GFp_simple_invert(const EC_GROUP *group, EC_RAW_POINT *point) {
  ec_felem_neg(group, &point->Y, &point->Y);
}

int ec_GFp_simple_is_at_infinity(const EC_GROUP *group,
                                 const EC_RAW_POINT *point) {
  return ec_felem_non_zero_mask(group, &point->Z) == 0;
}

int ec_GFp_simple_is_on_curve(const EC_GROUP *group,
                              const EC_RAW_POINT *point) {
  // We have a curve defined by a Weierstrass equation
  //      y^2 = x^3 + a*x + b.
  // The point to consider is given in Jacobian projective coordinates
  // where  (X, Y, Z)  represents  (x, y) = (X/Z^2, Y/Z^3).
  // Substituting this and multiplying by  Z^6  transforms the above equation
  // into
  //      Y^2 = X^3 + a*X*Z^4 + b*Z^6.
  // To test this, we add up the right-hand side in 'rh'.
  //
  // This function may be used when double-checking the secret result of a point
  // multiplication, so we proceed in constant-time.

  void (*const felem_mul)(const EC_GROUP *, EC_FELEM *r, const EC_FELEM *a,
                          const EC_FELEM *b) = group->meth->felem_mul;
  void (*const felem_sqr)(const EC_GROUP *, EC_FELEM *r, const EC_FELEM *a) =
      group->meth->felem_sqr;

  // rh := X^2
  EC_FELEM rh;
  felem_sqr(group, &rh, &point->X);

  EC_FELEM tmp, Z4, Z6;
  felem_sqr(group, &tmp, &point->Z);
  felem_sqr(group, &Z4, &tmp);
  felem_mul(group, &Z6, &Z4, &tmp);

  // rh := rh + a*Z^4
  if (group->a_is_minus3) {
    ec_felem_add(group, &tmp, &Z4, &Z4);
    ec_felem_add(group, &tmp, &tmp, &Z4);
    ec_felem_sub(group, &rh, &rh, &tmp);
  } else {
    felem_mul(group, &tmp, &Z4, &group->a);
    ec_felem_add(group, &rh, &rh, &tmp);
  }

  // rh := (rh + a*Z^4)*X
  felem_mul(group, &rh, &rh, &point->X);

  // rh := rh + b*Z^6
  felem_mul(group, &tmp, &group->b, &Z6);
  ec_felem_add(group, &rh, &rh, &tmp);

  // 'lh' := Y^2
  felem_sqr(group, &tmp, &point->Y);

  ec_felem_sub(group, &tmp, &tmp, &rh);
  BN_ULONG not_equal = ec_felem_non_zero_mask(group, &tmp);

  // If Z = 0, the point is infinity, which is always on the curve.
  BN_ULONG not_infinity = ec_felem_non_zero_mask(group, &point->Z);

  return 1 & ~(not_infinity & not_equal);
}

int ec_GFp_simple_points_equal(const EC_GROUP *group, const EC_RAW_POINT *a,
                               const EC_RAW_POINT *b) {
  // This function is implemented in constant-time for two reasons. First,
  // although EC points are usually public, their Jacobian Z coordinates may be
  // secret, or at least are not obviously public. Second, more complex
  // protocols will sometimes manipulate secret points.
  //
  // This does mean that we pay a 6M+2S Jacobian comparison when comparing two
  // publicly affine points costs no field operations at all. If needed, we can
  // restore this optimization by keeping better track of affine vs. Jacobian
  // forms. See https://crbug.com/boringssl/326.

  // If neither |a| or |b| is infinity, we have to decide whether
  //     (X_a/Z_a^2, Y_a/Z_a^3) = (X_b/Z_b^2, Y_b/Z_b^3),
  // or equivalently, whether
  //     (X_a*Z_b^2, Y_a*Z_b^3) = (X_b*Z_a^2, Y_b*Z_a^3).

  void (*const felem_mul)(const EC_GROUP *, EC_FELEM *r, const EC_FELEM *a,
                          const EC_FELEM *b) = group->meth->felem_mul;
  void (*const felem_sqr)(const EC_GROUP *, EC_FELEM *r, const EC_FELEM *a) =
      group->meth->felem_sqr;

  EC_FELEM tmp1, tmp2, Za23, Zb23;
  felem_sqr(group, &Zb23, &b->Z);         // Zb23 = Z_b^2
  felem_mul(group, &tmp1, &a->X, &Zb23);  // tmp1 = X_a * Z_b^2
  felem_sqr(group, &Za23, &a->Z);         // Za23 = Z_a^2
  felem_mul(group, &tmp2, &b->X, &Za23);  // tmp2 = X_b * Z_a^2
  ec_felem_sub(group, &tmp1, &tmp1, &tmp2);
  const BN_ULONG x_not_equal = ec_felem_non_zero_mask(group, &tmp1);

  felem_mul(group, &Zb23, &Zb23, &b->Z);  // Zb23 = Z_b^3
  felem_mul(group, &tmp1, &a->Y, &Zb23);  // tmp1 = Y_a * Z_b^3
  felem_mul(group, &Za23, &Za23, &a->Z);  // Za23 = Z_a^3
  felem_mul(group, &tmp2, &b->Y, &Za23);  // tmp2 = Y_b * Z_a^3
  ec_felem_sub(group, &tmp1, &tmp1, &tmp2);
  const BN_ULONG y_not_equal = ec_felem_non_zero_mask(group, &tmp1);
  const BN_ULONG x_and_y_equal = ~(x_not_equal | y_not_equal);

  const BN_ULONG a_not_infinity = ec_felem_non_zero_mask(group, &a->Z);
  const BN_ULONG b_not_infinity = ec_felem_non_zero_mask(group, &b->Z);
  const BN_ULONG a_and_b_infinity = ~(a_not_infinity | b_not_infinity);

  const BN_ULONG equal =
      a_and_b_infinity | (a_not_infinity & b_not_infinity & x_and_y_equal);
  return equal & 1;
}

int ec_affine_jacobian_equal(const EC_GROUP *group, const EC_AFFINE *a,
                             const EC_RAW_POINT *b) {
  // If |b| is not infinity, we have to decide whether
  //     (X_a, Y_a) = (X_b/Z_b^2, Y_b/Z_b^3),
  // or equivalently, whether
  //     (X_a*Z_b^2, Y_a*Z_b^3) = (X_b, Y_b).

  void (*const felem_mul)(const EC_GROUP *, EC_FELEM *r, const EC_FELEM *a,
                          const EC_FELEM *b) = group->meth->felem_mul;
  void (*const felem_sqr)(const EC_GROUP *, EC_FELEM *r, const EC_FELEM *a) =
      group->meth->felem_sqr;

  EC_FELEM tmp, Zb2;
  felem_sqr(group, &Zb2, &b->Z);        // Zb2 = Z_b^2
  felem_mul(group, &tmp, &a->X, &Zb2);  // tmp = X_a * Z_b^2
  ec_felem_sub(group, &tmp, &tmp, &b->X);
  const BN_ULONG x_not_equal = ec_felem_non_zero_mask(group, &tmp);

  felem_mul(group, &tmp, &a->Y, &Zb2);  // tmp = Y_a * Z_b^2
  felem_mul(group, &tmp, &tmp, &b->Z);  // tmp = Y_a * Z_b^3
  ec_felem_sub(group, &tmp, &tmp, &b->Y);
  const BN_ULONG y_not_equal = ec_felem_non_zero_mask(group, &tmp);
  const BN_ULONG x_and_y_equal = ~(x_not_equal | y_not_equal);

  const BN_ULONG b_not_infinity = ec_felem_non_zero_mask(group, &b->Z);

  const BN_ULONG equal = b_not_infinity & x_and_y_equal;
  return equal & 1;
}

int ec_GFp_simple_cmp_x_coordinate(const EC_GROUP *group, const EC_RAW_POINT *p,
                                   const EC_SCALAR *r) {
  if (ec_GFp_simple_is_at_infinity(group, p)) {
    // |ec_get_x_coordinate_as_scalar| will check this internally, but this way
    // we do not push to the error queue.
    return 0;
  }

  EC_SCALAR x;
  return ec_get_x_coordinate_as_scalar(group, &x, p) &&
         ec_scalar_equal_vartime(group, &x, r);
}

void ec_GFp_simple_felem_to_bytes(const EC_GROUP *group, uint8_t *out,
                                  size_t *out_len, const EC_FELEM *in) {
  size_t len = BN_num_bytes(&group->field);
  bn_words_to_big_endian(out, len, in->words, group->field.width);
  *out_len = len;
}

int ec_GFp_simple_felem_from_bytes(const EC_GROUP *group, EC_FELEM *out,
                                   const uint8_t *in, size_t len) {
  if (len != BN_num_bytes(&group->field)) {
    OPENSSL_PUT_ERROR(EC, EC_R_DECODE_ERROR);
    return 0;
  }

  bn_big_endian_to_words(out->words, group->field.width, in, len);

  if (!bn_less_than_words(out->words, group->field.d, group->field.width)) {
    OPENSSL_PUT_ERROR(EC, EC_R_DECODE_ERROR);
    return 0;
  }

  return 1;
}
