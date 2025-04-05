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
 * [including the GNU Public Licence.]
 */
/* ====================================================================
 * Copyright (c) 1998-2001 The OpenSSL Project.  All rights reserved.
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

#include <openssl/bn.h>

#include <openssl/err.h>

#include "internal.h"


int BN_mod_inverse_odd(BIGNUM *out, int *out_no_inverse, const BIGNUM *a,
                       const BIGNUM *n, BN_CTX *ctx) {
  *out_no_inverse = 0;

  if (!BN_is_odd(n)) {
    OPENSSL_PUT_ERROR(BN, BN_R_CALLED_WITH_EVEN_MODULUS);
    return 0;
  }

  if (BN_is_negative(a) || BN_cmp(a, n) >= 0) {
    OPENSSL_PUT_ERROR(BN, BN_R_INPUT_NOT_REDUCED);
    return 0;
  }

  BIGNUM *A, *B, *X, *Y;
  int ret = 0;
  int sign;

  BN_CTX_start(ctx);
  A = BN_CTX_get(ctx);
  B = BN_CTX_get(ctx);
  X = BN_CTX_get(ctx);
  Y = BN_CTX_get(ctx);
  if (Y == NULL) {
    goto err;
  }

  BIGNUM *R = out;

  BN_zero(Y);
  if (!BN_one(X) || BN_copy(B, a) == NULL || BN_copy(A, n) == NULL) {
    goto err;
  }
  A->neg = 0;
  sign = -1;
  // From  B = a mod |n|,  A = |n|  it follows that
  //
  //      0 <= B < A,
  //     -sign*X*a  ==  B   (mod |n|),
  //      sign*Y*a  ==  A   (mod |n|).

  // Binary inversion algorithm; requires odd modulus. This is faster than the
  // general algorithm if the modulus is sufficiently small (about 400 .. 500
  // bits on 32-bit systems, but much more on 64-bit systems)
  int shift;

  while (!BN_is_zero(B)) {
    //      0 < B < |n|,
    //      0 < A <= |n|,
    // (1) -sign*X*a  ==  B   (mod |n|),
    // (2)  sign*Y*a  ==  A   (mod |n|)

    // Now divide  B  by the maximum possible power of two in the integers,
    // and divide  X  by the same value mod |n|.
    // When we're done, (1) still holds.
    shift = 0;
    while (!BN_is_bit_set(B, shift)) {
      // note that 0 < B
      shift++;

      if (BN_is_odd(X)) {
        if (!BN_uadd(X, X, n)) {
          goto err;
        }
      }
      // now X is even, so we can easily divide it by two
      if (!BN_rshift1(X, X)) {
        goto err;
      }
    }
    if (shift > 0) {
      if (!BN_rshift(B, B, shift)) {
        goto err;
      }
    }

    // Same for A and Y. Afterwards, (2) still holds.
    shift = 0;
    while (!BN_is_bit_set(A, shift)) {
      // note that 0 < A
      shift++;

      if (BN_is_odd(Y)) {
        if (!BN_uadd(Y, Y, n)) {
          goto err;
        }
      }
      // now Y is even
      if (!BN_rshift1(Y, Y)) {
        goto err;
      }
    }
    if (shift > 0) {
      if (!BN_rshift(A, A, shift)) {
        goto err;
      }
    }

    // We still have (1) and (2).
    // Both  A  and  B  are odd.
    // The following computations ensure that
    //
    //     0 <= B < |n|,
    //      0 < A < |n|,
    // (1) -sign*X*a  ==  B   (mod |n|),
    // (2)  sign*Y*a  ==  A   (mod |n|),
    //
    // and that either  A  or  B  is even in the next iteration.
    if (BN_ucmp(B, A) >= 0) {
      // -sign*(X + Y)*a == B - A  (mod |n|)
      if (!BN_uadd(X, X, Y)) {
        goto err;
      }
      // NB: we could use BN_mod_add_quick(X, X, Y, n), but that
      // actually makes the algorithm slower
      if (!BN_usub(B, B, A)) {
        goto err;
      }
    } else {
      //  sign*(X + Y)*a == A - B  (mod |n|)
      if (!BN_uadd(Y, Y, X)) {
        goto err;
      }
      // as above, BN_mod_add_quick(Y, Y, X, n) would slow things down
      if (!BN_usub(A, A, B)) {
        goto err;
      }
    }
  }

  if (!BN_is_one(A)) {
    *out_no_inverse = 1;
    OPENSSL_PUT_ERROR(BN, BN_R_NO_INVERSE);
    goto err;
  }

  // The while loop (Euclid's algorithm) ends when
  //      A == gcd(a,n);
  // we have
  //       sign*Y*a  ==  A  (mod |n|),
  // where  Y  is non-negative.

  if (sign < 0) {
    if (!BN_sub(Y, n, Y)) {
      goto err;
    }
  }
  // Now  Y*a  ==  A  (mod |n|).

  // Y*a == 1  (mod |n|)
  if (!Y->neg && BN_ucmp(Y, n) < 0) {
    if (!BN_copy(R, Y)) {
      goto err;
    }
  } else {
    if (!BN_nnmod(R, Y, n, ctx)) {
      goto err;
    }
  }

  ret = 1;

err:
  BN_CTX_end(ctx);
  return ret;
}

BIGNUM *BN_mod_inverse(BIGNUM *out, const BIGNUM *a, const BIGNUM *n,
                       BN_CTX *ctx) {
  BIGNUM *new_out = NULL;
  if (out == NULL) {
    new_out = BN_new();
    if (new_out == NULL) {
      return NULL;
    }
    out = new_out;
  }

  int ok = 0;
  BIGNUM *a_reduced = NULL;
  if (a->neg || BN_ucmp(a, n) >= 0) {
    a_reduced = BN_dup(a);
    if (a_reduced == NULL) {
      goto err;
    }
    if (!BN_nnmod(a_reduced, a_reduced, n, ctx)) {
      goto err;
    }
    a = a_reduced;
  }

  int no_inverse;
  if (!BN_is_odd(n)) {
    if (!bn_mod_inverse_consttime(out, &no_inverse, a, n, ctx)) {
      goto err;
    }
  } else if (!BN_mod_inverse_odd(out, &no_inverse, a, n, ctx)) {
    goto err;
  }

  ok = 1;

err:
  if (!ok) {
    BN_free(new_out);
    out = NULL;
  }
  BN_free(a_reduced);
  return out;
}

int BN_mod_inverse_blinded(BIGNUM *out, int *out_no_inverse, const BIGNUM *a,
                           const BN_MONT_CTX *mont, BN_CTX *ctx) {
  *out_no_inverse = 0;

  if (BN_is_negative(a) || BN_cmp(a, &mont->N) >= 0) {
    OPENSSL_PUT_ERROR(BN, BN_R_INPUT_NOT_REDUCED);
    return 0;
  }

  int ret = 0;
  BIGNUM blinding_factor;
  BN_init(&blinding_factor);

  if (!BN_rand_range_ex(&blinding_factor, 1, &mont->N) ||
      !BN_mod_mul_montgomery(out, &blinding_factor, a, mont, ctx) ||
      !BN_mod_inverse_odd(out, out_no_inverse, out, &mont->N, ctx) ||
      !BN_mod_mul_montgomery(out, &blinding_factor, out, mont, ctx)) {
    OPENSSL_PUT_ERROR(BN, ERR_R_BN_LIB);
    goto err;
  }

  ret = 1;

err:
  BN_free(&blinding_factor);
  return ret;
}

int bn_mod_inverse_prime(BIGNUM *out, const BIGNUM *a, const BIGNUM *p,
                         BN_CTX *ctx, const BN_MONT_CTX *mont_p) {
  BN_CTX_start(ctx);
  BIGNUM *p_minus_2 = BN_CTX_get(ctx);
  int ok = p_minus_2 != NULL &&
           BN_copy(p_minus_2, p) &&
           BN_sub_word(p_minus_2, 2) &&
           BN_mod_exp_mont(out, a, p_minus_2, p, ctx, mont_p);
  BN_CTX_end(ctx);
  return ok;
}

int bn_mod_inverse_secret_prime(BIGNUM *out, const BIGNUM *a, const BIGNUM *p,
                                BN_CTX *ctx, const BN_MONT_CTX *mont_p) {
  BN_CTX_start(ctx);
  BIGNUM *p_minus_2 = BN_CTX_get(ctx);
  int ok = p_minus_2 != NULL &&
           BN_copy(p_minus_2, p) &&
           BN_sub_word(p_minus_2, 2) &&
           BN_mod_exp_mont_consttime(out, a, p_minus_2, p, ctx, mont_p);
  BN_CTX_end(ctx);
  return ok;
}
