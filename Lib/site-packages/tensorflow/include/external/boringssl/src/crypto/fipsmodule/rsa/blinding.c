/* ====================================================================
 * Copyright (c) 1998-2006 The OpenSSL Project.  All rights reserved.
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
 * Copyright (C) 1995-1998 Eric Young (eay@cryptsoft.com)
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

#include <openssl/rsa.h>

#include <string.h>

#include <openssl/bn.h>
#include <openssl/mem.h>
#include <openssl/err.h>

#include "internal.h"
#include "../../internal.h"


#define BN_BLINDING_COUNTER 32

struct bn_blinding_st {
  BIGNUM *A;  // The base blinding factor, Montgomery-encoded.
  BIGNUM *Ai;  // The inverse of the blinding factor, Montgomery-encoded.
  unsigned counter;
};

static int bn_blinding_create_param(BN_BLINDING *b, const BIGNUM *e,
                                    const BN_MONT_CTX *mont, BN_CTX *ctx);

BN_BLINDING *BN_BLINDING_new(void) {
  BN_BLINDING *ret = OPENSSL_malloc(sizeof(BN_BLINDING));
  if (ret == NULL) {
    return NULL;
  }
  OPENSSL_memset(ret, 0, sizeof(BN_BLINDING));

  ret->A = BN_new();
  if (ret->A == NULL) {
    goto err;
  }

  ret->Ai = BN_new();
  if (ret->Ai == NULL) {
    goto err;
  }

  // The blinding values need to be created before this blinding can be used.
  ret->counter = BN_BLINDING_COUNTER - 1;

  return ret;

err:
  BN_BLINDING_free(ret);
  return NULL;
}

void BN_BLINDING_free(BN_BLINDING *r) {
  if (r == NULL) {
    return;
  }

  BN_free(r->A);
  BN_free(r->Ai);
  OPENSSL_free(r);
}

void BN_BLINDING_invalidate(BN_BLINDING *b) {
  b->counter = BN_BLINDING_COUNTER - 1;
}

static int bn_blinding_update(BN_BLINDING *b, const BIGNUM *e,
                              const BN_MONT_CTX *mont, BN_CTX *ctx) {
  if (++b->counter == BN_BLINDING_COUNTER) {
    // re-create blinding parameters
    if (!bn_blinding_create_param(b, e, mont, ctx)) {
      goto err;
    }
    b->counter = 0;
  } else {
    if (!BN_mod_mul_montgomery(b->A, b->A, b->A, mont, ctx) ||
        !BN_mod_mul_montgomery(b->Ai, b->Ai, b->Ai, mont, ctx)) {
      goto err;
    }
  }

  return 1;

err:
  // |A| and |Ai| may be in an inconsistent state so they both need to be
  // replaced the next time this blinding is used. Note that this is only
  // sufficient because support for |BN_BLINDING_NO_UPDATE| and
  // |BN_BLINDING_NO_RECREATE| was previously dropped.
  b->counter = BN_BLINDING_COUNTER - 1;

  return 0;
}

int BN_BLINDING_convert(BIGNUM *n, BN_BLINDING *b, const BIGNUM *e,
                        const BN_MONT_CTX *mont, BN_CTX *ctx) {
  // |n| is not Montgomery-encoded and |b->A| is. |BN_mod_mul_montgomery|
  // cancels one Montgomery factor, so the resulting value of |n| is unencoded.
  if (!bn_blinding_update(b, e, mont, ctx) ||
      !BN_mod_mul_montgomery(n, n, b->A, mont, ctx)) {
    return 0;
  }

  return 1;
}

int BN_BLINDING_invert(BIGNUM *n, const BN_BLINDING *b, BN_MONT_CTX *mont,
                       BN_CTX *ctx) {
  // |n| is not Montgomery-encoded and |b->A| is. |BN_mod_mul_montgomery|
  // cancels one Montgomery factor, so the resulting value of |n| is unencoded.
  return BN_mod_mul_montgomery(n, n, b->Ai, mont, ctx);
}

static int bn_blinding_create_param(BN_BLINDING *b, const BIGNUM *e,
                                    const BN_MONT_CTX *mont, BN_CTX *ctx) {
  int no_inverse;
  if (!BN_rand_range_ex(b->A, 1, &mont->N) ||
      // Compute |b->A|^-1 in Montgomery form. Note |BN_from_montgomery| +
      // |BN_mod_inverse_blinded| is equivalent to, but more efficient than,
      // |BN_mod_inverse_blinded| + |BN_to_montgomery|.
      //
      // We do not retry if |b->A| has no inverse. Finding a non-invertible
      // value of |b->A| is equivalent to factoring |mont->N|. There is
      // negligible probability of stumbling on one at random.
      !BN_from_montgomery(b->Ai, b->A, mont, ctx) ||
      !BN_mod_inverse_blinded(b->Ai, &no_inverse, b->Ai, mont, ctx) ||
      // TODO(davidben): |BN_mod_exp_mont| internally computes the result in
      // Montgomery form. Save a pair of Montgomery reductions and a
      // multiplication by returning that value directly.
      !BN_mod_exp_mont(b->A, b->A, e, &mont->N, ctx, mont) ||
      !BN_to_montgomery(b->A, b->A, mont, ctx)) {
    OPENSSL_PUT_ERROR(RSA, ERR_R_INTERNAL_ERROR);
    return 0;
  }

  return 1;
}
