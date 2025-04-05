/* ====================================================================
 * Copyright 2002 Sun Microsystems, Inc. ALL RIGHTS RESERVED.
 *
 * The Elliptic Curve Public-Key Crypto Library (ECC Code) included
 * herein is developed by SUN MICROSYSTEMS, INC., and is contributed
 * to the OpenSSL project.
 *
 * The ECC Code is licensed pursuant to the OpenSSL open source
 * license provided below.
 *
 * The ECDH software is originally written by Douglas Stebila of
 * Sun Microsystems Laboratories.
 *
 */
/* ====================================================================
 * Copyright (c) 2000-2002 The OpenSSL Project.  All rights reserved.
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

#include <openssl/ecdh.h>

#include <string.h>

#include <openssl/ec.h>
#include <openssl/ec_key.h>
#include <openssl/err.h>
#include <openssl/mem.h>
#include <openssl/sha.h>

#include "../../internal.h"
#include "../ec/internal.h"
#include "../service_indicator/internal.h"


int ECDH_compute_key_fips(uint8_t *out, size_t out_len, const EC_POINT *pub_key,
                          const EC_KEY *priv_key) {
  boringssl_ensure_ecc_self_test();

  if (priv_key->priv_key == NULL) {
    OPENSSL_PUT_ERROR(ECDH, ECDH_R_NO_PRIVATE_VALUE);
    return 0;
  }
  const EC_SCALAR *const priv = &priv_key->priv_key->scalar;
  const EC_GROUP *const group = EC_KEY_get0_group(priv_key);
  if (EC_GROUP_cmp(group, pub_key->group, NULL) != 0) {
    OPENSSL_PUT_ERROR(EC, EC_R_INCOMPATIBLE_OBJECTS);
    return 0;
  }

  EC_RAW_POINT shared_point;
  uint8_t buf[EC_MAX_BYTES];
  size_t buflen;
  if (!ec_point_mul_scalar(group, &shared_point, &pub_key->raw, priv) ||
      !ec_get_x_coordinate_as_bytes(group, buf, &buflen, sizeof(buf),
                                    &shared_point)) {
    OPENSSL_PUT_ERROR(ECDH, ECDH_R_POINT_ARITHMETIC_FAILURE);
    return 0;
  }

  FIPS_service_indicator_lock_state();
  switch (out_len) {
    case SHA224_DIGEST_LENGTH:
      SHA224(buf, buflen, out);
      break;
    case SHA256_DIGEST_LENGTH:
      SHA256(buf, buflen, out);
      break;
    case SHA384_DIGEST_LENGTH:
      SHA384(buf, buflen, out);
      break;
    case SHA512_DIGEST_LENGTH:
      SHA512(buf, buflen, out);
      break;
    default:
      OPENSSL_PUT_ERROR(ECDH, ECDH_R_UNKNOWN_DIGEST_LENGTH);
      FIPS_service_indicator_unlock_state();
      return 0;
  }
  FIPS_service_indicator_unlock_state();

  ECDH_verify_service_indicator(priv_key);
  return 1;
}
