/* Written by Dr Stephen N Henson (steve@openssl.org) for the OpenSSL
 * project 2006.
 */
/* ====================================================================
 * Copyright (c) 2006,2007 The OpenSSL Project.  All rights reserved.
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

#include <openssl/evp.h>

#include <openssl/err.h>

#include "../../evp/internal.h"
#include "../delocate.h"
#include "../digest/internal.h"
#include "../service_indicator/internal.h"


enum evp_sign_verify_t {
  evp_sign,
  evp_verify,
};

DEFINE_LOCAL_DATA(struct evp_md_pctx_ops, md_pctx_ops) {
  out->free = EVP_PKEY_CTX_free;
  out->dup = EVP_PKEY_CTX_dup;
};

static int uses_prehash(EVP_MD_CTX *ctx, enum evp_sign_verify_t op) {
  return (op == evp_sign) ? (ctx->pctx->pmeth->sign != NULL)
                          : (ctx->pctx->pmeth->verify != NULL);
}

static int do_sigver_init(EVP_MD_CTX *ctx, EVP_PKEY_CTX **pctx,
                          const EVP_MD *type, ENGINE *e, EVP_PKEY *pkey,
                          enum evp_sign_verify_t op) {
  if (ctx->pctx == NULL) {
    ctx->pctx = EVP_PKEY_CTX_new(pkey, e);
  }
  if (ctx->pctx == NULL) {
    return 0;
  }
  ctx->pctx_ops = md_pctx_ops();

  if (op == evp_verify) {
    if (!EVP_PKEY_verify_init(ctx->pctx)) {
      return 0;
    }
  } else {
    if (!EVP_PKEY_sign_init(ctx->pctx)) {
      return 0;
    }
  }

  if (type != NULL &&
      !EVP_PKEY_CTX_set_signature_md(ctx->pctx, type)) {
    return 0;
  }

  if (uses_prehash(ctx, op)) {
    if (type == NULL) {
      OPENSSL_PUT_ERROR(EVP, EVP_R_NO_DEFAULT_DIGEST);
      return 0;
    }
    if (!EVP_DigestInit_ex(ctx, type, e)) {
      return 0;
    }
  }

  if (pctx) {
    *pctx = ctx->pctx;
  }
  return 1;
}

int EVP_DigestSignInit(EVP_MD_CTX *ctx, EVP_PKEY_CTX **pctx, const EVP_MD *type,
                       ENGINE *e, EVP_PKEY *pkey) {
  return do_sigver_init(ctx, pctx, type, e, pkey, evp_sign);
}

int EVP_DigestVerifyInit(EVP_MD_CTX *ctx, EVP_PKEY_CTX **pctx,
                         const EVP_MD *type, ENGINE *e, EVP_PKEY *pkey) {
  return do_sigver_init(ctx, pctx, type, e, pkey, evp_verify);
}

int EVP_DigestSignUpdate(EVP_MD_CTX *ctx, const void *data, size_t len) {
  if (!uses_prehash(ctx, evp_sign)) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_OPERATION_NOT_SUPPORTED_FOR_THIS_KEYTYPE);
    return 0;
  }

  return EVP_DigestUpdate(ctx, data, len);
}

int EVP_DigestVerifyUpdate(EVP_MD_CTX *ctx, const void *data, size_t len) {
  if (!uses_prehash(ctx, evp_verify)) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_OPERATION_NOT_SUPPORTED_FOR_THIS_KEYTYPE);
    return 0;
  }

  return EVP_DigestUpdate(ctx, data, len);
}

int EVP_DigestSignFinal(EVP_MD_CTX *ctx, uint8_t *out_sig,
                        size_t *out_sig_len) {
  if (!uses_prehash(ctx, evp_sign)) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_OPERATION_NOT_SUPPORTED_FOR_THIS_KEYTYPE);
    return 0;
  }

  if (out_sig) {
    EVP_MD_CTX tmp_ctx;
    int ret;
    uint8_t md[EVP_MAX_MD_SIZE];
    unsigned int mdlen;

    FIPS_service_indicator_lock_state();
    EVP_MD_CTX_init(&tmp_ctx);
    ret = EVP_MD_CTX_copy_ex(&tmp_ctx, ctx) &&
          EVP_DigestFinal_ex(&tmp_ctx, md, &mdlen) &&
          EVP_PKEY_sign(ctx->pctx, out_sig, out_sig_len, md, mdlen);
    EVP_MD_CTX_cleanup(&tmp_ctx);
    FIPS_service_indicator_unlock_state();

    if (ret) {
      EVP_DigestSign_verify_service_indicator(ctx);
    }

    return ret;
  } else {
    size_t s = EVP_MD_size(ctx->digest);
    return EVP_PKEY_sign(ctx->pctx, out_sig, out_sig_len, NULL, s);
  }
}

int EVP_DigestVerifyFinal(EVP_MD_CTX *ctx, const uint8_t *sig,
                          size_t sig_len) {
  if (!uses_prehash(ctx, evp_verify)) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_OPERATION_NOT_SUPPORTED_FOR_THIS_KEYTYPE);
    return 0;
  }

  EVP_MD_CTX tmp_ctx;
  int ret;
  uint8_t md[EVP_MAX_MD_SIZE];
  unsigned int mdlen;

  FIPS_service_indicator_lock_state();
  EVP_MD_CTX_init(&tmp_ctx);
  ret = EVP_MD_CTX_copy_ex(&tmp_ctx, ctx) &&
        EVP_DigestFinal_ex(&tmp_ctx, md, &mdlen) &&
        EVP_PKEY_verify(ctx->pctx, sig, sig_len, md, mdlen);
  FIPS_service_indicator_unlock_state();
  EVP_MD_CTX_cleanup(&tmp_ctx);

  if (ret) {
    EVP_DigestVerify_verify_service_indicator(ctx);
  }

  return ret;
}

int EVP_DigestSign(EVP_MD_CTX *ctx, uint8_t *out_sig, size_t *out_sig_len,
                   const uint8_t *data, size_t data_len) {
  FIPS_service_indicator_lock_state();
  int ret = 0;

  if (uses_prehash(ctx, evp_sign)) {
    // If |out_sig| is NULL, the caller is only querying the maximum output
    // length. |data| should only be incorporated in the final call.
    if (out_sig != NULL &&
        !EVP_DigestSignUpdate(ctx, data, data_len)) {
      goto end;
    }

    ret = EVP_DigestSignFinal(ctx, out_sig, out_sig_len);
    goto end;
  }

  if (ctx->pctx->pmeth->sign_message == NULL) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_OPERATION_NOT_SUPPORTED_FOR_THIS_KEYTYPE);
    goto end;
  }

  ret = ctx->pctx->pmeth->sign_message(ctx->pctx, out_sig, out_sig_len, data,
                                       data_len);

end:
  FIPS_service_indicator_unlock_state();
  if (ret) {
    EVP_DigestSign_verify_service_indicator(ctx);
  }
  return ret;
}

int EVP_DigestVerify(EVP_MD_CTX *ctx, const uint8_t *sig, size_t sig_len,
                     const uint8_t *data, size_t len) {
  FIPS_service_indicator_lock_state();
  int ret = 0;

  if (uses_prehash(ctx, evp_verify)) {
    ret = EVP_DigestVerifyUpdate(ctx, data, len) &&
          EVP_DigestVerifyFinal(ctx, sig, sig_len);
    goto end;
  }

  if (ctx->pctx->pmeth->verify_message == NULL) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_OPERATION_NOT_SUPPORTED_FOR_THIS_KEYTYPE);
    goto end;
  }

  ret = ctx->pctx->pmeth->verify_message(ctx->pctx, sig, sig_len, data, len);

end:
  FIPS_service_indicator_unlock_state();
  if (ret) {
    EVP_DigestVerify_verify_service_indicator(ctx);
  }
  return ret;
}
