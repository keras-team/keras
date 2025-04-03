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

#ifndef OPENSSL_HEADER_DIGEST_INTERNAL_H
#define OPENSSL_HEADER_DIGEST_INTERNAL_H

#include <openssl/base.h>

#if defined(__cplusplus)
extern "C" {
#endif


struct env_md_st {
  // type contains a NID identifing the digest function. (For example,
  // NID_md5.)
  int type;

  // md_size contains the size, in bytes, of the resulting digest.
  unsigned md_size;

  // flags contains the OR of |EVP_MD_FLAG_*| values.
  uint32_t flags;

  // init initialises the state in |ctx->md_data|.
  void (*init)(EVP_MD_CTX *ctx);

  // update hashes |len| bytes of |data| into the state in |ctx->md_data|.
  void (*update)(EVP_MD_CTX *ctx, const void *data, size_t count);

  // final completes the hash and writes |md_size| bytes of digest to |out|.
  void (*final)(EVP_MD_CTX *ctx, uint8_t *out);

  // block_size contains the hash's native block size.
  unsigned block_size;

  // ctx_size contains the size, in bytes, of the state of the hash function.
  unsigned ctx_size;
};

// evp_md_pctx_ops contains function pointers to allow the |pctx| member of
// |EVP_MD_CTX| to be manipulated without breaking layering by calling EVP
// functions.
struct evp_md_pctx_ops {
  // free is called when an |EVP_MD_CTX| is being freed and the |pctx| also
  // needs to be freed.
  void (*free) (EVP_PKEY_CTX *pctx);

  // dup is called when an |EVP_MD_CTX| is copied and so the |pctx| also needs
  // to be copied.
  EVP_PKEY_CTX* (*dup) (EVP_PKEY_CTX *pctx);
};


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_DIGEST_INTERNAL
