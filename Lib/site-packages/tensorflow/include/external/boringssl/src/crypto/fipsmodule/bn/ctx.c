/* Written by Ulf Moeller for the OpenSSL project. */
/* ====================================================================
 * Copyright (c) 1998-2004 The OpenSSL Project.  All rights reserved.
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

#include <assert.h>
#include <string.h>

#include <openssl/err.h>
#include <openssl/mem.h>

#include "../../internal.h"


// The stack frame info is resizing, set a first-time expansion size;
#define BN_CTX_START_FRAMES 32


// BN_STACK

// A |BN_STACK| is a stack of |size_t| values.
typedef struct {
  // Array of indexes into |ctx->bignums|.
  size_t *indexes;
  // Number of stack frames, and the size of the allocated array
  size_t depth, size;
} BN_STACK;

static void BN_STACK_init(BN_STACK *);
static void BN_STACK_cleanup(BN_STACK *);
static int BN_STACK_push(BN_STACK *, size_t idx);
static size_t BN_STACK_pop(BN_STACK *);


// BN_CTX

DEFINE_STACK_OF(BIGNUM)

// The opaque BN_CTX type
struct bignum_ctx {
  // bignums is the stack of |BIGNUM|s managed by this |BN_CTX|.
  STACK_OF(BIGNUM) *bignums;
  // stack is the stack of |BN_CTX_start| frames. It is the value of |used| at
  // the time |BN_CTX_start| was called.
  BN_STACK stack;
  // used is the number of |BIGNUM|s from |bignums| that have been used.
  size_t used;
  // error is one if any operation on this |BN_CTX| failed. All subsequent
  // operations will fail.
  char error;
  // defer_error is one if an operation on this |BN_CTX| has failed, but no
  // error has been pushed to the queue yet. This is used to defer errors from
  // |BN_CTX_start| to |BN_CTX_get|.
  char defer_error;
};

BN_CTX *BN_CTX_new(void) {
  BN_CTX *ret = OPENSSL_malloc(sizeof(BN_CTX));
  if (!ret) {
    return NULL;
  }

  // Initialise the structure
  ret->bignums = NULL;
  BN_STACK_init(&ret->stack);
  ret->used = 0;
  ret->error = 0;
  ret->defer_error = 0;
  return ret;
}

void BN_CTX_free(BN_CTX *ctx) {
  if (ctx == NULL) {
    return;
  }

  // All |BN_CTX_start| calls must be matched with |BN_CTX_end|, otherwise the
  // function may use more memory than expected, potentially without bound if
  // done in a loop. Assert that all |BIGNUM|s have been released.
  assert(ctx->used == 0 || ctx->error);
  sk_BIGNUM_pop_free(ctx->bignums, BN_free);
  BN_STACK_cleanup(&ctx->stack);
  OPENSSL_free(ctx);
}

void BN_CTX_start(BN_CTX *ctx) {
  if (ctx->error) {
    // Once an operation has failed, |ctx->stack| no longer matches the number
    // of |BN_CTX_end| calls to come. Do nothing.
    return;
  }

  if (!BN_STACK_push(&ctx->stack, ctx->used)) {
    ctx->error = 1;
    // |BN_CTX_start| cannot fail, so defer the error to |BN_CTX_get|.
    ctx->defer_error = 1;
  }
}

BIGNUM *BN_CTX_get(BN_CTX *ctx) {
  // Once any operation has failed, they all do.
  if (ctx->error) {
    if (ctx->defer_error) {
      OPENSSL_PUT_ERROR(BN, BN_R_TOO_MANY_TEMPORARY_VARIABLES);
      ctx->defer_error = 0;
    }
    return NULL;
  }

  if (ctx->bignums == NULL) {
    ctx->bignums = sk_BIGNUM_new_null();
    if (ctx->bignums == NULL) {
      ctx->error = 1;
      return NULL;
    }
  }

  if (ctx->used == sk_BIGNUM_num(ctx->bignums)) {
    BIGNUM *bn = BN_new();
    if (bn == NULL || !sk_BIGNUM_push(ctx->bignums, bn)) {
      OPENSSL_PUT_ERROR(BN, BN_R_TOO_MANY_TEMPORARY_VARIABLES);
      BN_free(bn);
      ctx->error = 1;
      return NULL;
    }
  }

  BIGNUM *ret = sk_BIGNUM_value(ctx->bignums, ctx->used);
  BN_zero(ret);
  // This is bounded by |sk_BIGNUM_num|, so it cannot overflow.
  ctx->used++;
  return ret;
}

void BN_CTX_end(BN_CTX *ctx) {
  if (ctx->error) {
    // Once an operation has failed, |ctx->stack| no longer matches the number
    // of |BN_CTX_end| calls to come. Do nothing.
    return;
  }

  ctx->used = BN_STACK_pop(&ctx->stack);
}


// BN_STACK

static void BN_STACK_init(BN_STACK *st) {
  st->indexes = NULL;
  st->depth = st->size = 0;
}

static void BN_STACK_cleanup(BN_STACK *st) {
  OPENSSL_free(st->indexes);
}

static int BN_STACK_push(BN_STACK *st, size_t idx) {
  if (st->depth == st->size) {
    // This function intentionally does not push to the error queue on error.
    // Error-reporting is deferred to |BN_CTX_get|.
    size_t new_size = st->size != 0 ? st->size * 3 / 2 : BN_CTX_START_FRAMES;
    if (new_size <= st->size || new_size > ((size_t)-1) / sizeof(size_t)) {
      return 0;
    }
    size_t *new_indexes =
        OPENSSL_realloc(st->indexes, new_size * sizeof(size_t));
    if (new_indexes == NULL) {
      return 0;
    }
    st->indexes = new_indexes;
    st->size = new_size;
  }

  st->indexes[st->depth] = idx;
  st->depth++;
  return 1;
}

static size_t BN_STACK_pop(BN_STACK *st) {
  assert(st->depth > 0);
  st->depth--;
  return st->indexes[st->depth];
}
