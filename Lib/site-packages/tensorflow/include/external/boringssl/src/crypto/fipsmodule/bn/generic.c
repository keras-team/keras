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

#include <openssl/bn.h>

#include <assert.h>

#include "internal.h"


#if !defined(OPENSSL_NO_ASM) && defined(OPENSSL_X86)
// See asm/bn-586.pl.
#define BN_ADD_ASM
#define BN_MUL_ASM
#endif

#if !defined(OPENSSL_NO_ASM) && defined(OPENSSL_X86_64) && \
    (defined(__GNUC__) || defined(__clang__))
// See asm/x86_64-gcc.c
#define BN_ADD_ASM
#define BN_MUL_ASM
#endif

#if !defined(OPENSSL_NO_ASM) && defined(OPENSSL_AARCH64)
// See asm/bn-armv8.pl.
#define BN_ADD_ASM
#endif

#if !defined(BN_MUL_ASM)

#ifdef BN_ULLONG
#define mul_add(r, a, w, c)               \
  do {                                    \
    BN_ULLONG t;                          \
    t = (BN_ULLONG)(w) * (a) + (r) + (c); \
    (r) = Lw(t);                          \
    (c) = Hw(t);                          \
  } while (0)

#define mul(r, a, w, c)             \
  do {                              \
    BN_ULLONG t;                    \
    t = (BN_ULLONG)(w) * (a) + (c); \
    (r) = Lw(t);                    \
    (c) = Hw(t);                    \
  } while (0)

#define sqr(r0, r1, a)        \
  do {                        \
    BN_ULLONG t;              \
    t = (BN_ULLONG)(a) * (a); \
    (r0) = Lw(t);             \
    (r1) = Hw(t);             \
  } while (0)

#else

#define mul_add(r, a, w, c)             \
  do {                                  \
    BN_ULONG high, low, ret, tmp = (a); \
    ret = (r);                          \
    BN_UMULT_LOHI(low, high, w, tmp);   \
    ret += (c);                         \
    (c) = (ret < (c)) ? 1 : 0;          \
    (c) += high;                        \
    ret += low;                         \
    (c) += (ret < low) ? 1 : 0;         \
    (r) = ret;                          \
  } while (0)

#define mul(r, a, w, c)                \
  do {                                 \
    BN_ULONG high, low, ret, ta = (a); \
    BN_UMULT_LOHI(low, high, w, ta);   \
    ret = low + (c);                   \
    (c) = high;                        \
    (c) += (ret < low) ? 1 : 0;        \
    (r) = ret;                         \
  } while (0)

#define sqr(r0, r1, a)               \
  do {                               \
    BN_ULONG tmp = (a);              \
    BN_UMULT_LOHI(r0, r1, tmp, tmp); \
  } while (0)

#endif  // !BN_ULLONG

BN_ULONG bn_mul_add_words(BN_ULONG *rp, const BN_ULONG *ap, size_t num,
                          BN_ULONG w) {
  BN_ULONG c1 = 0;

  if (num == 0) {
    return c1;
  }

  while (num & ~3) {
    mul_add(rp[0], ap[0], w, c1);
    mul_add(rp[1], ap[1], w, c1);
    mul_add(rp[2], ap[2], w, c1);
    mul_add(rp[3], ap[3], w, c1);
    ap += 4;
    rp += 4;
    num -= 4;
  }

  while (num) {
    mul_add(rp[0], ap[0], w, c1);
    ap++;
    rp++;
    num--;
  }

  return c1;
}

BN_ULONG bn_mul_words(BN_ULONG *rp, const BN_ULONG *ap, size_t num,
                      BN_ULONG w) {
  BN_ULONG c1 = 0;

  if (num == 0) {
    return c1;
  }

  while (num & ~3) {
    mul(rp[0], ap[0], w, c1);
    mul(rp[1], ap[1], w, c1);
    mul(rp[2], ap[2], w, c1);
    mul(rp[3], ap[3], w, c1);
    ap += 4;
    rp += 4;
    num -= 4;
  }
  while (num) {
    mul(rp[0], ap[0], w, c1);
    ap++;
    rp++;
    num--;
  }
  return c1;
}

void bn_sqr_words(BN_ULONG *r, const BN_ULONG *a, size_t n) {
  if (n == 0) {
    return;
  }

  while (n & ~3) {
    sqr(r[0], r[1], a[0]);
    sqr(r[2], r[3], a[1]);
    sqr(r[4], r[5], a[2]);
    sqr(r[6], r[7], a[3]);
    a += 4;
    r += 8;
    n -= 4;
  }
  while (n) {
    sqr(r[0], r[1], a[0]);
    a++;
    r += 2;
    n--;
  }
}

// mul_add_c(a,b,c0,c1,c2)  -- c+=a*b for three word number c=(c2,c1,c0)
// mul_add_c2(a,b,c0,c1,c2) -- c+=2*a*b for three word number c=(c2,c1,c0)
// sqr_add_c(a,i,c0,c1,c2)  -- c+=a[i]^2 for three word number c=(c2,c1,c0)
// sqr_add_c2(a,i,c0,c1,c2) -- c+=2*a[i]*a[j] for three word number c=(c2,c1,c0)

#ifdef BN_ULLONG

// Keep in mind that additions to multiplication result can not overflow,
// because its high half cannot be all-ones.
#define mul_add_c(a, b, c0, c1, c2)     \
  do {                                  \
    BN_ULONG hi;                        \
    BN_ULLONG t = (BN_ULLONG)(a) * (b); \
    t += (c0); /* no carry */           \
    (c0) = (BN_ULONG)Lw(t);             \
    hi = (BN_ULONG)Hw(t);               \
    (c1) += (hi);                       \
    (c2) += (c1) < hi;                  \
  } while (0)

#define mul_add_c2(a, b, c0, c1, c2)        \
  do {                                      \
    BN_ULONG hi;                            \
    BN_ULLONG t = (BN_ULLONG)(a) * (b);     \
    BN_ULLONG tt = t + (c0); /* no carry */ \
    (c0) = (BN_ULONG)Lw(tt);                \
    hi = (BN_ULONG)Hw(tt);                  \
    (c1) += hi;                             \
    (c2) += (c1) < hi;                      \
    t += (c0); /* no carry */               \
    (c0) = (BN_ULONG)Lw(t);                 \
    hi = (BN_ULONG)Hw(t);                   \
    (c1) += hi;                             \
    (c2) += (c1) < hi;                      \
  } while (0)

#define sqr_add_c(a, i, c0, c1, c2)           \
  do {                                        \
    BN_ULONG hi;                              \
    BN_ULLONG t = (BN_ULLONG)(a)[i] * (a)[i]; \
    t += (c0); /* no carry */                 \
    (c0) = (BN_ULONG)Lw(t);                   \
    hi = (BN_ULONG)Hw(t);                     \
    (c1) += hi;                               \
    (c2) += (c1) < hi;                        \
  } while (0)

#define sqr_add_c2(a, i, j, c0, c1, c2) mul_add_c2((a)[i], (a)[j], c0, c1, c2)

#else

// Keep in mind that additions to hi can not overflow, because the high word of
// a multiplication result cannot be all-ones.
#define mul_add_c(a, b, c0, c1, c2) \
  do {                              \
    BN_ULONG ta = (a), tb = (b);    \
    BN_ULONG lo, hi;                \
    BN_UMULT_LOHI(lo, hi, ta, tb);  \
    (c0) += lo;                     \
    hi += ((c0) < lo) ? 1 : 0;      \
    (c1) += hi;                     \
    (c2) += ((c1) < hi) ? 1 : 0;    \
  } while (0)

#define mul_add_c2(a, b, c0, c1, c2) \
  do {                               \
    BN_ULONG ta = (a), tb = (b);     \
    BN_ULONG lo, hi, tt;             \
    BN_UMULT_LOHI(lo, hi, ta, tb);   \
    (c0) += lo;                      \
    tt = hi + (((c0) < lo) ? 1 : 0); \
    (c1) += tt;                      \
    (c2) += ((c1) < tt) ? 1 : 0;     \
    (c0) += lo;                      \
    hi += (c0 < lo) ? 1 : 0;         \
    (c1) += hi;                      \
    (c2) += ((c1) < hi) ? 1 : 0;     \
  } while (0)

#define sqr_add_c(a, i, c0, c1, c2) \
  do {                              \
    BN_ULONG ta = (a)[i];           \
    BN_ULONG lo, hi;                \
    BN_UMULT_LOHI(lo, hi, ta, ta);  \
    (c0) += lo;                     \
    hi += (c0 < lo) ? 1 : 0;        \
    (c1) += hi;                     \
    (c2) += ((c1) < hi) ? 1 : 0;    \
  } while (0)

#define sqr_add_c2(a, i, j, c0, c1, c2) mul_add_c2((a)[i], (a)[j], c0, c1, c2)

#endif  // !BN_ULLONG

void bn_mul_comba8(BN_ULONG r[16], const BN_ULONG a[8], const BN_ULONG b[8]) {
  BN_ULONG c1, c2, c3;

  c1 = 0;
  c2 = 0;
  c3 = 0;
  mul_add_c(a[0], b[0], c1, c2, c3);
  r[0] = c1;
  c1 = 0;
  mul_add_c(a[0], b[1], c2, c3, c1);
  mul_add_c(a[1], b[0], c2, c3, c1);
  r[1] = c2;
  c2 = 0;
  mul_add_c(a[2], b[0], c3, c1, c2);
  mul_add_c(a[1], b[1], c3, c1, c2);
  mul_add_c(a[0], b[2], c3, c1, c2);
  r[2] = c3;
  c3 = 0;
  mul_add_c(a[0], b[3], c1, c2, c3);
  mul_add_c(a[1], b[2], c1, c2, c3);
  mul_add_c(a[2], b[1], c1, c2, c3);
  mul_add_c(a[3], b[0], c1, c2, c3);
  r[3] = c1;
  c1 = 0;
  mul_add_c(a[4], b[0], c2, c3, c1);
  mul_add_c(a[3], b[1], c2, c3, c1);
  mul_add_c(a[2], b[2], c2, c3, c1);
  mul_add_c(a[1], b[3], c2, c3, c1);
  mul_add_c(a[0], b[4], c2, c3, c1);
  r[4] = c2;
  c2 = 0;
  mul_add_c(a[0], b[5], c3, c1, c2);
  mul_add_c(a[1], b[4], c3, c1, c2);
  mul_add_c(a[2], b[3], c3, c1, c2);
  mul_add_c(a[3], b[2], c3, c1, c2);
  mul_add_c(a[4], b[1], c3, c1, c2);
  mul_add_c(a[5], b[0], c3, c1, c2);
  r[5] = c3;
  c3 = 0;
  mul_add_c(a[6], b[0], c1, c2, c3);
  mul_add_c(a[5], b[1], c1, c2, c3);
  mul_add_c(a[4], b[2], c1, c2, c3);
  mul_add_c(a[3], b[3], c1, c2, c3);
  mul_add_c(a[2], b[4], c1, c2, c3);
  mul_add_c(a[1], b[5], c1, c2, c3);
  mul_add_c(a[0], b[6], c1, c2, c3);
  r[6] = c1;
  c1 = 0;
  mul_add_c(a[0], b[7], c2, c3, c1);
  mul_add_c(a[1], b[6], c2, c3, c1);
  mul_add_c(a[2], b[5], c2, c3, c1);
  mul_add_c(a[3], b[4], c2, c3, c1);
  mul_add_c(a[4], b[3], c2, c3, c1);
  mul_add_c(a[5], b[2], c2, c3, c1);
  mul_add_c(a[6], b[1], c2, c3, c1);
  mul_add_c(a[7], b[0], c2, c3, c1);
  r[7] = c2;
  c2 = 0;
  mul_add_c(a[7], b[1], c3, c1, c2);
  mul_add_c(a[6], b[2], c3, c1, c2);
  mul_add_c(a[5], b[3], c3, c1, c2);
  mul_add_c(a[4], b[4], c3, c1, c2);
  mul_add_c(a[3], b[5], c3, c1, c2);
  mul_add_c(a[2], b[6], c3, c1, c2);
  mul_add_c(a[1], b[7], c3, c1, c2);
  r[8] = c3;
  c3 = 0;
  mul_add_c(a[2], b[7], c1, c2, c3);
  mul_add_c(a[3], b[6], c1, c2, c3);
  mul_add_c(a[4], b[5], c1, c2, c3);
  mul_add_c(a[5], b[4], c1, c2, c3);
  mul_add_c(a[6], b[3], c1, c2, c3);
  mul_add_c(a[7], b[2], c1, c2, c3);
  r[9] = c1;
  c1 = 0;
  mul_add_c(a[7], b[3], c2, c3, c1);
  mul_add_c(a[6], b[4], c2, c3, c1);
  mul_add_c(a[5], b[5], c2, c3, c1);
  mul_add_c(a[4], b[6], c2, c3, c1);
  mul_add_c(a[3], b[7], c2, c3, c1);
  r[10] = c2;
  c2 = 0;
  mul_add_c(a[4], b[7], c3, c1, c2);
  mul_add_c(a[5], b[6], c3, c1, c2);
  mul_add_c(a[6], b[5], c3, c1, c2);
  mul_add_c(a[7], b[4], c3, c1, c2);
  r[11] = c3;
  c3 = 0;
  mul_add_c(a[7], b[5], c1, c2, c3);
  mul_add_c(a[6], b[6], c1, c2, c3);
  mul_add_c(a[5], b[7], c1, c2, c3);
  r[12] = c1;
  c1 = 0;
  mul_add_c(a[6], b[7], c2, c3, c1);
  mul_add_c(a[7], b[6], c2, c3, c1);
  r[13] = c2;
  c2 = 0;
  mul_add_c(a[7], b[7], c3, c1, c2);
  r[14] = c3;
  r[15] = c1;
}

void bn_mul_comba4(BN_ULONG r[8], const BN_ULONG a[4], const BN_ULONG b[4]) {
  BN_ULONG c1, c2, c3;

  c1 = 0;
  c2 = 0;
  c3 = 0;
  mul_add_c(a[0], b[0], c1, c2, c3);
  r[0] = c1;
  c1 = 0;
  mul_add_c(a[0], b[1], c2, c3, c1);
  mul_add_c(a[1], b[0], c2, c3, c1);
  r[1] = c2;
  c2 = 0;
  mul_add_c(a[2], b[0], c3, c1, c2);
  mul_add_c(a[1], b[1], c3, c1, c2);
  mul_add_c(a[0], b[2], c3, c1, c2);
  r[2] = c3;
  c3 = 0;
  mul_add_c(a[0], b[3], c1, c2, c3);
  mul_add_c(a[1], b[2], c1, c2, c3);
  mul_add_c(a[2], b[1], c1, c2, c3);
  mul_add_c(a[3], b[0], c1, c2, c3);
  r[3] = c1;
  c1 = 0;
  mul_add_c(a[3], b[1], c2, c3, c1);
  mul_add_c(a[2], b[2], c2, c3, c1);
  mul_add_c(a[1], b[3], c2, c3, c1);
  r[4] = c2;
  c2 = 0;
  mul_add_c(a[2], b[3], c3, c1, c2);
  mul_add_c(a[3], b[2], c3, c1, c2);
  r[5] = c3;
  c3 = 0;
  mul_add_c(a[3], b[3], c1, c2, c3);
  r[6] = c1;
  r[7] = c2;
}

void bn_sqr_comba8(BN_ULONG r[16], const BN_ULONG a[8]) {
  BN_ULONG c1, c2, c3;

  c1 = 0;
  c2 = 0;
  c3 = 0;
  sqr_add_c(a, 0, c1, c2, c3);
  r[0] = c1;
  c1 = 0;
  sqr_add_c2(a, 1, 0, c2, c3, c1);
  r[1] = c2;
  c2 = 0;
  sqr_add_c(a, 1, c3, c1, c2);
  sqr_add_c2(a, 2, 0, c3, c1, c2);
  r[2] = c3;
  c3 = 0;
  sqr_add_c2(a, 3, 0, c1, c2, c3);
  sqr_add_c2(a, 2, 1, c1, c2, c3);
  r[3] = c1;
  c1 = 0;
  sqr_add_c(a, 2, c2, c3, c1);
  sqr_add_c2(a, 3, 1, c2, c3, c1);
  sqr_add_c2(a, 4, 0, c2, c3, c1);
  r[4] = c2;
  c2 = 0;
  sqr_add_c2(a, 5, 0, c3, c1, c2);
  sqr_add_c2(a, 4, 1, c3, c1, c2);
  sqr_add_c2(a, 3, 2, c3, c1, c2);
  r[5] = c3;
  c3 = 0;
  sqr_add_c(a, 3, c1, c2, c3);
  sqr_add_c2(a, 4, 2, c1, c2, c3);
  sqr_add_c2(a, 5, 1, c1, c2, c3);
  sqr_add_c2(a, 6, 0, c1, c2, c3);
  r[6] = c1;
  c1 = 0;
  sqr_add_c2(a, 7, 0, c2, c3, c1);
  sqr_add_c2(a, 6, 1, c2, c3, c1);
  sqr_add_c2(a, 5, 2, c2, c3, c1);
  sqr_add_c2(a, 4, 3, c2, c3, c1);
  r[7] = c2;
  c2 = 0;
  sqr_add_c(a, 4, c3, c1, c2);
  sqr_add_c2(a, 5, 3, c3, c1, c2);
  sqr_add_c2(a, 6, 2, c3, c1, c2);
  sqr_add_c2(a, 7, 1, c3, c1, c2);
  r[8] = c3;
  c3 = 0;
  sqr_add_c2(a, 7, 2, c1, c2, c3);
  sqr_add_c2(a, 6, 3, c1, c2, c3);
  sqr_add_c2(a, 5, 4, c1, c2, c3);
  r[9] = c1;
  c1 = 0;
  sqr_add_c(a, 5, c2, c3, c1);
  sqr_add_c2(a, 6, 4, c2, c3, c1);
  sqr_add_c2(a, 7, 3, c2, c3, c1);
  r[10] = c2;
  c2 = 0;
  sqr_add_c2(a, 7, 4, c3, c1, c2);
  sqr_add_c2(a, 6, 5, c3, c1, c2);
  r[11] = c3;
  c3 = 0;
  sqr_add_c(a, 6, c1, c2, c3);
  sqr_add_c2(a, 7, 5, c1, c2, c3);
  r[12] = c1;
  c1 = 0;
  sqr_add_c2(a, 7, 6, c2, c3, c1);
  r[13] = c2;
  c2 = 0;
  sqr_add_c(a, 7, c3, c1, c2);
  r[14] = c3;
  r[15] = c1;
}

void bn_sqr_comba4(BN_ULONG r[8], const BN_ULONG a[4]) {
  BN_ULONG c1, c2, c3;

  c1 = 0;
  c2 = 0;
  c3 = 0;
  sqr_add_c(a, 0, c1, c2, c3);
  r[0] = c1;
  c1 = 0;
  sqr_add_c2(a, 1, 0, c2, c3, c1);
  r[1] = c2;
  c2 = 0;
  sqr_add_c(a, 1, c3, c1, c2);
  sqr_add_c2(a, 2, 0, c3, c1, c2);
  r[2] = c3;
  c3 = 0;
  sqr_add_c2(a, 3, 0, c1, c2, c3);
  sqr_add_c2(a, 2, 1, c1, c2, c3);
  r[3] = c1;
  c1 = 0;
  sqr_add_c(a, 2, c2, c3, c1);
  sqr_add_c2(a, 3, 1, c2, c3, c1);
  r[4] = c2;
  c2 = 0;
  sqr_add_c2(a, 3, 2, c3, c1, c2);
  r[5] = c3;
  c3 = 0;
  sqr_add_c(a, 3, c1, c2, c3);
  r[6] = c1;
  r[7] = c2;
}

#undef mul_add
#undef mul
#undef sqr
#undef mul_add_c
#undef mul_add_c2
#undef sqr_add_c
#undef sqr_add_c2

#endif  // !BN_MUL_ASM

#if !defined(BN_ADD_ASM)

#ifdef BN_ULLONG
BN_ULONG bn_add_words(BN_ULONG *r, const BN_ULONG *a, const BN_ULONG *b,
                      size_t n) {
  BN_ULLONG ll = 0;

  if (n == 0) {
    return 0;
  }

  while (n & ~3) {
    ll += (BN_ULLONG)a[0] + b[0];
    r[0] = (BN_ULONG)ll;
    ll >>= BN_BITS2;
    ll += (BN_ULLONG)a[1] + b[1];
    r[1] = (BN_ULONG)ll;
    ll >>= BN_BITS2;
    ll += (BN_ULLONG)a[2] + b[2];
    r[2] = (BN_ULONG)ll;
    ll >>= BN_BITS2;
    ll += (BN_ULLONG)a[3] + b[3];
    r[3] = (BN_ULONG)ll;
    ll >>= BN_BITS2;
    a += 4;
    b += 4;
    r += 4;
    n -= 4;
  }
  while (n) {
    ll += (BN_ULLONG)a[0] + b[0];
    r[0] = (BN_ULONG)ll;
    ll >>= BN_BITS2;
    a++;
    b++;
    r++;
    n--;
  }
  return (BN_ULONG)ll;
}

#else  // !BN_ULLONG

BN_ULONG bn_add_words(BN_ULONG *r, const BN_ULONG *a, const BN_ULONG *b,
                      size_t n) {
  BN_ULONG c, l, t;

  if (n == 0) {
    return (BN_ULONG)0;
  }

  c = 0;
  while (n & ~3) {
    t = a[0];
    t += c;
    c = (t < c);
    l = t + b[0];
    c += (l < t);
    r[0] = l;
    t = a[1];
    t += c;
    c = (t < c);
    l = t + b[1];
    c += (l < t);
    r[1] = l;
    t = a[2];
    t += c;
    c = (t < c);
    l = t + b[2];
    c += (l < t);
    r[2] = l;
    t = a[3];
    t += c;
    c = (t < c);
    l = t + b[3];
    c += (l < t);
    r[3] = l;
    a += 4;
    b += 4;
    r += 4;
    n -= 4;
  }
  while (n) {
    t = a[0];
    t += c;
    c = (t < c);
    l = t + b[0];
    c += (l < t);
    r[0] = l;
    a++;
    b++;
    r++;
    n--;
  }
  return (BN_ULONG)c;
}

#endif  // !BN_ULLONG

BN_ULONG bn_sub_words(BN_ULONG *r, const BN_ULONG *a, const BN_ULONG *b,
                      size_t n) {
  BN_ULONG t1, t2;
  BN_ULONG c = 0;

  if (n == 0) {
    return (BN_ULONG)0;
  }

  while (n & ~3) {
    t1 = a[0];
    t2 = b[0];
    r[0] = t1 - t2 - c;
    c = (t1 < t2) | ((t1 == t2) & c);
    t1 = a[1];
    t2 = b[1];
    r[1] = t1 - t2 - c;
    c = (t1 < t2) | ((t1 == t2) & c);
    t1 = a[2];
    t2 = b[2];
    r[2] = t1 - t2 - c;
    c = (t1 < t2) | ((t1 == t2) & c);
    t1 = a[3];
    t2 = b[3];
    r[3] = t1 - t2 - c;
    c = (t1 < t2) | ((t1 == t2) & c);
    a += 4;
    b += 4;
    r += 4;
    n -= 4;
  }
  while (n) {
    t1 = a[0];
    t2 = b[0];
    r[0] = t1 - t2 - c;
    c = (t1 < t2) | ((t1 == t2) & c);
    a++;
    b++;
    r++;
    n--;
  }
  return c;
}

#endif  // !BN_ADD_ASM
