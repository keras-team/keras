/* Copyright (c) 2020, Google Inc.
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

#ifndef OPENSSL_HEADER_CURVE25519_INTERNAL_H
#define OPENSSL_HEADER_CURVE25519_INTERNAL_H

#if defined(__cplusplus)
extern "C" {
#endif

#include <openssl/base.h>

#include "../internal.h"


#if defined(OPENSSL_ARM) && !defined(OPENSSL_NO_ASM) && !defined(OPENSSL_APPLE)
#define BORINGSSL_X25519_NEON

// x25519_NEON is defined in asm/x25519-arm.S.
void x25519_NEON(uint8_t out[32], const uint8_t scalar[32],
                 const uint8_t point[32]);
#endif

#if defined(BORINGSSL_HAS_UINT128)
#define BORINGSSL_CURVE25519_64BIT
#endif

#if defined(BORINGSSL_CURVE25519_64BIT)
// fe means field element. Here the field is \Z/(2^255-19). An element t,
// entries t[0]...t[4], represents the integer t[0]+2^51 t[1]+2^102 t[2]+2^153
// t[3]+2^204 t[4].
// fe limbs are bounded by 1.125*2^51.
// Multiplication and carrying produce fe from fe_loose.
typedef struct fe { uint64_t v[5]; } fe;

// fe_loose limbs are bounded by 3.375*2^51.
// Addition and subtraction produce fe_loose from (fe, fe).
typedef struct fe_loose { uint64_t v[5]; } fe_loose;
#else
// fe means field element. Here the field is \Z/(2^255-19). An element t,
// entries t[0]...t[9], represents the integer t[0]+2^26 t[1]+2^51 t[2]+2^77
// t[3]+2^102 t[4]+...+2^230 t[9].
// fe limbs are bounded by 1.125*2^26,1.125*2^25,1.125*2^26,1.125*2^25,etc.
// Multiplication and carrying produce fe from fe_loose.
typedef struct fe { uint32_t v[10]; } fe;

// fe_loose limbs are bounded by 3.375*2^26,3.375*2^25,3.375*2^26,3.375*2^25,etc.
// Addition and subtraction produce fe_loose from (fe, fe).
typedef struct fe_loose { uint32_t v[10]; } fe_loose;
#endif

// ge means group element.
//
// Here the group is the set of pairs (x,y) of field elements (see fe.h)
// satisfying -x^2 + y^2 = 1 + d x^2y^2
// where d = -121665/121666.
//
// Representations:
//   ge_p2 (projective): (X:Y:Z) satisfying x=X/Z, y=Y/Z
//   ge_p3 (extended): (X:Y:Z:T) satisfying x=X/Z, y=Y/Z, XY=ZT
//   ge_p1p1 (completed): ((X:Z),(Y:T)) satisfying x=X/Z, y=Y/T
//   ge_precomp (Duif): (y+x,y-x,2dxy)

typedef struct {
  fe X;
  fe Y;
  fe Z;
} ge_p2;

typedef struct {
  fe X;
  fe Y;
  fe Z;
  fe T;
} ge_p3;

typedef struct {
  fe_loose X;
  fe_loose Y;
  fe_loose Z;
  fe_loose T;
} ge_p1p1;

typedef struct {
  fe_loose yplusx;
  fe_loose yminusx;
  fe_loose xy2d;
} ge_precomp;

typedef struct {
  fe_loose YplusX;
  fe_loose YminusX;
  fe_loose Z;
  fe_loose T2d;
} ge_cached;

void x25519_ge_tobytes(uint8_t s[32], const ge_p2 *h);
int x25519_ge_frombytes_vartime(ge_p3 *h, const uint8_t s[32]);
void x25519_ge_p3_to_cached(ge_cached *r, const ge_p3 *p);
void x25519_ge_p1p1_to_p2(ge_p2 *r, const ge_p1p1 *p);
void x25519_ge_p1p1_to_p3(ge_p3 *r, const ge_p1p1 *p);
void x25519_ge_add(ge_p1p1 *r, const ge_p3 *p, const ge_cached *q);
void x25519_ge_sub(ge_p1p1 *r, const ge_p3 *p, const ge_cached *q);
void x25519_ge_scalarmult_small_precomp(
    ge_p3 *h, const uint8_t a[32], const uint8_t precomp_table[15 * 2 * 32]);
void x25519_ge_scalarmult_base(ge_p3 *h, const uint8_t a[32]);
void x25519_ge_scalarmult(ge_p2 *r, const uint8_t *scalar, const ge_p3 *A);
void x25519_sc_reduce(uint8_t s[64]);

enum spake2_state_t {
  spake2_state_init = 0,
  spake2_state_msg_generated,
  spake2_state_key_generated,
};

struct spake2_ctx_st {
  uint8_t private_key[32];
  uint8_t my_msg[32];
  uint8_t password_scalar[32];
  uint8_t password_hash[64];
  uint8_t *my_name;
  size_t my_name_len;
  uint8_t *their_name;
  size_t their_name_len;
  enum spake2_role_t my_role;
  enum spake2_state_t state;
  char disable_password_scalar_hack;
};


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CURVE25519_INTERNAL_H
