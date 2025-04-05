/*
 * Written by Dr Stephen N Henson (steve@openssl.org) for the OpenSSL project
 * 2006.
 */
/* ====================================================================
 * Copyright (c) 2006 The OpenSSL Project.  All rights reserved.
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
 * Hudson (tjh@cryptsoft.com).
 *
 */

#ifndef OPENSSL_HEADER_ASN1_ASN1_LOCL_H
#define OPENSSL_HEADER_ASN1_ASN1_LOCL_H

#include <time.h>

#include <openssl/asn1.h>
#include <openssl/asn1t.h>

#if defined(__cplusplus)
extern "C" {
#endif


// Wrapper functions for time functions.

// OPENSSL_gmtime converts a time_t value in |time| which must be in the range
// of year 0000 to 9999 to a broken out time value in |tm|. On success |tm| is
// returned. On failure NULL is returned.
OPENSSL_EXPORT struct tm *OPENSSL_gmtime(const time_t *time, struct tm *result);

// OPENSSL_timegm converts a time value between the years 0 and 9999 in |tm| to
// a time_t value in |out|. One is returned on success, zero is returned on
// failure. It is a failure if the converted time can not be represented in a
// time_t, or if the tm contains out of range values.
OPENSSL_EXPORT int OPENSSL_timegm(const struct tm *tm, time_t *out);

// OPENSSL_gmtime_adj returns one on success, and updates |tm| by adding
// |offset_day| days and |offset_sec| seconds. It returns zero on failure. |tm|
// must be in the range of year 0000 to 9999 both before and after the update or
// a failure will be returned.
OPENSSL_EXPORT int OPENSSL_gmtime_adj(struct tm *tm, int offset_day,
                                      long offset_sec);

// OPENSSL_gmtime_diff calculates the difference between |from| and |to|. It
// returns one, and outputs the difference as a number of days and seconds in
// |*out_days| and |*out_secs| on success. It returns zero on failure.  Both
// |from| and |to| must be in the range of year 0000 to 9999 or a failure will
// be returned.
OPENSSL_EXPORT int OPENSSL_gmtime_diff(int *out_days, int *out_secs,
                                       const struct tm *from,
                                       const struct tm *to);

// Internal ASN1 structures and functions: not for application use

// These are used internally in the ASN1_OBJECT to keep track of
// whether the names and data need to be free()ed
#define ASN1_OBJECT_FLAG_DYNAMIC 0x01          // internal use
#define ASN1_OBJECT_FLAG_DYNAMIC_STRINGS 0x04  // internal use
#define ASN1_OBJECT_FLAG_DYNAMIC_DATA 0x08     // internal use

// An asn1_object_st (aka |ASN1_OBJECT|) represents an ASN.1 OBJECT IDENTIFIER.
// Note: Mutating an |ASN1_OBJECT| is only permitted when initializing it. The
// library maintains a table of static |ASN1_OBJECT|s, which may be referenced
// by non-const |ASN1_OBJECT| pointers. Code which receives an |ASN1_OBJECT|
// pointer externally must assume it is immutable, even if the pointer is not
// const.
struct asn1_object_st {
  const char *sn, *ln;
  int nid;
  int length;
  const unsigned char *data;  // data remains const after init
  int flags;                  // Should we free this one
};

ASN1_OBJECT *ASN1_OBJECT_new(void);

// ASN1_ENCODING is used to save the received encoding of an ASN.1 type. This
// avoids problems with invalid encodings that break signatures.
typedef struct ASN1_ENCODING_st {
  // enc is the saved DER encoding. Its ownership is determined by |buf|.
  uint8_t *enc;
  // len is the length of |enc|. If zero, there is no saved encoding.
  size_t len;
  // buf, if non-NULL, is the |CRYPTO_BUFFER| that |enc| points into. If NULL,
  // |enc| must be released with |OPENSSL_free|.
  CRYPTO_BUFFER *buf;
} ASN1_ENCODING;

OPENSSL_EXPORT int asn1_utctime_to_tm(struct tm *tm, const ASN1_UTCTIME *d,
                                      int allow_timezone_offset);
OPENSSL_EXPORT int asn1_generalizedtime_to_tm(struct tm *tm,
                                              const ASN1_GENERALIZEDTIME *d);

int ASN1_item_ex_new(ASN1_VALUE **pval, const ASN1_ITEM *it);
void ASN1_item_ex_free(ASN1_VALUE **pval, const ASN1_ITEM *it);

void ASN1_template_free(ASN1_VALUE **pval, const ASN1_TEMPLATE *tt);

// ASN1_item_ex_d2i parses |len| bytes from |*in| as a structure of type |it|
// and writes the result to |*pval|. If |tag| is non-negative, |it| is
// implicitly tagged with the tag specified by |tag| and |aclass|. If |opt| is
// non-zero, the value is optional. If |buf| is non-NULL, |*in| must point into
// |buf|.
//
// This function returns one and advances |*in| if an object was successfully
// parsed, -1 if an optional value was successfully skipped, and zero on error.
int ASN1_item_ex_d2i(ASN1_VALUE **pval, const unsigned char **in, long len,
                     const ASN1_ITEM *it, int tag, int aclass, char opt,
                     CRYPTO_BUFFER *buf);

// ASN1_item_ex_i2d encodes |*pval| as a value of type |it| to |out| under the
// i2d output convention. It returns a non-zero length on success and -1 on
// error. If |tag| is -1. the tag and class come from |it|. Otherwise, the tag
// number is |tag| and the class is |aclass|. This is used for implicit tagging.
// This function treats a missing value as an error, not an optional field.
int ASN1_item_ex_i2d(ASN1_VALUE **pval, unsigned char **out,
                     const ASN1_ITEM *it, int tag, int aclass);

void ASN1_primitive_free(ASN1_VALUE **pval, const ASN1_ITEM *it);

// asn1_get_choice_selector returns the CHOICE selector value for |*pval|, which
// must of type |it|.
int asn1_get_choice_selector(ASN1_VALUE **pval, const ASN1_ITEM *it);

int asn1_set_choice_selector(ASN1_VALUE **pval, int value, const ASN1_ITEM *it);

// asn1_get_field_ptr returns a pointer to the field in |*pval| corresponding to
// |tt|.
ASN1_VALUE **asn1_get_field_ptr(ASN1_VALUE **pval, const ASN1_TEMPLATE *tt);

// asn1_do_adb returns the |ASN1_TEMPLATE| for the ANY DEFINED BY field |tt|,
// based on the selector INTEGER or OID in |*pval|. If |tt| is not an ADB field,
// it returns |tt|. If the selector does not match any value, it returns NULL.
// If |nullerr| is non-zero, it will additionally push an error to the error
// queue when there is no match.
const ASN1_TEMPLATE *asn1_do_adb(ASN1_VALUE **pval, const ASN1_TEMPLATE *tt,
                                 int nullerr);

void asn1_refcount_set_one(ASN1_VALUE **pval, const ASN1_ITEM *it);
int asn1_refcount_dec_and_test_zero(ASN1_VALUE **pval, const ASN1_ITEM *it);

void asn1_enc_init(ASN1_VALUE **pval, const ASN1_ITEM *it);
void asn1_enc_free(ASN1_VALUE **pval, const ASN1_ITEM *it);

// asn1_enc_restore, if |*pval| has a saved encoding, writes it to |out| under
// the i2d output convention, sets |*len| to the length, and returns one. If it
// has no saved encoding, it returns zero.
int asn1_enc_restore(int *len, unsigned char **out, ASN1_VALUE **pval,
                     const ASN1_ITEM *it);

// asn1_enc_save saves |inlen| bytes from |in| as |*pval|'s saved encoding. It
// returns one on success and zero on error. If |buf| is non-NULL, |in| must
// point into |buf|.
int asn1_enc_save(ASN1_VALUE **pval, const uint8_t *in, size_t inlen,
                  const ASN1_ITEM *it, CRYPTO_BUFFER *buf);

// asn1_encoding_clear clears the cached encoding in |enc|.
void asn1_encoding_clear(ASN1_ENCODING *enc);

// asn1_type_value_as_pointer returns |a|'s value in pointer form. This is
// usually the value object but, for BOOLEAN values, is 0 or 0xff cast to
// a pointer.
const void *asn1_type_value_as_pointer(const ASN1_TYPE *a);

// asn1_is_printable returns one if |value| is a valid Unicode codepoint for an
// ASN.1 PrintableString, and zero otherwise.
int asn1_is_printable(uint32_t value);

// asn1_bit_string_length returns the number of bytes in |str| and sets
// |*out_padding_bits| to the number of padding bits.
//
// This function should be used instead of |ASN1_STRING_length| to correctly
// handle the non-|ASN1_STRING_FLAG_BITS_LEFT| case.
int asn1_bit_string_length(const ASN1_BIT_STRING *str,
                           uint8_t *out_padding_bits);

typedef struct {
  int nid;
  long minsize;
  long maxsize;
  unsigned long mask;
  unsigned long flags;
} ASN1_STRING_TABLE;

// asn1_get_string_table_for_testing sets |*out_ptr| and |*out_len| to the table
// of built-in |ASN1_STRING_TABLE| values. It is exported for testing.
OPENSSL_EXPORT void asn1_get_string_table_for_testing(
    const ASN1_STRING_TABLE **out_ptr, size_t *out_len);

typedef ASN1_VALUE *ASN1_new_func(void);
typedef void ASN1_free_func(ASN1_VALUE *a);
typedef ASN1_VALUE *ASN1_d2i_func(ASN1_VALUE **a, const unsigned char **in,
                                  long length);
typedef int ASN1_i2d_func(ASN1_VALUE *a, unsigned char **in);

typedef int ASN1_ex_d2i(ASN1_VALUE **pval, const unsigned char **in, long len,
                        const ASN1_ITEM *it, int opt, ASN1_TLC *ctx);

typedef int ASN1_ex_i2d(ASN1_VALUE **pval, unsigned char **out,
                        const ASN1_ITEM *it);
typedef int ASN1_ex_new_func(ASN1_VALUE **pval, const ASN1_ITEM *it);
typedef void ASN1_ex_free_func(ASN1_VALUE **pval, const ASN1_ITEM *it);

typedef struct ASN1_EXTERN_FUNCS_st {
  ASN1_ex_new_func *asn1_ex_new;
  ASN1_ex_free_func *asn1_ex_free;
  ASN1_ex_free_func *asn1_ex_clear;
  ASN1_ex_d2i *asn1_ex_d2i;
  ASN1_ex_i2d *asn1_ex_i2d;
} ASN1_EXTERN_FUNCS;


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_ASN1_ASN1_LOCL_H
