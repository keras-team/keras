/*
 * Written by Dr Stephen N Henson (steve@openssl.org) for the OpenSSL project
 * 2004.
 */
/* ====================================================================
 * Copyright (c) 2004 The OpenSSL Project.  All rights reserved.
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

#ifndef OPENSSL_HEADER_X509V3_INTERNAL_H
#define OPENSSL_HEADER_X509V3_INTERNAL_H

#include <openssl/base.h>

#include <openssl/conf.h>
#include <openssl/stack.h>
#include <openssl/x509v3.h>

// TODO(davidben): Merge x509 and x509v3. This include is needed because some
// internal typedefs are shared between the two, but the two modules depend on
// each other circularly.
#include "../x509/internal.h"

#if defined(__cplusplus)
extern "C" {
#endif


// x509v3_bytes_to_hex encodes |len| bytes from |in| to hex and returns a
// newly-allocated NUL-terminated string containing the result, or NULL on
// allocation error.
//
// This function was historically named |hex_to_string| in OpenSSL. Despite the
// name, |hex_to_string| converted to hex.
OPENSSL_EXPORT char *x509v3_bytes_to_hex(const uint8_t *in, size_t len);

// x509v3_hex_string_to_bytes decodes |str| in hex and returns a newly-allocated
// array containing the result, or NULL on error. On success, it sets |*len| to
// the length of the result. Colon separators between bytes in the input are
// allowed and ignored.
//
// This function was historically named |string_to_hex| in OpenSSL. Despite the
// name, |string_to_hex| converted from hex.
unsigned char *x509v3_hex_to_bytes(const char *str, long *len);

// x509v3_conf_name_matches returns one if |name| is equal to |cmp| or begins
// with |cmp| followed by '.', and zero otherwise.
int x509v3_conf_name_matches(const char *name, const char *cmp);

// x509v3_looks_like_dns_name returns one if |in| looks like a DNS name and zero
// otherwise.
OPENSSL_EXPORT int x509v3_looks_like_dns_name(const unsigned char *in,
                                              size_t len);

// x509v3_cache_extensions fills in a number of fields relating to X.509
// extensions in |x|. It returns one on success and zero if some extensions were
// invalid.
OPENSSL_EXPORT int x509v3_cache_extensions(X509 *x);

// x509v3_a2i_ipadd decodes |ipasc| as an IPv4 or IPv6 address. IPv6 addresses
// use colon-separated syntax while IPv4 addresses use dotted decimal syntax. If
// it decodes an IPv4 address, it writes the result to the first four bytes of
// |ipout| and returns four. If it decodes an IPv6 address, it writes the result
// to all 16 bytes of |ipout| and returns 16. Otherwise, it returns zero.
int x509v3_a2i_ipadd(unsigned char ipout[16], const char *ipasc);

// A |BIT_STRING_BITNAME| is used to contain a list of bit names.
typedef struct {
  int bitnum;
  const char *lname;
  const char *sname;
} BIT_STRING_BITNAME;

// x509V3_add_value_asn1_string appends a |CONF_VALUE| with the specified name
// and value to |*extlist|. if |*extlist| is NULL, it sets |*extlist| to a
// newly-allocated |STACK_OF(CONF_VALUE)| first. It returns one on success and
// zero on error.
int x509V3_add_value_asn1_string(const char *name, const ASN1_STRING *value,
                                 STACK_OF(CONF_VALUE) **extlist);

// X509V3_NAME_from_section adds attributes to |nm| by interpreting the
// key/value pairs in |dn_sk|. It returns one on success and zero on error.
// |chtype|, which should be one of |MBSTRING_*| constants, determines the
// character encoding used to interpret values.
int X509V3_NAME_from_section(X509_NAME *nm, const STACK_OF(CONF_VALUE) *dn_sk,
                             int chtype);

// X509V3_bool_from_string decodes |str| as a boolean. On success, it returns
// one and sets |*out_bool| to resulting value. Otherwise, it returns zero.
int X509V3_bool_from_string(const char *str, ASN1_BOOLEAN *out_bool);

// X509V3_get_value_bool decodes |value| as a boolean. On success, it returns
// one and sets |*out_bool| to the resulting value. Otherwise, it returns zero.
int X509V3_get_value_bool(const CONF_VALUE *value, ASN1_BOOLEAN *out_bool);

// X509V3_get_value_int decodes |value| as an integer. On success, it returns
// one and sets |*aint| to the resulting value. Otherwise, it returns zero. If
// |*aint| was non-NULL at the start of the function, it frees the previous
// value before writing a new one.
int X509V3_get_value_int(const CONF_VALUE *value, ASN1_INTEGER **aint);

// X509V3_get_section behaves like |NCONF_get_section| but queries |ctx|'s
// config database.
const STACK_OF(CONF_VALUE) *X509V3_get_section(const X509V3_CTX *ctx,
                                               const char *section);

// X509V3_add_value appends a |CONF_VALUE| containing |name| and |value| to
// |*extlist|. It returns one on success and zero on error. If |*extlist| is
// NULL, it sets |*extlist| to a newly-allocated |STACK_OF(CONF_VALUE)|
// containing the result. Either |name| or |value| may be NULL to omit the
// field.
//
// On failure, if |*extlist| was NULL, |*extlist| will remain NULL when the
// function returns.
int X509V3_add_value(const char *name, const char *value,
                     STACK_OF(CONF_VALUE) **extlist);

// X509V3_add_value_bool behaves like |X509V3_add_value| but stores the value
// "TRUE" if |asn1_bool| is non-zero and "FALSE" otherwise.
int X509V3_add_value_bool(const char *name, int asn1_bool,
                          STACK_OF(CONF_VALUE) **extlist);

// X509V3_add_value_bool behaves like |X509V3_add_value| but stores a string
// representation of |aint|. Note this string representation may be decimal or
// hexadecimal, depending on the size of |aint|.
int X509V3_add_value_int(const char *name, const ASN1_INTEGER *aint,
                         STACK_OF(CONF_VALUE) **extlist);

STACK_OF(CONF_VALUE) *X509V3_parse_list(const char *line);

#define X509V3_conf_err(val)                                               \
  ERR_add_error_data(6, "section:", (val)->section, ",name:", (val)->name, \
                     ",value:", (val)->value);

// GENERAL_NAME_cmp returns zero if |a| and |b| are equal and a non-zero
// value otherwise. Note this function does not provide a comparison suitable
// for sorting.
//
// This function is exported for testing.
OPENSSL_EXPORT int GENERAL_NAME_cmp(const GENERAL_NAME *a,
                                    const GENERAL_NAME *b);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_X509V3_INTERNAL_H
