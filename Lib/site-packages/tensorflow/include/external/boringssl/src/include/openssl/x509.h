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
 * Copyright 2002 Sun Microsystems, Inc. ALL RIGHTS RESERVED.
 * ECDH support in OpenSSL originally developed by
 * SUN MICROSYSTEMS, INC., and contributed to the OpenSSL project.
 */

#ifndef OPENSSL_HEADER_X509_H
#define OPENSSL_HEADER_X509_H

#include <openssl/asn1.h>
#include <openssl/base.h>
#include <openssl/bio.h>
#include <openssl/cipher.h>
#include <openssl/dh.h>
#include <openssl/dsa.h>
#include <openssl/ec.h>
#include <openssl/ecdh.h>
#include <openssl/ecdsa.h>
#include <openssl/evp.h>
#include <openssl/obj.h>
#include <openssl/pkcs7.h>
#include <openssl/pool.h>
#include <openssl/rsa.h>
#include <openssl/sha.h>
#include <openssl/stack.h>
#include <openssl/thread.h>
#include <time.h>

#if defined(__cplusplus)
extern "C" {
#endif


// Legacy X.509 library.
//
// This header is part of OpenSSL's X.509 implementation. It is retained for
// compatibility but should not be used by new code. The functions are difficult
// to use correctly, and have buggy or non-standard behaviors. They are thus
// particularly prone to behavior changes and API removals, as BoringSSL
// iterates on these issues.
//
// In the future, a replacement library will be available. Meanwhile, minimize
// dependencies on this header where possible.
//
// TODO(https://crbug.com/boringssl/426): Documentation for this library is
// still in progress. Some functions have not yet been documented, and some
// functions have not yet been grouped into sections.


// Certificates.
//
// An |X509| object represents an X.509 certificate, defined in RFC 5280.
//
// Although an |X509| is a mutable object, mutating an |X509| can give incorrect
// results. Callers typically obtain |X509|s by parsing some input with
// |d2i_X509|, etc. Such objects carry information such as the serialized
// TBSCertificate and decoded extensions, which will become inconsistent when
// mutated.
//
// Instead, mutation functions should only be used when issuing new
// certificates, as described in a later section.

DEFINE_STACK_OF(X509)

// X509 is an |ASN1_ITEM| whose ASN.1 type is X.509 Certificate (RFC 5280) and C
// type is |X509*|.
DECLARE_ASN1_ITEM(X509)

// X509_up_ref adds one to the reference count of |x509| and returns one.
OPENSSL_EXPORT int X509_up_ref(X509 *x509);

// X509_chain_up_ref returns a newly-allocated |STACK_OF(X509)| containing a
// shallow copy of |chain|, or NULL on error. That is, the return value has the
// same contents as |chain|, and each |X509|'s reference count is incremented by
// one.
OPENSSL_EXPORT STACK_OF(X509) *X509_chain_up_ref(STACK_OF(X509) *chain);

// X509_dup returns a newly-allocated copy of |x509|, or NULL on error. This
// function works by serializing the structure, so auxiliary properties (see
// |i2d_X509_AUX|) are not preserved. Additionally, if |x509| is incomplete,
// this function may fail.
//
// TODO(https://crbug.com/boringssl/407): This function should be const and
// thread-safe but is currently neither in some cases, notably if |crl| was
// mutated.
OPENSSL_EXPORT X509 *X509_dup(X509 *x509);

// X509_free decrements |x509|'s reference count and, if zero, releases memory
// associated with |x509|.
OPENSSL_EXPORT void X509_free(X509 *x509);

// d2i_X509 parses up to |len| bytes from |*inp| as a DER-encoded X.509
// Certificate (RFC 5280), as described in |d2i_SAMPLE|.
OPENSSL_EXPORT X509 *d2i_X509(X509 **out, const uint8_t **inp, long len);

// X509_parse_from_buffer parses an X.509 structure from |buf| and returns a
// fresh X509 or NULL on error. There must not be any trailing data in |buf|.
// The returned structure (if any) holds a reference to |buf| rather than
// copying parts of it as a normal |d2i_X509| call would do.
OPENSSL_EXPORT X509 *X509_parse_from_buffer(CRYPTO_BUFFER *buf);

// i2d_X509 marshals |x509| as a DER-encoded X.509 Certificate (RFC 5280), as
// described in |i2d_SAMPLE|.
//
// TODO(https://crbug.com/boringssl/407): This function should be const and
// thread-safe but is currently neither in some cases, notably if |x509| was
// mutated.
OPENSSL_EXPORT int i2d_X509(X509 *x509, uint8_t **outp);

// X509_VERSION_* are X.509 version numbers. Note the numerical values of all
// defined X.509 versions are one less than the named version.
#define X509_VERSION_1 0
#define X509_VERSION_2 1
#define X509_VERSION_3 2

// X509_get_version returns the numerical value of |x509|'s version, which will
// be one of the |X509_VERSION_*| constants.
OPENSSL_EXPORT long X509_get_version(const X509 *x509);

// X509_get0_serialNumber returns |x509|'s serial number.
OPENSSL_EXPORT const ASN1_INTEGER *X509_get0_serialNumber(const X509 *x509);

// X509_get0_notBefore returns |x509|'s notBefore time.
OPENSSL_EXPORT const ASN1_TIME *X509_get0_notBefore(const X509 *x509);

// X509_get0_notAfter returns |x509|'s notAfter time.
OPENSSL_EXPORT const ASN1_TIME *X509_get0_notAfter(const X509 *x509);

// X509_get_issuer_name returns |x509|'s issuer.
OPENSSL_EXPORT X509_NAME *X509_get_issuer_name(const X509 *x509);

// X509_get_subject_name returns |x509|'s subject.
OPENSSL_EXPORT X509_NAME *X509_get_subject_name(const X509 *x509);

// X509_get_X509_PUBKEY returns the public key of |x509|. Note this function is
// not const-correct for legacy reasons. Callers should not modify the returned
// object.
OPENSSL_EXPORT X509_PUBKEY *X509_get_X509_PUBKEY(const X509 *x509);

// X509_get_pubkey returns |x509|'s public key as an |EVP_PKEY|, or NULL if the
// public key was unsupported or could not be decoded. This function returns a
// reference to the |EVP_PKEY|. The caller must release the result with
// |EVP_PKEY_free| when done.
OPENSSL_EXPORT EVP_PKEY *X509_get_pubkey(X509 *x509);

// X509_get0_pubkey_bitstr returns the BIT STRING portion of |x509|'s public
// key. Note this does not contain the AlgorithmIdentifier portion.
//
// WARNING: This function returns a non-const pointer for OpenSSL compatibility,
// but the caller must not modify the resulting object. Doing so will break
// internal invariants in |x509|.
OPENSSL_EXPORT ASN1_BIT_STRING *X509_get0_pubkey_bitstr(const X509 *x509);

// X509_get0_uids sets |*out_issuer_uid| to a non-owning pointer to the
// issuerUID field of |x509|, or NULL if |x509| has no issuerUID. It similarly
// outputs |x509|'s subjectUID field to |*out_subject_uid|.
//
// Callers may pass NULL to either |out_issuer_uid| or |out_subject_uid| to
// ignore the corresponding field.
OPENSSL_EXPORT void X509_get0_uids(const X509 *x509,
                                   const ASN1_BIT_STRING **out_issuer_uid,
                                   const ASN1_BIT_STRING **out_subject_uid);

// X509_get0_extensions returns |x509|'s extension list, or NULL if |x509| omits
// it.
OPENSSL_EXPORT const STACK_OF(X509_EXTENSION) *X509_get0_extensions(
    const X509 *x509);

// X509_get_ext_count returns the number of extensions in |x|.
OPENSSL_EXPORT int X509_get_ext_count(const X509 *x);

// X509_get_ext_by_NID behaves like |X509v3_get_ext_by_NID| but searches for
// extensions in |x|.
OPENSSL_EXPORT int X509_get_ext_by_NID(const X509 *x, int nid, int lastpos);

// X509_get_ext_by_OBJ behaves like |X509v3_get_ext_by_OBJ| but searches for
// extensions in |x|.
OPENSSL_EXPORT int X509_get_ext_by_OBJ(const X509 *x, const ASN1_OBJECT *obj,
                                       int lastpos);

// X509_get_ext_by_critical behaves like |X509v3_get_ext_by_critical| but
// searches for extensions in |x|.
OPENSSL_EXPORT int X509_get_ext_by_critical(const X509 *x, int crit,
                                            int lastpos);

// X509_get_ext returns the extension in |x| at index |loc|, or NULL if |loc| is
// out of bounds. This function returns a non-const pointer for OpenSSL
// compatibility, but callers should not mutate the result.
OPENSSL_EXPORT X509_EXTENSION *X509_get_ext(const X509 *x, int loc);

// X509_get0_tbs_sigalg returns the signature algorithm in |x509|'s
// TBSCertificate. For the outer signature algorithm, see |X509_get0_signature|.
//
// Certificates with mismatched signature algorithms will successfully parse,
// but they will be rejected when verifying.
OPENSSL_EXPORT const X509_ALGOR *X509_get0_tbs_sigalg(const X509 *x509);

// X509_get0_signature sets |*out_sig| and |*out_alg| to the signature and
// signature algorithm of |x509|, respectively. Either output pointer may be
// NULL to ignore the value.
//
// This function outputs the outer signature algorithm. For the one in the
// TBSCertificate, see |X509_get0_tbs_sigalg|. Certificates with mismatched
// signature algorithms will successfully parse, but they will be rejected when
// verifying.
OPENSSL_EXPORT void X509_get0_signature(const ASN1_BIT_STRING **out_sig,
                                        const X509_ALGOR **out_alg,
                                        const X509 *x509);

// X509_get_signature_nid returns the NID corresponding to |x509|'s signature
// algorithm, or |NID_undef| if the signature algorithm does not correspond to
// a known NID.
OPENSSL_EXPORT int X509_get_signature_nid(const X509 *x509);

// i2d_X509_tbs serializes the TBSCertificate portion of |x509|, as described in
// |i2d_SAMPLE|.
//
// This function preserves the original encoding of the TBSCertificate and may
// not reflect modifications made to |x509|. It may be used to manually verify
// the signature of an existing certificate. To generate certificates, use
// |i2d_re_X509_tbs| instead.
OPENSSL_EXPORT int i2d_X509_tbs(X509 *x509, unsigned char **outp);


// Issuing certificates.
//
// An |X509| object may also represent an incomplete certificate. Callers may
// construct empty |X509| objects, fill in fields individually, and finally sign
// the result. The following functions may be used for this purpose.

// X509_new returns a newly-allocated, empty |X509| object, or NULL on error.
// This produces an incomplete certificate which may be filled in to issue a new
// certificate.
OPENSSL_EXPORT X509 *X509_new(void);

// X509_set_version sets |x509|'s version to |version|, which should be one of
// the |X509V_VERSION_*| constants. It returns one on success and zero on error.
//
// If unsure, use |X509_VERSION_3|.
OPENSSL_EXPORT int X509_set_version(X509 *x509, long version);

// X509_set_serialNumber sets |x509|'s serial number to |serial|. It returns one
// on success and zero on error.
OPENSSL_EXPORT int X509_set_serialNumber(X509 *x509,
                                         const ASN1_INTEGER *serial);

// X509_set1_notBefore sets |x509|'s notBefore time to |tm|. It returns one on
// success and zero on error.
OPENSSL_EXPORT int X509_set1_notBefore(X509 *x509, const ASN1_TIME *tm);

// X509_set1_notAfter sets |x509|'s notAfter time to |tm|. it returns one on
// success and zero on error.
OPENSSL_EXPORT int X509_set1_notAfter(X509 *x509, const ASN1_TIME *tm);

// X509_getm_notBefore returns a mutable pointer to |x509|'s notBefore time.
OPENSSL_EXPORT ASN1_TIME *X509_getm_notBefore(X509 *x509);

// X509_getm_notAfter returns a mutable pointer to |x509|'s notAfter time.
OPENSSL_EXPORT ASN1_TIME *X509_getm_notAfter(X509 *x);

// X509_set_issuer_name sets |x509|'s issuer to a copy of |name|. It returns one
// on success and zero on error.
OPENSSL_EXPORT int X509_set_issuer_name(X509 *x509, X509_NAME *name);

// X509_set_subject_name sets |x509|'s subject to a copy of |name|. It returns
// one on success and zero on error.
OPENSSL_EXPORT int X509_set_subject_name(X509 *x509, X509_NAME *name);

// X509_set_pubkey sets |x509|'s public key to |pkey|. It returns one on success
// and zero on error. This function does not take ownership of |pkey| and
// internally copies and updates reference counts as needed.
OPENSSL_EXPORT int X509_set_pubkey(X509 *x509, EVP_PKEY *pkey);

// X509_delete_ext removes the extension in |x| at index |loc| and returns the
// removed extension, or NULL if |loc| was out of bounds. If non-NULL, the
// caller must release the result with |X509_EXTENSION_free|.
OPENSSL_EXPORT X509_EXTENSION *X509_delete_ext(X509 *x, int loc);

// X509_add_ext adds a copy of |ex| to |x|. It returns one on success and zero
// on failure. The caller retains ownership of |ex| and can release it
// independently of |x|.
//
// The new extension is inserted at index |loc|, shifting extensions to the
// right. If |loc| is -1 or out of bounds, the new extension is appended to the
// list.
OPENSSL_EXPORT int X509_add_ext(X509 *x, const X509_EXTENSION *ex, int loc);

// X509_sign signs |x509| with |pkey| and replaces the signature algorithm and
// signature fields. It returns one on success and zero on error. This function
// uses digest algorithm |md|, or |pkey|'s default if NULL. Other signing
// parameters use |pkey|'s defaults. To customize them, use |X509_sign_ctx|.
OPENSSL_EXPORT int X509_sign(X509 *x509, EVP_PKEY *pkey, const EVP_MD *md);

// X509_sign_ctx signs |x509| with |ctx| and replaces the signature algorithm
// and signature fields. It returns one on success and zero on error. The
// signature algorithm and parameters come from |ctx|, which must have been
// initialized with |EVP_DigestSignInit|. The caller should configure the
// corresponding |EVP_PKEY_CTX| before calling this function.
OPENSSL_EXPORT int X509_sign_ctx(X509 *x509, EVP_MD_CTX *ctx);

// i2d_re_X509_tbs serializes the TBSCertificate portion of |x509|, as described
// in |i2d_SAMPLE|.
//
// This function re-encodes the TBSCertificate and may not reflect |x509|'s
// original encoding. It may be used to manually generate a signature for a new
// certificate. To verify certificates, use |i2d_X509_tbs| instead.
OPENSSL_EXPORT int i2d_re_X509_tbs(X509 *x509, unsigned char **outp);

// X509_set1_signature_algo sets |x509|'s signature algorithm to |algo| and
// returns one on success or zero on error. It updates both the signature field
// of the TBSCertificate structure, and the signatureAlgorithm field of the
// Certificate.
OPENSSL_EXPORT int X509_set1_signature_algo(X509 *x509, const X509_ALGOR *algo);

// X509_set1_signature_value sets |x509|'s signature to a copy of the |sig_len|
// bytes pointed by |sig|. It returns one on success and zero on error.
//
// Due to a specification error, X.509 certificates store signatures in ASN.1
// BIT STRINGs, but signature algorithms return byte strings rather than bit
// strings. This function creates a BIT STRING containing a whole number of
// bytes, with the bit order matching the DER encoding. This matches the
// encoding used by all X.509 signature algorithms.
OPENSSL_EXPORT int X509_set1_signature_value(X509 *x509, const uint8_t *sig,
                                             size_t sig_len);


// Auxiliary certificate properties.
//
// |X509| objects optionally maintain auxiliary properties. These are not part
// of the certificates themselves, and thus are not covered by signatures or
// preserved by the standard serialization. They are used as inputs or outputs
// to other functions in this library.

// i2d_X509_AUX marshals |x509| as a DER-encoded X.509 Certificate (RFC 5280),
// followed optionally by a separate, OpenSSL-specific structure with auxiliary
// properties. It behaves as described in |i2d_SAMPLE|.
//
// Unlike similarly-named functions, this function does not output a single
// ASN.1 element. Directly embedding the output in a larger ASN.1 structure will
// not behave correctly.
OPENSSL_EXPORT int i2d_X509_AUX(X509 *x509, unsigned char **outp);

// d2i_X509_AUX parses up to |length| bytes from |*inp| as a DER-encoded X.509
// Certificate (RFC 5280), followed optionally by a separate, OpenSSL-specific
// structure with auxiliary properties. It behaves as described in |d2i_SAMPLE|.
//
// Some auxiliary properties affect trust decisions, so this function should not
// be used with untrusted input.
//
// Unlike similarly-named functions, this function does not parse a single
// ASN.1 element. Trying to parse data directly embedded in a larger ASN.1
// structure will not behave correctly.
OPENSSL_EXPORT X509 *d2i_X509_AUX(X509 **x509, const unsigned char **inp,
                                  long length);

// X509_alias_set1 sets |x509|'s alias to |len| bytes from |name|. If |name| is
// NULL, the alias is cleared instead. Aliases are not part of the certificate
// itself and will not be serialized by |i2d_X509|.
OPENSSL_EXPORT int X509_alias_set1(X509 *x509, const unsigned char *name,
                                   int len);

// X509_keyid_set1 sets |x509|'s key ID to |len| bytes from |id|. If |id| is
// NULL, the key ID is cleared instead. Key IDs are not part of the certificate
// itself and will not be serialized by |i2d_X509|.
OPENSSL_EXPORT int X509_keyid_set1(X509 *x509, const unsigned char *id,
                                   int len);

// X509_alias_get0 looks up |x509|'s alias. If found, it sets |*out_len| to the
// alias's length and returns a pointer to a buffer containing the contents. If
// not found, it outputs the empty string by returning NULL and setting
// |*out_len| to zero.
//
// If |x509| was parsed from a PKCS#12 structure (see
// |PKCS12_get_key_and_certs|), the alias will reflect the friendlyName
// attribute (RFC 2985).
//
// WARNING: In OpenSSL, this function did not set |*out_len| when the alias was
// missing. Callers that target both OpenSSL and BoringSSL should set the value
// to zero before calling this function.
OPENSSL_EXPORT unsigned char *X509_alias_get0(X509 *x509, int *out_len);

// X509_keyid_get0 looks up |x509|'s key ID. If found, it sets |*out_len| to the
// key ID's length and returns a pointer to a buffer containing the contents. If
// not found, it outputs the empty string by returning NULL and setting
// |*out_len| to zero.
//
// WARNING: In OpenSSL, this function did not set |*out_len| when the alias was
// missing. Callers that target both OpenSSL and BoringSSL should set the value
// to zero before calling this function.
OPENSSL_EXPORT unsigned char *X509_keyid_get0(X509 *x509, int *out_len);


// Certificate revocation lists.
//
// An |X509_CRL| object represents an X.509 certificate revocation list (CRL),
// defined in RFC 5280. A CRL is a signed list of certificates which are no
// longer considered valid.
//
// Although an |X509_CRL| is a mutable object, mutating an |X509_CRL| can give
// incorrect results. Callers typically obtain |X509_CRL|s by parsing some input
// with |d2i_X509_CRL|, etc. Such objects carry information such as the
// serialized TBSCertList and decoded extensions, which will become inconsistent
// when mutated.
//
// Instead, mutation functions should only be used when issuing new CRLs, as
// described in a later section.

DEFINE_STACK_OF(X509_CRL)

// X509_CRL is an |ASN1_ITEM| whose ASN.1 type is X.509 CertificateList (RFC
// 5280) and C type is |X509_CRL*|.
DECLARE_ASN1_ITEM(X509_CRL)

// X509_CRL_up_ref adds one to the reference count of |crl| and returns one.
OPENSSL_EXPORT int X509_CRL_up_ref(X509_CRL *crl);

// X509_CRL_dup returns a newly-allocated copy of |crl|, or NULL on error. This
// function works by serializing the structure, so if |crl| is incomplete, it
// may fail.
//
// TODO(https://crbug.com/boringssl/407): This function should be const and
// thread-safe but is currently neither in some cases, notably if |crl| was
// mutated.
OPENSSL_EXPORT X509_CRL *X509_CRL_dup(X509_CRL *crl);

// X509_CRL_free decrements |crl|'s reference count and, if zero, releases
// memory associated with |crl|.
OPENSSL_EXPORT void X509_CRL_free(X509_CRL *crl);

// d2i_X509_CRL parses up to |len| bytes from |*inp| as a DER-encoded X.509
// CertificateList (RFC 5280), as described in |d2i_SAMPLE|.
OPENSSL_EXPORT X509_CRL *d2i_X509_CRL(X509_CRL **out, const uint8_t **inp,
                                      long len);

// i2d_X509_CRL marshals |crl| as a X.509 CertificateList (RFC 5280), as
// described in |i2d_SAMPLE|.
//
// TODO(https://crbug.com/boringssl/407): This function should be const and
// thread-safe but is currently neither in some cases, notably if |crl| was
// mutated.
OPENSSL_EXPORT int i2d_X509_CRL(X509_CRL *crl, uint8_t **outp);

#define X509_CRL_VERSION_1 0
#define X509_CRL_VERSION_2 1

// X509_CRL_get_version returns the numerical value of |crl|'s version, which
// will be one of the |X509_CRL_VERSION_*| constants.
OPENSSL_EXPORT long X509_CRL_get_version(const X509_CRL *crl);

// X509_CRL_get0_lastUpdate returns |crl|'s thisUpdate time. The OpenSSL API
// refers to this field as lastUpdate.
OPENSSL_EXPORT const ASN1_TIME *X509_CRL_get0_lastUpdate(const X509_CRL *crl);

// X509_CRL_get0_nextUpdate returns |crl|'s nextUpdate time, or NULL if |crl|
// has none.
OPENSSL_EXPORT const ASN1_TIME *X509_CRL_get0_nextUpdate(const X509_CRL *crl);

// X509_CRL_get_issuer returns |crl|'s issuer name. Note this function is not
// const-correct for legacy reasons.
OPENSSL_EXPORT X509_NAME *X509_CRL_get_issuer(const X509_CRL *crl);

// X509_CRL_get_REVOKED returns the list of revoked certificates in |crl|, or
// NULL if |crl| omits it.
//
// TOOD(davidben): This function was originally a macro, without clear const
// semantics. It should take a const input and give const output, but the latter
// would break existing callers. For now, we match upstream.
OPENSSL_EXPORT STACK_OF(X509_REVOKED) *X509_CRL_get_REVOKED(X509_CRL *crl);

// X509_CRL_get0_extensions returns |crl|'s extension list, or NULL if |crl|
// omits it.
OPENSSL_EXPORT const STACK_OF(X509_EXTENSION) *X509_CRL_get0_extensions(
    const X509_CRL *crl);

// X509_CRL_get_ext_count returns the number of extensions in |x|.
OPENSSL_EXPORT int X509_CRL_get_ext_count(const X509_CRL *x);

// X509_CRL_get_ext_by_NID behaves like |X509v3_get_ext_by_NID| but searches for
// extensions in |x|.
OPENSSL_EXPORT int X509_CRL_get_ext_by_NID(const X509_CRL *x, int nid,
                                           int lastpos);

// X509_CRL_get_ext_by_OBJ behaves like |X509v3_get_ext_by_OBJ| but searches for
// extensions in |x|.
OPENSSL_EXPORT int X509_CRL_get_ext_by_OBJ(const X509_CRL *x,
                                           const ASN1_OBJECT *obj, int lastpos);

// X509_CRL_get_ext_by_critical behaves like |X509v3_get_ext_by_critical| but
// searches for extensions in |x|.
OPENSSL_EXPORT int X509_CRL_get_ext_by_critical(const X509_CRL *x, int crit,
                                                int lastpos);

// X509_CRL_get_ext returns the extension in |x| at index |loc|, or NULL if
// |loc| is out of bounds. This function returns a non-const pointer for OpenSSL
// compatibility, but callers should not mutate the result.
OPENSSL_EXPORT X509_EXTENSION *X509_CRL_get_ext(const X509_CRL *x, int loc);

// X509_CRL_get0_signature sets |*out_sig| and |*out_alg| to the signature and
// signature algorithm of |crl|, respectively. Either output pointer may be NULL
// to ignore the value.
//
// This function outputs the outer signature algorithm, not the one in the
// TBSCertList. CRLs with mismatched signature algorithms will successfully
// parse, but they will be rejected when verifying.
OPENSSL_EXPORT void X509_CRL_get0_signature(const X509_CRL *crl,
                                            const ASN1_BIT_STRING **out_sig,
                                            const X509_ALGOR **out_alg);

// X509_CRL_get_signature_nid returns the NID corresponding to |crl|'s signature
// algorithm, or |NID_undef| if the signature algorithm does not correspond to
// a known NID.
OPENSSL_EXPORT int X509_CRL_get_signature_nid(const X509_CRL *crl);

// i2d_X509_CRL_tbs serializes the TBSCertList portion of |crl|, as described in
// |i2d_SAMPLE|.
//
// This function preserves the original encoding of the TBSCertList and may not
// reflect modifications made to |crl|. It may be used to manually verify the
// signature of an existing CRL. To generate CRLs, use |i2d_re_X509_CRL_tbs|
// instead.
OPENSSL_EXPORT int i2d_X509_CRL_tbs(X509_CRL *crl, unsigned char **outp);


// Issuing certificate revocation lists.
//
// An |X509_CRL| object may also represent an incomplete CRL. Callers may
// construct empty |X509_CRL| objects, fill in fields individually, and finally
// sign the result. The following functions may be used for this purpose.

// X509_CRL_new returns a newly-allocated, empty |X509_CRL| object, or NULL on
// error. This object may be filled in and then signed to construct a CRL.
OPENSSL_EXPORT X509_CRL *X509_CRL_new(void);

// X509_CRL_set_version sets |crl|'s version to |version|, which should be one
// of the |X509_CRL_VERSION_*| constants. It returns one on success and zero on
// error.
//
// If unsure, use |X509_CRL_VERSION_2|. Note that, unlike certificates, CRL
// versions are only defined up to v2. Callers should not use |X509_VERSION_3|.
OPENSSL_EXPORT int X509_CRL_set_version(X509_CRL *crl, long version);

// X509_CRL_set_issuer_name sets |crl|'s issuer to a copy of |name|. It returns
// one on success and zero on error.
OPENSSL_EXPORT int X509_CRL_set_issuer_name(X509_CRL *crl, X509_NAME *name);

// X509_CRL_set1_lastUpdate sets |crl|'s thisUpdate time to |tm|. It returns one
// on success and zero on error. The OpenSSL API refers to this field as
// lastUpdate.
OPENSSL_EXPORT int X509_CRL_set1_lastUpdate(X509_CRL *crl, const ASN1_TIME *tm);

// X509_CRL_set1_nextUpdate sets |crl|'s nextUpdate time to |tm|. It returns one
// on success and zero on error.
OPENSSL_EXPORT int X509_CRL_set1_nextUpdate(X509_CRL *crl, const ASN1_TIME *tm);

// X509_CRL_delete_ext removes the extension in |x| at index |loc| and returns
// the removed extension, or NULL if |loc| was out of bounds. If non-NULL, the
// caller must release the result with |X509_EXTENSION_free|.
OPENSSL_EXPORT X509_EXTENSION *X509_CRL_delete_ext(X509_CRL *x, int loc);

// X509_CRL_add_ext adds a copy of |ex| to |x|. It returns one on success and
// zero on failure. The caller retains ownership of |ex| and can release it
// independently of |x|.
//
// The new extension is inserted at index |loc|, shifting extensions to the
// right. If |loc| is -1 or out of bounds, the new extension is appended to the
// list.
OPENSSL_EXPORT int X509_CRL_add_ext(X509_CRL *x, const X509_EXTENSION *ex,
                                    int loc);

// X509_CRL_sign signs |crl| with |pkey| and replaces the signature algorithm
// and signature fields. It returns one on success and zero on error. This
// function uses digest algorithm |md|, or |pkey|'s default if NULL. Other
// signing parameters use |pkey|'s defaults. To customize them, use
// |X509_CRL_sign_ctx|.
OPENSSL_EXPORT int X509_CRL_sign(X509_CRL *crl, EVP_PKEY *pkey,
                                 const EVP_MD *md);

// X509_CRL_sign_ctx signs |crl| with |ctx| and replaces the signature algorithm
// and signature fields. It returns one on success and zero on error. The
// signature algorithm and parameters come from |ctx|, which must have been
// initialized with |EVP_DigestSignInit|. The caller should configure the
// corresponding |EVP_PKEY_CTX| before calling this function.
OPENSSL_EXPORT int X509_CRL_sign_ctx(X509_CRL *crl, EVP_MD_CTX *ctx);

// i2d_re_X509_CRL_tbs serializes the TBSCertList portion of |crl|, as described
// in |i2d_SAMPLE|.
//
// This function re-encodes the TBSCertList and may not reflect |crl|'s original
// encoding. It may be used to manually generate a signature for a new CRL. To
// verify CRLs, use |i2d_X509_CRL_tbs| instead.
OPENSSL_EXPORT int i2d_re_X509_CRL_tbs(X509_CRL *crl, unsigned char **outp);

// X509_CRL_set1_signature_algo sets |crl|'s signature algorithm to |algo| and
// returns one on success or zero on error. It updates both the signature field
// of the TBSCertList structure, and the signatureAlgorithm field of the CRL.
OPENSSL_EXPORT int X509_CRL_set1_signature_algo(X509_CRL *crl,
                                                const X509_ALGOR *algo);

// X509_CRL_set1_signature_value sets |crl|'s signature to a copy of the
// |sig_len| bytes pointed by |sig|. It returns one on success and zero on
// error.
//
// Due to a specification error, X.509 CRLs store signatures in ASN.1 BIT
// STRINGs, but signature algorithms return byte strings rather than bit
// strings. This function creates a BIT STRING containing a whole number of
// bytes, with the bit order matching the DER encoding. This matches the
// encoding used by all X.509 signature algorithms.
OPENSSL_EXPORT int X509_CRL_set1_signature_value(X509_CRL *crl,
                                                 const uint8_t *sig,
                                                 size_t sig_len);


// Certificate requests.
//
// An |X509_REQ| represents a PKCS #10 certificate request (RFC 2986). These are
// also referred to as certificate signing requests or CSRs. CSRs are a common
// format used to request a certificate from a CA.
//
// Although an |X509_REQ| is a mutable object, mutating an |X509_REQ| can give
// incorrect results. Callers typically obtain |X509_REQ|s by parsing some input
// with |d2i_X509_REQ|, etc. Such objects carry information such as the
// serialized CertificationRequestInfo, which will become inconsistent when
// mutated.
//
// Instead, mutation functions should only be used when issuing new CRLs, as
// described in a later section.

// X509_REQ is an |ASN1_ITEM| whose ASN.1 type is CertificateRequest (RFC 2986)
// and C type is |X509_REQ*|.
DECLARE_ASN1_ITEM(X509_REQ)

// X509_REQ_dup returns a newly-allocated copy of |req|, or NULL on error. This
// function works by serializing the structure, so if |req| is incomplete, it
// may fail.
//
// TODO(https://crbug.com/boringssl/407): This function should be const and
// thread-safe but is currently neither in some cases, notably if |req| was
// mutated.
OPENSSL_EXPORT X509_REQ *X509_REQ_dup(X509_REQ *req);

// X509_REQ_free releases memory associated with |req|.
OPENSSL_EXPORT void X509_REQ_free(X509_REQ *req);

// d2i_X509_REQ parses up to |len| bytes from |*inp| as a DER-encoded
// CertificateRequest (RFC 2986), as described in |d2i_SAMPLE|.
OPENSSL_EXPORT X509_REQ *d2i_X509_REQ(X509_REQ **out, const uint8_t **inp,
                                      long len);

// i2d_X509_REQ marshals |req| as a CertificateRequest (RFC 2986), as described
// in |i2d_SAMPLE|.
//
// TODO(https://crbug.com/boringssl/407): This function should be const and
// thread-safe but is currently neither in some cases, notably if |req| was
// mutated.
OPENSSL_EXPORT int i2d_X509_REQ(X509_REQ *req, uint8_t **outp);


// X509_REQ_VERSION_1 is the version constant for |X509_REQ| objects. No other
// versions are defined.
#define X509_REQ_VERSION_1 0

// X509_REQ_get_version returns the numerical value of |req|'s version. This
// will always be |X509_REQ_VERSION_1| for valid CSRs. For compatibility,
// |d2i_X509_REQ| also accepts some invalid version numbers, in which case this
// function may return other values.
OPENSSL_EXPORT long X509_REQ_get_version(const X509_REQ *req);

// X509_REQ_get_subject_name returns |req|'s subject name. Note this function is
// not const-correct for legacy reasons.
OPENSSL_EXPORT X509_NAME *X509_REQ_get_subject_name(const X509_REQ *req);

// X509_REQ_get_pubkey returns |req|'s public key as an |EVP_PKEY|, or NULL if
// the public key was unsupported or could not be decoded. This function returns
// a reference to the |EVP_PKEY|. The caller must release the result with
// |EVP_PKEY_free| when done.
OPENSSL_EXPORT EVP_PKEY *X509_REQ_get_pubkey(X509_REQ *req);

// X509_REQ_get0_signature sets |*out_sig| and |*out_alg| to the signature and
// signature algorithm of |req|, respectively. Either output pointer may be NULL
// to ignore the value.
OPENSSL_EXPORT void X509_REQ_get0_signature(const X509_REQ *req,
                                            const ASN1_BIT_STRING **out_sig,
                                            const X509_ALGOR **out_alg);

// X509_REQ_get_signature_nid returns the NID corresponding to |req|'s signature
// algorithm, or |NID_undef| if the signature algorithm does not correspond to
// a known NID.
OPENSSL_EXPORT int X509_REQ_get_signature_nid(const X509_REQ *req);


// Issuing certificate requests.
//
// An |X509_REQ| object may also represent an incomplete CSR. Callers may
// construct empty |X509_REQ| objects, fill in fields individually, and finally
// sign the result. The following functions may be used for this purpose.

// X509_REQ_new returns a newly-allocated, empty |X509_REQ| object, or NULL on
// error. This object may be filled in and then signed to construct a CSR.
OPENSSL_EXPORT X509_REQ *X509_REQ_new(void);

// X509_REQ_set_version sets |req|'s version to |version|, which should be
// |X509_REQ_VERSION_1|. It returns one on success and zero on error.
//
// The only defined CSR version is |X509_REQ_VERSION_1|, so there is no need to
// call this function.
OPENSSL_EXPORT int X509_REQ_set_version(X509_REQ *req, long version);

// X509_REQ_set_subject_name sets |req|'s subject to a copy of |name|. It
// returns one on success and zero on error.
OPENSSL_EXPORT int X509_REQ_set_subject_name(X509_REQ *req, X509_NAME *name);

// X509_REQ_set_pubkey sets |req|'s public key to |pkey|. It returns one on
// success and zero on error. This function does not take ownership of |pkey|
// and internally copies and updates reference counts as needed.
OPENSSL_EXPORT int X509_REQ_set_pubkey(X509_REQ *req, EVP_PKEY *pkey);

// X509_REQ_sign signs |req| with |pkey| and replaces the signature algorithm
// and signature fields. It returns one on success and zero on error. This
// function uses digest algorithm |md|, or |pkey|'s default if NULL. Other
// signing parameters use |pkey|'s defaults. To customize them, use
// |X509_REQ_sign_ctx|.
OPENSSL_EXPORT int X509_REQ_sign(X509_REQ *req, EVP_PKEY *pkey,
                                 const EVP_MD *md);

// X509_REQ_sign_ctx signs |req| with |ctx| and replaces the signature algorithm
// and signature fields. It returns one on success and zero on error. The
// signature algorithm and parameters come from |ctx|, which must have been
// initialized with |EVP_DigestSignInit|. The caller should configure the
// corresponding |EVP_PKEY_CTX| before calling this function.
OPENSSL_EXPORT int X509_REQ_sign_ctx(X509_REQ *req, EVP_MD_CTX *ctx);

// i2d_re_X509_REQ_tbs serializes the CertificationRequestInfo (see RFC 2986)
// portion of |req|, as described in |i2d_SAMPLE|.
//
// This function re-encodes the CertificationRequestInfo and may not reflect
// |req|'s original encoding. It may be used to manually generate a signature
// for a new certificate request.
OPENSSL_EXPORT int i2d_re_X509_REQ_tbs(X509_REQ *req, uint8_t **outp);

// X509_REQ_set1_signature_algo sets |req|'s signature algorithm to |algo| and
// returns one on success or zero on error.
OPENSSL_EXPORT int X509_REQ_set1_signature_algo(X509_REQ *req,
                                                const X509_ALGOR *algo);

// X509_REQ_set1_signature_value sets |req|'s signature to a copy of the
// |sig_len| bytes pointed by |sig|. It returns one on success and zero on
// error.
//
// Due to a specification error, PKCS#10 certificate requests store signatures
// in ASN.1 BIT STRINGs, but signature algorithms return byte strings rather
// than bit strings. This function creates a BIT STRING containing a whole
// number of bytes, with the bit order matching the DER encoding. This matches
// the encoding used by all X.509 signature algorithms.
OPENSSL_EXPORT int X509_REQ_set1_signature_value(X509_REQ *req,
                                                 const uint8_t *sig,
                                                 size_t sig_len);


// Names.
//
// An |X509_NAME| represents an X.509 Name structure (RFC 5280). X.509 names are
// a complex, hierarchical structure over a collection of attributes. Each name
// is sequence of relative distinguished names (RDNs), decreasing in
// specificity. For example, the first RDN may specify the country, while the
// next RDN may specify a locality. Each RDN is, itself, a set of attributes.
// Having more than one attribute in an RDN is uncommon, but possible. Within an
// RDN, attributes have the same level in specificity. Attribute types are
// OBJECT IDENTIFIERs. This determines the ASN.1 type of the value, which is
// commonly a string but may be other types.
//
// The |X509_NAME| representation flattens this two-level structure into a
// single list of attributes. Each attribute is stored in an |X509_NAME_ENTRY|,
// with also maintains the index of the RDN it is part of, accessible via
// |X509_NAME_ENTRY_set|. This can be used to recover the two-level structure.
//
// X.509 names are largely vestigial. Historically, DNS names were parsed out of
// the subject's common name attribute, but this is deprecated and has since
// moved to the subject alternative name extension. In modern usage, X.509 names
// are primarily opaque identifiers to link a certificate with its issuer.

DEFINE_STACK_OF(X509_NAME_ENTRY)
DEFINE_STACK_OF(X509_NAME)

// X509_NAME is an |ASN1_ITEM| whose ASN.1 type is X.509 Name (RFC 5280) and C
// type is |X509_NAME*|.
DECLARE_ASN1_ITEM(X509_NAME)

// X509_NAME_new returns a new, empty |X509_NAME_new|, or NULL on
// error.
OPENSSL_EXPORT X509_NAME *X509_NAME_new(void);

// X509_NAME_free releases memory associated with |name|.
OPENSSL_EXPORT void X509_NAME_free(X509_NAME *name);

// d2i_X509_NAME parses up to |len| bytes from |*inp| as a DER-encoded X.509
// Name (RFC 5280), as described in |d2i_SAMPLE|.
OPENSSL_EXPORT X509_NAME *d2i_X509_NAME(X509_NAME **out, const uint8_t **inp,
                                        long len);

// i2d_X509_NAME marshals |in| as a DER-encoded X.509 Name (RFC 5280), as
// described in |i2d_SAMPLE|.
//
// TODO(https://crbug.com/boringssl/407): This function should be const and
// thread-safe but is currently neither in some cases, notably if |in| was
// mutated.
OPENSSL_EXPORT int i2d_X509_NAME(X509_NAME *in, uint8_t **outp);

// X509_NAME_dup returns a newly-allocated copy of |name|, or NULL on error.
//
// TODO(https://crbug.com/boringssl/407): This function should be const and
// thread-safe but is currently neither in some cases, notably if |name| was
// mutated.
OPENSSL_EXPORT X509_NAME *X509_NAME_dup(X509_NAME *name);

// X509_NAME_get0_der sets |*out_der| and |*out_der_len|
//
// Avoid this function and prefer |i2d_X509_NAME|. It is one of the reasons
// these functions are not consistently thread-safe or const-correct. Depending
// on the resolution of https://crbug.com/boringssl/407, this function may be
// removed or cause poor performance.
OPENSSL_EXPORT int X509_NAME_get0_der(X509_NAME *name, const uint8_t **out_der,
                                      size_t *out_der_len);

// X509_NAME_set makes a copy of |name|. On success, it frees |*xn|, sets |*xn|
// to the copy, and returns one. Otherwise, it returns zero.
//
// TODO(https://crbug.com/boringssl/407): This function should be const and
// thread-safe but is currently neither in some cases, notably if |name| was
// mutated.
OPENSSL_EXPORT int X509_NAME_set(X509_NAME **xn, X509_NAME *name);

// X509_NAME_entry_count returns the number of entries in |name|.
OPENSSL_EXPORT int X509_NAME_entry_count(const X509_NAME *name);

// X509_NAME_get_index_by_NID returns the zero-based index of the first
// attribute in |name| with type |nid|, or -1 if there is none. |nid| should be
// one of the |NID_*| constants. If |lastpos| is non-negative, it begins
// searching at |lastpos+1|. To search all attributes, pass in -1, not zero.
//
// Indices from this function refer to |X509_NAME|'s flattened representation.
OPENSSL_EXPORT int X509_NAME_get_index_by_NID(const X509_NAME *name, int nid,
                                              int lastpos);

// X509_NAME_get_index_by_OBJ behaves like |X509_NAME_get_index_by_NID| but
// looks for attributes with type |obj|.
OPENSSL_EXPORT int X509_NAME_get_index_by_OBJ(const X509_NAME *name,
                                              const ASN1_OBJECT *obj,
                                              int lastpos);

// X509_NAME_get_entry returns the attribute in |name| at index |loc|, or NULL
// if |loc| is out of range. |loc| is interpreted using |X509_NAME|'s flattened
// representation. This function returns a non-const pointer for OpenSSL
// compatibility, but callers should not mutate the result. Doing so will break
// internal invariants in the library.
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_get_entry(const X509_NAME *name,
                                                    int loc);

// X509_NAME_delete_entry removes and returns the attribute in |name| at index
// |loc|, or NULL if |loc| is out of range. |loc| is interpreted using
// |X509_NAME|'s flattened representation. If the attribute is found, the caller
// is responsible for releasing the result with |X509_NAME_ENTRY_free|.
//
// This function will internally update RDN indices (see |X509_NAME_ENTRY_set|)
// so they continue to be consecutive.
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_delete_entry(X509_NAME *name,
                                                       int loc);

// X509_NAME_add_entry adds a copy of |entry| to |name| and returns one on
// success or zero on error. If |loc| is -1, the entry is appended to |name|.
// Otherwise, it is inserted at index |loc|. If |set| is -1, the entry is added
// to the previous entry's RDN. If it is 0, the entry becomes a singleton RDN.
// If 1, it is added to next entry's RDN.
//
// This function will internally update RDN indices (see |X509_NAME_ENTRY_set|)
// so they continue to be consecutive.
OPENSSL_EXPORT int X509_NAME_add_entry(X509_NAME *name,
                                       const X509_NAME_ENTRY *entry, int loc,
                                       int set);

// X509_NAME_add_entry_by_OBJ adds a new entry to |name| and returns one on
// success or zero on error. The entry's attribute type is |obj|. The entry's
// attribute value is determined by |type|, |bytes|, and |len|, as in
// |X509_NAME_ENTRY_set_data|. The entry's position is determined by |loc| and
// |set| as in |X509_NAME_add_entry|.
OPENSSL_EXPORT int X509_NAME_add_entry_by_OBJ(X509_NAME *name,
                                              const ASN1_OBJECT *obj, int type,
                                              const uint8_t *bytes, int len,
                                              int loc, int set);

// X509_NAME_add_entry_by_NID behaves like |X509_NAME_add_entry_by_OBJ| but sets
// the entry's attribute type to |nid|, which should be one of the |NID_*|
// constants.
OPENSSL_EXPORT int X509_NAME_add_entry_by_NID(X509_NAME *name, int nid,
                                              int type, const uint8_t *bytes,
                                              int len, int loc, int set);

// X509_NAME_add_entry_by_txt behaves like |X509_NAME_add_entry_by_OBJ| but sets
// the entry's attribute type to |field|, which is passed to |OBJ_txt2obj|.
OPENSSL_EXPORT int X509_NAME_add_entry_by_txt(X509_NAME *name,
                                              const char *field, int type,
                                              const uint8_t *bytes, int len,
                                              int loc, int set);

// X509_NAME_ENTRY is an |ASN1_ITEM| whose ASN.1 type is AttributeTypeAndValue
// (RFC 5280) and C type is |X509_NAME_ENTRY*|.
DECLARE_ASN1_ITEM(X509_NAME_ENTRY)

// X509_NAME_ENTRY_new returns a new, empty |X509_NAME_ENTRY_new|, or NULL on
// error.
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_ENTRY_new(void);

// X509_NAME_ENTRY_free releases memory associated with |entry|.
OPENSSL_EXPORT void X509_NAME_ENTRY_free(X509_NAME_ENTRY *entry);

// d2i_X509_NAME_ENTRY parses up to |len| bytes from |*inp| as a DER-encoded
// AttributeTypeAndValue (RFC 5280), as described in |d2i_SAMPLE|.
OPENSSL_EXPORT X509_NAME_ENTRY *d2i_X509_NAME_ENTRY(X509_NAME_ENTRY **out,
                                                    const uint8_t **inp,
                                                    long len);

// i2d_X509_NAME_ENTRY marshals |in| as a DER-encoded AttributeTypeAndValue (RFC
// 5280), as described in |i2d_SAMPLE|.
OPENSSL_EXPORT int i2d_X509_NAME_ENTRY(const X509_NAME_ENTRY *in,
                                       uint8_t **outp);

// X509_NAME_ENTRY_dup returns a newly-allocated copy of |entry|, or NULL on
// error.
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_ENTRY_dup(
    const X509_NAME_ENTRY *entry);

// X509_NAME_ENTRY_get_object returns |entry|'s attribute type. This function
// returns a non-const pointer for OpenSSL compatibility, but callers should not
// mutate the result. Doing so will break internal invariants in the library.
OPENSSL_EXPORT ASN1_OBJECT *X509_NAME_ENTRY_get_object(
    const X509_NAME_ENTRY *entry);

// X509_NAME_ENTRY_set_object sets |entry|'s attribute type to |obj|. It returns
// one on success and zero on error.
OPENSSL_EXPORT int X509_NAME_ENTRY_set_object(X509_NAME_ENTRY *entry,
                                              const ASN1_OBJECT *obj);

// X509_NAME_ENTRY_get_data returns |entry|'s attribute value, represented as an
// |ASN1_STRING|. This value may have any ASN.1 type, so callers must check the
// type before interpreting the contents. This function returns a non-const
// pointer for OpenSSL compatibility, but callers should not mutate the result.
// Doing so will break internal invariants in the library.
//
// TODO(https://crbug.com/boringssl/412): Although the spec says any ASN.1 type
// is allowed, we currently only allow an ad-hoc set of types. Additionally, it
// is unclear if some types can even be represented by this function.
OPENSSL_EXPORT ASN1_STRING *X509_NAME_ENTRY_get_data(
    const X509_NAME_ENTRY *entry);

// X509_NAME_ENTRY_set_data sets |entry|'s value to |len| bytes from |bytes|. It
// returns one on success and zero on error. If |len| is -1, |bytes| must be a
// NUL-terminated C string and the length is determined by |strlen|. |bytes| is
// converted to an ASN.1 type as follows:
//
// If |type| is a |MBSTRING_*| constant, the value is an ASN.1 string. The
// string is determined by decoding |bytes| in the encoding specified by |type|,
// and then re-encoding it in a form appropriate for |entry|'s attribute type.
// See |ASN1_STRING_set_by_NID| for details.
//
// Otherwise, the value is an |ASN1_STRING| with type |type| and value |bytes|.
// See |ASN1_STRING| for how to format ASN.1 types as an |ASN1_STRING|. If
// |type| is |V_ASN1_UNDEF| the previous |ASN1_STRING| type is reused.
OPENSSL_EXPORT int X509_NAME_ENTRY_set_data(X509_NAME_ENTRY *entry, int type,
                                            const uint8_t *bytes, int len);

// X509_NAME_ENTRY_set returns the zero-based index of the RDN which contains
// |entry|. Consecutive entries with the same index are part of the same RDN.
OPENSSL_EXPORT int X509_NAME_ENTRY_set(const X509_NAME_ENTRY *entry);

// X509_NAME_ENTRY_create_by_OBJ creates a new |X509_NAME_ENTRY| with attribute
// type |obj|. The attribute value is determined from |type|, |bytes|, and |len|
// as in |X509_NAME_ENTRY_set_data|. It returns the |X509_NAME_ENTRY| on success
// and NULL on error.
//
// If |out| is non-NULL and |*out| is NULL, it additionally sets |*out| to the
// result on success. If both |out| and |*out| are non-NULL, it updates the
// object at |*out| instead of allocating a new one.
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_ENTRY_create_by_OBJ(
    X509_NAME_ENTRY **out, const ASN1_OBJECT *obj, int type,
    const uint8_t *bytes, int len);

// X509_NAME_ENTRY_create_by_NID behaves like |X509_NAME_ENTRY_create_by_OBJ|
// except the attribute type is |nid|, which should be one of the |NID_*|
// constants.
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_ENTRY_create_by_NID(
    X509_NAME_ENTRY **out, int nid, int type, const uint8_t *bytes, int len);

// X509_NAME_ENTRY_create_by_txt behaves like |X509_NAME_ENTRY_create_by_OBJ|
// except the attribute type is |field|, which is passed to |OBJ_txt2obj|.
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_ENTRY_create_by_txt(
    X509_NAME_ENTRY **out, const char *field, int type, const uint8_t *bytes,
    int len);


// Extensions.
//
// X.509 certificates and CRLs may contain a list of extensions (RFC 5280).
// Extensions have a type, specified by an object identifier (|ASN1_OBJECT|) and
// a byte string value, which should a DER-encoded structure whose type is
// determined by the extension type. This library represents extensions with the
// |X509_EXTENSION| type.

// X509_EXTENSION is an |ASN1_ITEM| whose ASN.1 type is X.509 Extension (RFC
// 5280) and C type is |X509_EXTENSION*|.
DECLARE_ASN1_ITEM(X509_EXTENSION)

// X509_EXTENSION_new returns a newly-allocated, empty |X509_EXTENSION| object
// or NULL on error.
OPENSSL_EXPORT X509_EXTENSION *X509_EXTENSION_new(void);

// X509_EXTENSION_free releases memory associated with |ex|.
OPENSSL_EXPORT void X509_EXTENSION_free(X509_EXTENSION *ex);

// d2i_X509_EXTENSION parses up to |len| bytes from |*inp| as a DER-encoded
// X.509 Extension (RFC 5280), as described in |d2i_SAMPLE|.
OPENSSL_EXPORT X509_EXTENSION *d2i_X509_EXTENSION(X509_EXTENSION **out,
                                                  const uint8_t **inp,
                                                  long len);

// i2d_X509_EXTENSION marshals |alg| as a DER-encoded X.509 Extension (RFC
// 5280), as described in |i2d_SAMPLE|.
OPENSSL_EXPORT int i2d_X509_EXTENSION(const X509_EXTENSION *alg,
                                      uint8_t **outp);

// X509_EXTENSION_dup returns a newly-allocated copy of |ex|, or NULL on error.
// This function works by serializing the structure, so if |ex| is incomplete,
// it may fail.
OPENSSL_EXPORT X509_EXTENSION *X509_EXTENSION_dup(const X509_EXTENSION *ex);

// X509_EXTENSION_create_by_NID creates a new |X509_EXTENSION| with type |nid|,
// value |data|, and critical bit |crit|. It returns an |X509_EXTENSION| on
// success, and NULL on error. |nid| should be a |NID_*| constant.
//
// If |ex| and |*ex| are both non-NULL, |*ex| is used to hold the result,
// otherwise a new object is allocated. If |ex| is non-NULL and |*ex| is NULL,
// the function sets |*ex| to point to the newly allocated result, in addition
// to returning the result.
OPENSSL_EXPORT X509_EXTENSION *X509_EXTENSION_create_by_NID(
    X509_EXTENSION **ex, int nid, int crit, const ASN1_OCTET_STRING *data);

// X509_EXTENSION_create_by_OBJ behaves like |X509_EXTENSION_create_by_NID|, but
// the extension type is determined by an |ASN1_OBJECT|.
OPENSSL_EXPORT X509_EXTENSION *X509_EXTENSION_create_by_OBJ(
    X509_EXTENSION **ex, const ASN1_OBJECT *obj, int crit,
    const ASN1_OCTET_STRING *data);

// X509_EXTENSION_get_object returns |ex|'s extension type. This function
// returns a non-const pointer for OpenSSL compatibility, but callers should not
// mutate the result.
OPENSSL_EXPORT ASN1_OBJECT *X509_EXTENSION_get_object(const X509_EXTENSION *ex);

// X509_EXTENSION_get_data returns |ne|'s extension value. This function returns
// a non-const pointer for OpenSSL compatibility, but callers should not mutate
// the result.
OPENSSL_EXPORT ASN1_OCTET_STRING *X509_EXTENSION_get_data(
    const X509_EXTENSION *ne);

// X509_EXTENSION_get_critical returns one if |ex| is critical and zero
// otherwise.
OPENSSL_EXPORT int X509_EXTENSION_get_critical(const X509_EXTENSION *ex);

// X509_EXTENSION_set_object sets |ex|'s extension type to |obj|. It returns one
// on success and zero on error.
OPENSSL_EXPORT int X509_EXTENSION_set_object(X509_EXTENSION *ex,
                                             const ASN1_OBJECT *obj);

// X509_EXTENSION_set_critical sets |ex| to critical if |crit| is non-zero and
// to non-critical if |crit| is zero.
OPENSSL_EXPORT int X509_EXTENSION_set_critical(X509_EXTENSION *ex, int crit);

// X509_EXTENSION_set_data set's |ex|'s extension value to a copy of |data|. It
// returns one on success and zero on error.
OPENSSL_EXPORT int X509_EXTENSION_set_data(X509_EXTENSION *ex,
                                           const ASN1_OCTET_STRING *data);


// Extension lists.
//
// The following functions manipulate lists of extensions. Most of them have
// corresponding functions on the containing |X509|, |X509_CRL|, or
// |X509_REVOKED|.

DEFINE_STACK_OF(X509_EXTENSION)
typedef STACK_OF(X509_EXTENSION) X509_EXTENSIONS;

// X509_EXTENSIONS is an |ASN1_ITEM| whose ASN.1 type is SEQUENCE of Extension
// (RFC 5280) and C type is |STACK_OF(X509_EXTENSION)*|.
DECLARE_ASN1_ITEM(X509_EXTENSIONS)

// d2i_X509_EXTENSIONS parses up to |len| bytes from |*inp| as a DER-encoded
// SEQUENCE OF Extension (RFC 5280), as described in |d2i_SAMPLE|.
OPENSSL_EXPORT X509_EXTENSIONS *d2i_X509_EXTENSIONS(X509_EXTENSIONS **out,
                                                    const uint8_t **inp,
                                                    long len);

// i2d_X509_EXTENSIONS marshals |alg| as a DER-encoded SEQUENCE OF Extension
// (RFC 5280), as described in |i2d_SAMPLE|.
OPENSSL_EXPORT int i2d_X509_EXTENSIONS(const X509_EXTENSIONS *alg,
                                       uint8_t **outp);

// X509v3_get_ext_count returns the number of extensions in |x|.
OPENSSL_EXPORT int X509v3_get_ext_count(const STACK_OF(X509_EXTENSION) *x);

// X509v3_get_ext_by_NID returns the index of the first extension in |x| with
// type |nid|, or a negative number if not found. If found, callers can use
// |X509v3_get_ext| to look up the extension by index.
//
// If |lastpos| is non-negative, it begins searching at |lastpos| + 1. Callers
// can thus loop over all matching extensions by first passing -1 and then
// passing the previously-returned value until no match is returned.
OPENSSL_EXPORT int X509v3_get_ext_by_NID(const STACK_OF(X509_EXTENSION) *x,
                                         int nid, int lastpos);

// X509v3_get_ext_by_OBJ behaves like |X509v3_get_ext_by_NID| but looks for
// extensions matching |obj|.
OPENSSL_EXPORT int X509v3_get_ext_by_OBJ(const STACK_OF(X509_EXTENSION) *x,
                                         const ASN1_OBJECT *obj, int lastpos);

// X509v3_get_ext_by_critical returns the index of the first extension in |x|
// whose critical bit matches |crit|, or a negative number if no such extension
// was found.
//
// If |lastpos| is non-negative, it begins searching at |lastpos| + 1. Callers
// can thus loop over all matching extensions by first passing -1 and then
// passing the previously-returned value until no match is returned.
OPENSSL_EXPORT int X509v3_get_ext_by_critical(const STACK_OF(X509_EXTENSION) *x,
                                              int crit, int lastpos);

// X509v3_get_ext returns the extension in |x| at index |loc|, or NULL if |loc|
// is out of bounds. This function returns a non-const pointer for OpenSSL
// compatibility, but callers should not mutate the result.
OPENSSL_EXPORT X509_EXTENSION *X509v3_get_ext(const STACK_OF(X509_EXTENSION) *x,
                                              int loc);

// X509v3_delete_ext removes the extension in |x| at index |loc| and returns the
// removed extension, or NULL if |loc| was out of bounds. If an extension was
// returned, the caller must release it with |X509_EXTENSION_free|.
OPENSSL_EXPORT X509_EXTENSION *X509v3_delete_ext(STACK_OF(X509_EXTENSION) *x,
                                                 int loc);

// X509v3_add_ext adds a copy of |ex| to the extension list in |*x|. If |*x| is
// NULL, it allocates a new |STACK_OF(X509_EXTENSION)| to hold the copy and sets
// |*x| to the new list. It returns |*x| on success and NULL on error. The
// caller retains ownership of |ex| and can release it independently of |*x|.
//
// The new extension is inserted at index |loc|, shifting extensions to the
// right. If |loc| is -1 or out of bounds, the new extension is appended to the
// list.
OPENSSL_EXPORT STACK_OF(X509_EXTENSION) *X509v3_add_ext(
    STACK_OF(X509_EXTENSION) **x, const X509_EXTENSION *ex, int loc);


// Algorithm identifiers.
//
// An |X509_ALGOR| represents an AlgorithmIdentifier structure, used in X.509
// to represent signature algorithms and public key algorithms.

DEFINE_STACK_OF(X509_ALGOR)

// X509_ALGOR is an |ASN1_ITEM| whose ASN.1 type is AlgorithmIdentifier and C
// type is |X509_ALGOR*|.
DECLARE_ASN1_ITEM(X509_ALGOR)

// X509_ALGOR_new returns a newly-allocated, empty |X509_ALGOR| object, or NULL
// on error.
OPENSSL_EXPORT X509_ALGOR *X509_ALGOR_new(void);

// X509_ALGOR_dup returns a newly-allocated copy of |alg|, or NULL on error.
// This function works by serializing the structure, so if |alg| is incomplete,
// it may fail.
OPENSSL_EXPORT X509_ALGOR *X509_ALGOR_dup(const X509_ALGOR *alg);

// X509_ALGOR_free releases memory associated with |alg|.
OPENSSL_EXPORT void X509_ALGOR_free(X509_ALGOR *alg);

// d2i_X509_ALGOR parses up to |len| bytes from |*inp| as a DER-encoded
// AlgorithmIdentifier, as described in |d2i_SAMPLE|.
OPENSSL_EXPORT X509_ALGOR *d2i_X509_ALGOR(X509_ALGOR **out, const uint8_t **inp,
                                          long len);

// i2d_X509_ALGOR marshals |alg| as a DER-encoded AlgorithmIdentifier, as
// described in |i2d_SAMPLE|.
OPENSSL_EXPORT int i2d_X509_ALGOR(const X509_ALGOR *alg, uint8_t **outp);

// X509_ALGOR_set0 sets |alg| to an AlgorithmIdentifier with algorithm |obj| and
// parameter determined by |param_type| and |param_value|. It returns one on
// success and zero on error. This function takes ownership of |obj| and
// |param_value| on success.
//
// If |param_type| is |V_ASN1_UNDEF|, the parameter is omitted. If |param_type|
// is zero, the parameter is left unchanged. Otherwise, |param_type| and
// |param_value| are interpreted as in |ASN1_TYPE_set|.
//
// Note omitting the parameter (|V_ASN1_UNDEF|) and encoding an explicit NULL
// value (|V_ASN1_NULL|) are different. Some algorithms require one and some the
// other. Consult the relevant specification before calling this function. The
// correct parameter for an RSASSA-PKCS1-v1_5 signature is |V_ASN1_NULL|. The
// correct one for an ECDSA or Ed25519 signature is |V_ASN1_UNDEF|.
OPENSSL_EXPORT int X509_ALGOR_set0(X509_ALGOR *alg, ASN1_OBJECT *obj,
                                   int param_type, void *param_value);

// X509_ALGOR_get0 sets |*out_obj| to the |alg|'s algorithm. If |alg|'s
// parameter is omitted, it sets |*out_param_type| and |*out_param_value| to
// |V_ASN1_UNDEF| and NULL. Otherwise, it sets |*out_param_type| and
// |*out_param_value| to the parameter, using the same representation as
// |ASN1_TYPE_set0|. See |ASN1_TYPE_set0| and |ASN1_TYPE| for details.
//
// Callers that require the parameter in serialized form should, after checking
// for |V_ASN1_UNDEF|, use |ASN1_TYPE_set1| and |d2i_ASN1_TYPE|, rather than
// inspecting |*out_param_value|.
//
// Each of |out_obj|, |out_param_type|, and |out_param_value| may be NULL to
// ignore the output. If |out_param_type| is NULL, |out_param_value| is ignored.
//
// WARNING: If |*out_param_type| is set to |V_ASN1_UNDEF|, OpenSSL and older
// revisions of BoringSSL leave |*out_param_value| unset rather than setting it
// to NULL. Callers that support both OpenSSL and BoringSSL should not assume
// |*out_param_value| is uniformly initialized.
OPENSSL_EXPORT void X509_ALGOR_get0(const ASN1_OBJECT **out_obj,
                                    int *out_param_type,
                                    const void **out_param_value,
                                    const X509_ALGOR *alg);

// X509_ALGOR_set_md sets |alg| to the hash function |md|. Note this
// AlgorithmIdentifier represents the hash function itself, not a signature
// algorithm that uses |md|.
OPENSSL_EXPORT void X509_ALGOR_set_md(X509_ALGOR *alg, const EVP_MD *md);

// X509_ALGOR_cmp returns zero if |a| and |b| are equal, and some non-zero value
// otherwise. Note this function can only be used for equality checks, not an
// ordering.
OPENSSL_EXPORT int X509_ALGOR_cmp(const X509_ALGOR *a, const X509_ALGOR *b);


// Printing functions.
//
// The following functions output human-readable representations of
// X.509-related structures. They should only be used for debugging or logging
// and not parsed programmatically.

// X509_signature_dump writes a human-readable representation of |sig| to |bio|,
// indented with |indent| spaces. It returns one on success and zero on error.
OPENSSL_EXPORT int X509_signature_dump(BIO *bio, const ASN1_STRING *sig,
                                       int indent);

// X509_signature_print writes a human-readable representation of |alg| and
// |sig| to |bio|. It returns one on success and zero on error.
OPENSSL_EXPORT int X509_signature_print(BIO *bio, const X509_ALGOR *alg,
                                        const ASN1_STRING *sig);


// Convenience functions.

// X509_pubkey_digest hashes the contents of the BIT STRING in |x509|'s
// subjectPublicKeyInfo field with |md| and writes the result to |out|.
// |EVP_MD_CTX_size| bytes are written, which is at most |EVP_MAX_MD_SIZE|. If
// |out_len| is not NULL, |*out_len| is set to the number of bytes written. This
// function returns one on success and zero on error.
//
// This hash omits the BIT STRING tag, length, and number of unused bits. It
// also omits the AlgorithmIdentifier which describes the key type. It
// corresponds to the OCSP KeyHash definition and is not suitable for other
// purposes.
OPENSSL_EXPORT int X509_pubkey_digest(const X509 *x509, const EVP_MD *md,
                                      uint8_t *out, unsigned *out_len);

// X509_digest hashes |x509|'s DER encoding with |md| and writes the result to
// |out|. |EVP_MD_CTX_size| bytes are written, which is at most
// |EVP_MAX_MD_SIZE|. If |out_len| is not NULL, |*out_len| is set to the number
// of bytes written. This function returns one on success and zero on error.
// Note this digest covers the entire certificate, not just the signed portion.
OPENSSL_EXPORT int X509_digest(const X509 *x509, const EVP_MD *md, uint8_t *out,
                               unsigned *out_len);

// X509_CRL_digest hashes |crl|'s DER encoding with |md| and writes the result
// to |out|. |EVP_MD_CTX_size| bytes are written, which is at most
// |EVP_MAX_MD_SIZE|. If |out_len| is not NULL, |*out_len| is set to the number
// of bytes written. This function returns one on success and zero on error.
// Note this digest covers the entire CRL, not just the signed portion.
OPENSSL_EXPORT int X509_CRL_digest(const X509_CRL *crl, const EVP_MD *md,
                                   uint8_t *out, unsigned *out_len);

// X509_REQ_digest hashes |req|'s DER encoding with |md| and writes the result
// to |out|. |EVP_MD_CTX_size| bytes are written, which is at most
// |EVP_MAX_MD_SIZE|. If |out_len| is not NULL, |*out_len| is set to the number
// of bytes written. This function returns one on success and zero on error.
// Note this digest covers the entire certificate request, not just the signed
// portion.
OPENSSL_EXPORT int X509_REQ_digest(const X509_REQ *req, const EVP_MD *md,
                                   uint8_t *out, unsigned *out_len);

// X509_NAME_digest hashes |name|'s DER encoding with |md| and writes the result
// to |out|. |EVP_MD_CTX_size| bytes are written, which is at most
// |EVP_MAX_MD_SIZE|. If |out_len| is not NULL, |*out_len| is set to the number
// of bytes written. This function returns one on success and zero on error.
OPENSSL_EXPORT int X509_NAME_digest(const X509_NAME *name, const EVP_MD *md,
                                    uint8_t *out, unsigned *out_len);

// The following functions behave like the corresponding unsuffixed |d2i_*|
// functions, but read the result from |bp| instead. Callers using these
// functions with memory |BIO|s to parse structures already in memory should use
// |d2i_*| instead.
OPENSSL_EXPORT X509 *d2i_X509_bio(BIO *bp, X509 **x509);
OPENSSL_EXPORT X509_CRL *d2i_X509_CRL_bio(BIO *bp, X509_CRL **crl);
OPENSSL_EXPORT X509_REQ *d2i_X509_REQ_bio(BIO *bp, X509_REQ **req);
OPENSSL_EXPORT RSA *d2i_RSAPrivateKey_bio(BIO *bp, RSA **rsa);
OPENSSL_EXPORT RSA *d2i_RSAPublicKey_bio(BIO *bp, RSA **rsa);
OPENSSL_EXPORT RSA *d2i_RSA_PUBKEY_bio(BIO *bp, RSA **rsa);
OPENSSL_EXPORT DSA *d2i_DSA_PUBKEY_bio(BIO *bp, DSA **dsa);
OPENSSL_EXPORT DSA *d2i_DSAPrivateKey_bio(BIO *bp, DSA **dsa);
OPENSSL_EXPORT EC_KEY *d2i_EC_PUBKEY_bio(BIO *bp, EC_KEY **eckey);
OPENSSL_EXPORT EC_KEY *d2i_ECPrivateKey_bio(BIO *bp, EC_KEY **eckey);
OPENSSL_EXPORT X509_SIG *d2i_PKCS8_bio(BIO *bp, X509_SIG **p8);
OPENSSL_EXPORT PKCS8_PRIV_KEY_INFO *d2i_PKCS8_PRIV_KEY_INFO_bio(
    BIO *bp, PKCS8_PRIV_KEY_INFO **p8inf);
OPENSSL_EXPORT EVP_PKEY *d2i_PUBKEY_bio(BIO *bp, EVP_PKEY **a);
OPENSSL_EXPORT DH *d2i_DHparams_bio(BIO *bp, DH **dh);

// d2i_PrivateKey_bio behaves like |d2i_AutoPrivateKey|, but reads from |bp|
// instead.
OPENSSL_EXPORT EVP_PKEY *d2i_PrivateKey_bio(BIO *bp, EVP_PKEY **a);

// The following functions behave like the corresponding unsuffixed |i2d_*|
// functions, but write the result to |bp|. They return one on success and zero
// on error. Callers using them with memory |BIO|s to encode structures to
// memory should use |i2d_*| directly instead.
OPENSSL_EXPORT int i2d_X509_bio(BIO *bp, X509 *x509);
OPENSSL_EXPORT int i2d_X509_CRL_bio(BIO *bp, X509_CRL *crl);
OPENSSL_EXPORT int i2d_X509_REQ_bio(BIO *bp, X509_REQ *req);
OPENSSL_EXPORT int i2d_RSAPrivateKey_bio(BIO *bp, RSA *rsa);
OPENSSL_EXPORT int i2d_RSAPublicKey_bio(BIO *bp, RSA *rsa);
OPENSSL_EXPORT int i2d_RSA_PUBKEY_bio(BIO *bp, RSA *rsa);
OPENSSL_EXPORT int i2d_DSA_PUBKEY_bio(BIO *bp, DSA *dsa);
OPENSSL_EXPORT int i2d_DSAPrivateKey_bio(BIO *bp, DSA *dsa);
OPENSSL_EXPORT int i2d_EC_PUBKEY_bio(BIO *bp, EC_KEY *eckey);
OPENSSL_EXPORT int i2d_ECPrivateKey_bio(BIO *bp, EC_KEY *eckey);
OPENSSL_EXPORT int i2d_PKCS8_bio(BIO *bp, X509_SIG *p8);
OPENSSL_EXPORT int i2d_PKCS8_PRIV_KEY_INFO_bio(BIO *bp,
                                               PKCS8_PRIV_KEY_INFO *p8inf);
OPENSSL_EXPORT int i2d_PrivateKey_bio(BIO *bp, EVP_PKEY *pkey);
OPENSSL_EXPORT int i2d_PUBKEY_bio(BIO *bp, EVP_PKEY *pkey);
OPENSSL_EXPORT int i2d_DHparams_bio(BIO *bp, const DH *dh);

// i2d_PKCS8PrivateKeyInfo_bio encodes |key| as a PKCS#8 PrivateKeyInfo
// structure (see |EVP_marshal_private_key|) and writes the result to |bp|. It
// returns one on success and zero on error.
OPENSSL_EXPORT int i2d_PKCS8PrivateKeyInfo_bio(BIO *bp, EVP_PKEY *key);

// The following functions behave like the corresponding |d2i_*_bio| functions,
// but read from |fp| instead.
OPENSSL_EXPORT X509 *d2i_X509_fp(FILE *fp, X509 **x509);
OPENSSL_EXPORT X509_CRL *d2i_X509_CRL_fp(FILE *fp, X509_CRL **crl);
OPENSSL_EXPORT X509_REQ *d2i_X509_REQ_fp(FILE *fp, X509_REQ **req);
OPENSSL_EXPORT RSA *d2i_RSAPrivateKey_fp(FILE *fp, RSA **rsa);
OPENSSL_EXPORT RSA *d2i_RSAPublicKey_fp(FILE *fp, RSA **rsa);
OPENSSL_EXPORT RSA *d2i_RSA_PUBKEY_fp(FILE *fp, RSA **rsa);
OPENSSL_EXPORT DSA *d2i_DSA_PUBKEY_fp(FILE *fp, DSA **dsa);
OPENSSL_EXPORT DSA *d2i_DSAPrivateKey_fp(FILE *fp, DSA **dsa);
OPENSSL_EXPORT EC_KEY *d2i_EC_PUBKEY_fp(FILE *fp, EC_KEY **eckey);
OPENSSL_EXPORT EC_KEY *d2i_ECPrivateKey_fp(FILE *fp, EC_KEY **eckey);
OPENSSL_EXPORT X509_SIG *d2i_PKCS8_fp(FILE *fp, X509_SIG **p8);
OPENSSL_EXPORT PKCS8_PRIV_KEY_INFO *d2i_PKCS8_PRIV_KEY_INFO_fp(
    FILE *fp, PKCS8_PRIV_KEY_INFO **p8inf);
OPENSSL_EXPORT EVP_PKEY *d2i_PrivateKey_fp(FILE *fp, EVP_PKEY **a);
OPENSSL_EXPORT EVP_PKEY *d2i_PUBKEY_fp(FILE *fp, EVP_PKEY **a);

// The following functions behave like the corresponding |i2d_*_bio| functions,
// but write to |fp| instead.
OPENSSL_EXPORT int i2d_X509_fp(FILE *fp, X509 *x509);
OPENSSL_EXPORT int i2d_X509_CRL_fp(FILE *fp, X509_CRL *crl);
OPENSSL_EXPORT int i2d_X509_REQ_fp(FILE *fp, X509_REQ *req);
OPENSSL_EXPORT int i2d_RSAPrivateKey_fp(FILE *fp, RSA *rsa);
OPENSSL_EXPORT int i2d_RSAPublicKey_fp(FILE *fp, RSA *rsa);
OPENSSL_EXPORT int i2d_RSA_PUBKEY_fp(FILE *fp, RSA *rsa);
OPENSSL_EXPORT int i2d_DSA_PUBKEY_fp(FILE *fp, DSA *dsa);
OPENSSL_EXPORT int i2d_DSAPrivateKey_fp(FILE *fp, DSA *dsa);
OPENSSL_EXPORT int i2d_EC_PUBKEY_fp(FILE *fp, EC_KEY *eckey);
OPENSSL_EXPORT int i2d_ECPrivateKey_fp(FILE *fp, EC_KEY *eckey);
OPENSSL_EXPORT int i2d_PKCS8_fp(FILE *fp, X509_SIG *p8);
OPENSSL_EXPORT int i2d_PKCS8_PRIV_KEY_INFO_fp(FILE *fp,
                                              PKCS8_PRIV_KEY_INFO *p8inf);
OPENSSL_EXPORT int i2d_PKCS8PrivateKeyInfo_fp(FILE *fp, EVP_PKEY *key);
OPENSSL_EXPORT int i2d_PrivateKey_fp(FILE *fp, EVP_PKEY *pkey);
OPENSSL_EXPORT int i2d_PUBKEY_fp(FILE *fp, EVP_PKEY *pkey);

// X509_find_by_issuer_and_serial returns the first |X509| in |sk| whose issuer
// and serial are |name| and |serial|, respectively. If no match is found, it
// returns NULL.
OPENSSL_EXPORT X509 *X509_find_by_issuer_and_serial(const STACK_OF(X509) *sk,
                                                    X509_NAME *name,
                                                    const ASN1_INTEGER *serial);

// X509_find_by_subject returns the first |X509| in |sk| whose subject is
// |name|. If no match is found, it returns NULL.
OPENSSL_EXPORT X509 *X509_find_by_subject(const STACK_OF(X509) *sk,
                                          X509_NAME *name);


// ex_data functions.
//
// See |ex_data.h| for details.

OPENSSL_EXPORT int X509_get_ex_new_index(long argl, void *argp,
                                         CRYPTO_EX_unused *unused,
                                         CRYPTO_EX_dup *dup_unused,
                                         CRYPTO_EX_free *free_func);
OPENSSL_EXPORT int X509_set_ex_data(X509 *r, int idx, void *arg);
OPENSSL_EXPORT void *X509_get_ex_data(X509 *r, int idx);

OPENSSL_EXPORT int X509_STORE_CTX_get_ex_new_index(long argl, void *argp,
                                                   CRYPTO_EX_unused *unused,
                                                   CRYPTO_EX_dup *dup_unused,
                                                   CRYPTO_EX_free *free_func);
OPENSSL_EXPORT int X509_STORE_CTX_set_ex_data(X509_STORE_CTX *ctx, int idx,
                                              void *data);
OPENSSL_EXPORT void *X509_STORE_CTX_get_ex_data(X509_STORE_CTX *ctx, int idx);


// Deprecated functions.

// X509_get_notBefore returns |x509|'s notBefore time. Note this function is not
// const-correct for legacy reasons. Use |X509_get0_notBefore| or
// |X509_getm_notBefore| instead.
OPENSSL_EXPORT ASN1_TIME *X509_get_notBefore(const X509 *x509);

// X509_get_notAfter returns |x509|'s notAfter time. Note this function is not
// const-correct for legacy reasons. Use |X509_get0_notAfter| or
// |X509_getm_notAfter| instead.
OPENSSL_EXPORT ASN1_TIME *X509_get_notAfter(const X509 *x509);

// X509_set_notBefore calls |X509_set1_notBefore|. Use |X509_set1_notBefore|
// instead.
OPENSSL_EXPORT int X509_set_notBefore(X509 *x509, const ASN1_TIME *tm);

// X509_set_notAfter calls |X509_set1_notAfter|. Use |X509_set1_notAfter|
// instead.
OPENSSL_EXPORT int X509_set_notAfter(X509 *x509, const ASN1_TIME *tm);

// X509_CRL_get_lastUpdate returns a mutable pointer to |crl|'s thisUpdate time.
// The OpenSSL API refers to this field as lastUpdate.
//
// Use |X509_CRL_get0_lastUpdate| or |X509_CRL_set1_lastUpdate| instead.
OPENSSL_EXPORT ASN1_TIME *X509_CRL_get_lastUpdate(X509_CRL *crl);

// X509_CRL_get_nextUpdate returns a mutable pointer to |crl|'s nextUpdate time,
// or NULL if |crl| has none. Use |X509_CRL_get0_nextUpdate| or
// |X509_CRL_set1_nextUpdate| instead.
OPENSSL_EXPORT ASN1_TIME *X509_CRL_get_nextUpdate(X509_CRL *crl);

// X509_extract_key is a legacy alias to |X509_get_pubkey|. Use
// |X509_get_pubkey| instead.
#define X509_extract_key(x) X509_get_pubkey(x)

// X509_REQ_extract_key is a legacy alias for |X509_REQ_get_pubkey|.
#define X509_REQ_extract_key(a) X509_REQ_get_pubkey(a)

// X509_name_cmp is a legacy alias for |X509_NAME_cmp|.
#define X509_name_cmp(a, b) X509_NAME_cmp((a), (b))

// The following symbols are deprecated aliases to |X509_CRL_set1_*|.
#define X509_CRL_set_lastUpdate X509_CRL_set1_lastUpdate
#define X509_CRL_set_nextUpdate X509_CRL_set1_nextUpdate

// X509_get_serialNumber returns a mutable pointer to |x509|'s serial number.
// Prefer |X509_get0_serialNumber|.
OPENSSL_EXPORT ASN1_INTEGER *X509_get_serialNumber(X509 *x509);

// X509_NAME_get_text_by_OBJ finds the first attribute with type |obj| in
// |name|. If found, it ignores the value's ASN.1 type, writes the raw
// |ASN1_STRING| representation to |buf|, followed by a NUL byte, and
// returns the number of bytes in output, excluding the NUL byte.
//
// This function writes at most |len| bytes, including the NUL byte. If |len| is
// not large enough, it silently truncates the output to fit. If |buf| is NULL,
// it instead writes enough and returns the number of bytes in the output,
// excluding the NUL byte.
//
// WARNING: Do not use this function. It does not return enough information for
// the caller to correctly interpret its output. The attribute value may be of
// any type, including one of several ASN.1 string encodings, but this function
// only outputs the raw |ASN1_STRING| representation. See
// https://crbug.com/boringssl/436.
OPENSSL_EXPORT int X509_NAME_get_text_by_OBJ(const X509_NAME *name,
                                             const ASN1_OBJECT *obj, char *buf,
                                             int len);

// X509_NAME_get_text_by_NID behaves like |X509_NAME_get_text_by_OBJ| except it
// finds an attribute of type |nid|, which should be one of the |NID_*|
// constants.
OPENSSL_EXPORT int X509_NAME_get_text_by_NID(const X509_NAME *name, int nid,
                                             char *buf, int len);


// Private structures.

struct X509_algor_st {
  ASN1_OBJECT *algorithm;
  ASN1_TYPE *parameter;
} /* X509_ALGOR */;


// Functions below this point have not yet been organized into sections.

#define X509_FILETYPE_PEM 1
#define X509_FILETYPE_ASN1 2
#define X509_FILETYPE_DEFAULT 3

#define X509v3_KU_DIGITAL_SIGNATURE 0x0080
#define X509v3_KU_NON_REPUDIATION 0x0040
#define X509v3_KU_KEY_ENCIPHERMENT 0x0020
#define X509v3_KU_DATA_ENCIPHERMENT 0x0010
#define X509v3_KU_KEY_AGREEMENT 0x0008
#define X509v3_KU_KEY_CERT_SIGN 0x0004
#define X509v3_KU_CRL_SIGN 0x0002
#define X509v3_KU_ENCIPHER_ONLY 0x0001
#define X509v3_KU_DECIPHER_ONLY 0x8000
#define X509v3_KU_UNDEF 0xffff

DEFINE_STACK_OF(X509_ATTRIBUTE)

// This stuff is certificate "auxiliary info"
// it contains details which are useful in certificate
// stores and databases. When used this is tagged onto
// the end of the certificate itself

DECLARE_STACK_OF(DIST_POINT)
DECLARE_STACK_OF(GENERAL_NAME)

// This is used for a table of trust checking functions

struct x509_trust_st {
  int trust;
  int flags;
  int (*check_trust)(struct x509_trust_st *, X509 *, int);
  char *name;
  int arg1;
  void *arg2;
} /* X509_TRUST */;

DEFINE_STACK_OF(X509_TRUST)

// standard trust ids

#define X509_TRUST_DEFAULT (-1)  // Only valid in purpose settings

#define X509_TRUST_COMPAT 1
#define X509_TRUST_SSL_CLIENT 2
#define X509_TRUST_SSL_SERVER 3
#define X509_TRUST_EMAIL 4
#define X509_TRUST_OBJECT_SIGN 5
#define X509_TRUST_OCSP_SIGN 6
#define X509_TRUST_OCSP_REQUEST 7
#define X509_TRUST_TSA 8

// Keep these up to date!
#define X509_TRUST_MIN 1
#define X509_TRUST_MAX 8


// trust_flags values
#define X509_TRUST_DYNAMIC 1
#define X509_TRUST_DYNAMIC_NAME 2

// check_trust return codes

#define X509_TRUST_TRUSTED 1
#define X509_TRUST_REJECTED 2
#define X509_TRUST_UNTRUSTED 3

// Flags for X509_print_ex()

#define X509_FLAG_COMPAT 0
#define X509_FLAG_NO_HEADER 1L
#define X509_FLAG_NO_VERSION (1L << 1)
#define X509_FLAG_NO_SERIAL (1L << 2)
#define X509_FLAG_NO_SIGNAME (1L << 3)
#define X509_FLAG_NO_ISSUER (1L << 4)
#define X509_FLAG_NO_VALIDITY (1L << 5)
#define X509_FLAG_NO_SUBJECT (1L << 6)
#define X509_FLAG_NO_PUBKEY (1L << 7)
#define X509_FLAG_NO_EXTENSIONS (1L << 8)
#define X509_FLAG_NO_SIGDUMP (1L << 9)
#define X509_FLAG_NO_AUX (1L << 10)
#define X509_FLAG_NO_ATTRIBUTES (1L << 11)
#define X509_FLAG_NO_IDS (1L << 12)

// Flags specific to X509_NAME_print_ex(). These flags must not collide with
// |ASN1_STRFLGS_*|.

// The field separator information

#define XN_FLAG_SEP_MASK (0xf << 16)

#define XN_FLAG_COMPAT 0  // Traditional SSLeay: use old X509_NAME_print
#define XN_FLAG_SEP_COMMA_PLUS (1 << 16)  // RFC 2253 ,+
#define XN_FLAG_SEP_CPLUS_SPC (2 << 16)   // ,+ spaced: more readable
#define XN_FLAG_SEP_SPLUS_SPC (3 << 16)   // ;+ spaced
#define XN_FLAG_SEP_MULTILINE (4 << 16)   // One line per field

#define XN_FLAG_DN_REV (1 << 20)  // Reverse DN order

// How the field name is shown

#define XN_FLAG_FN_MASK (0x3 << 21)

#define XN_FLAG_FN_SN 0            // Object short name
#define XN_FLAG_FN_LN (1 << 21)    // Object long name
#define XN_FLAG_FN_OID (2 << 21)   // Always use OIDs
#define XN_FLAG_FN_NONE (3 << 21)  // No field names

#define XN_FLAG_SPC_EQ (1 << 23)  // Put spaces round '='

// This determines if we dump fields we don't recognise:
// RFC 2253 requires this.

#define XN_FLAG_DUMP_UNKNOWN_FIELDS (1 << 24)

#define XN_FLAG_FN_ALIGN (1 << 25)  // Align field names to 20 characters

// Complete set of RFC 2253 flags

#define XN_FLAG_RFC2253                                             \
  (ASN1_STRFLGS_RFC2253 | XN_FLAG_SEP_COMMA_PLUS | XN_FLAG_DN_REV | \
   XN_FLAG_FN_SN | XN_FLAG_DUMP_UNKNOWN_FIELDS)

// readable oneline form

#define XN_FLAG_ONELINE                                                    \
  (ASN1_STRFLGS_RFC2253 | ASN1_STRFLGS_ESC_QUOTE | XN_FLAG_SEP_CPLUS_SPC | \
   XN_FLAG_SPC_EQ | XN_FLAG_FN_SN)

// readable multiline form

#define XN_FLAG_MULTILINE                                                 \
  (ASN1_STRFLGS_ESC_CTRL | ASN1_STRFLGS_ESC_MSB | XN_FLAG_SEP_MULTILINE | \
   XN_FLAG_SPC_EQ | XN_FLAG_FN_LN | XN_FLAG_FN_ALIGN)

DEFINE_STACK_OF(X509_REVOKED)

DECLARE_STACK_OF(GENERAL_NAMES)

struct private_key_st {
  int version;
  // The PKCS#8 data types
  X509_ALGOR *enc_algor;
  ASN1_OCTET_STRING *enc_pkey;  // encrypted pub key

  // When decrypted, the following will not be NULL
  EVP_PKEY *dec_pkey;

  // used to encrypt and decrypt
  int key_length;
  char *key_data;
  int key_free;  // true if we should auto free key_data

  // expanded version of 'enc_algor'
  EVP_CIPHER_INFO cipher;
} /* X509_PKEY */;

struct X509_info_st {
  X509 *x509;
  X509_CRL *crl;
  X509_PKEY *x_pkey;

  EVP_CIPHER_INFO enc_cipher;
  int enc_len;
  char *enc_data;

} /* X509_INFO */;

DEFINE_STACK_OF(X509_INFO)

// The next 2 structures and their 8 routines were sent to me by
// Pat Richard <patr@x509.com> and are used to manipulate
// Netscapes spki structures - useful if you are writing a CA web page
struct Netscape_spkac_st {
  X509_PUBKEY *pubkey;
  ASN1_IA5STRING *challenge;  // challenge sent in atlas >= PR2
} /* NETSCAPE_SPKAC */;

struct Netscape_spki_st {
  NETSCAPE_SPKAC *spkac;  // signed public key and challenge
  X509_ALGOR *sig_algor;
  ASN1_BIT_STRING *signature;
} /* NETSCAPE_SPKI */;

// X509_get_pathlen returns path length constraint from the basic constraints
// extension in |x509|. (See RFC 5280, section 4.2.1.9.) It returns -1 if the
// constraint is not present, or if some extension in |x509| was invalid.
//
// Note that decoding an |X509| object will not check for invalid extensions. To
// detect the error case, call |X509_get_extensions_flags| and check the
// |EXFLAG_INVALID| bit.
OPENSSL_EXPORT long X509_get_pathlen(X509 *x509);

// X509_SIG_get0 sets |*out_alg| and |*out_digest| to non-owning pointers to
// |sig|'s algorithm and digest fields, respectively. Either |out_alg| and
// |out_digest| may be NULL to skip those fields.
OPENSSL_EXPORT void X509_SIG_get0(const X509_SIG *sig,
                                  const X509_ALGOR **out_alg,
                                  const ASN1_OCTET_STRING **out_digest);

// X509_SIG_getm behaves like |X509_SIG_get0| but returns mutable pointers.
OPENSSL_EXPORT void X509_SIG_getm(X509_SIG *sig, X509_ALGOR **out_alg,
                                  ASN1_OCTET_STRING **out_digest);

// X509_verify_cert_error_string returns |err| as a human-readable string, where
// |err| should be one of the |X509_V_*| values. If |err| is unknown, it returns
// a default description.
OPENSSL_EXPORT const char *X509_verify_cert_error_string(long err);

// X509_verify checks that |x509| has a valid signature by |pkey|. It returns
// one if the signature is valid and zero otherwise. Note this function only
// checks the signature itself and does not perform a full certificate
// validation.
OPENSSL_EXPORT int X509_verify(X509 *x509, EVP_PKEY *pkey);

// X509_REQ_verify checks that |req| has a valid signature by |pkey|. It returns
// one if the signature is valid and zero otherwise.
OPENSSL_EXPORT int X509_REQ_verify(X509_REQ *req, EVP_PKEY *pkey);

// X509_CRL_verify checks that |crl| has a valid signature by |pkey|. It returns
// one if the signature is valid and zero otherwise.
OPENSSL_EXPORT int X509_CRL_verify(X509_CRL *crl, EVP_PKEY *pkey);

// NETSCAPE_SPKI_verify checks that |spki| has a valid signature by |pkey|. It
// returns one if the signature is valid and zero otherwise.
OPENSSL_EXPORT int NETSCAPE_SPKI_verify(NETSCAPE_SPKI *spki, EVP_PKEY *pkey);

// NETSCAPE_SPKI_b64_decode decodes |len| bytes from |str| as a base64-encoded
// Netscape signed public key and challenge (SPKAC) structure. It returns a
// newly-allocated |NETSCAPE_SPKI| structure with the result, or NULL on error.
// If |len| is 0 or negative, the length is calculated with |strlen| and |str|
// must be a NUL-terminated C string.
OPENSSL_EXPORT NETSCAPE_SPKI *NETSCAPE_SPKI_b64_decode(const char *str,
                                                       int len);

// NETSCAPE_SPKI_b64_encode encodes |spki| as a base64-encoded Netscape signed
// public key and challenge (SPKAC) structure. It returns a newly-allocated
// NUL-terminated C string with the result, or NULL on error. The caller must
// release the memory with |OPENSSL_free| when done.
OPENSSL_EXPORT char *NETSCAPE_SPKI_b64_encode(NETSCAPE_SPKI *spki);

// NETSCAPE_SPKI_get_pubkey decodes and returns the public key in |spki| as an
// |EVP_PKEY|, or NULL on error. The caller takes ownership of the resulting
// pointer and must call |EVP_PKEY_free| when done.
OPENSSL_EXPORT EVP_PKEY *NETSCAPE_SPKI_get_pubkey(NETSCAPE_SPKI *spki);

// NETSCAPE_SPKI_set_pubkey sets |spki|'s public key to |pkey|. It returns one
// on success or zero on error. This function does not take ownership of |pkey|,
// so the caller may continue to manage its lifetime independently of |spki|.
OPENSSL_EXPORT int NETSCAPE_SPKI_set_pubkey(NETSCAPE_SPKI *spki,
                                            EVP_PKEY *pkey);

// NETSCAPE_SPKI_sign signs |spki| with |pkey| and replaces the signature
// algorithm and signature fields. It returns one on success and zero on error.
// This function uses digest algorithm |md|, or |pkey|'s default if NULL. Other
// signing parameters use |pkey|'s defaults.
OPENSSL_EXPORT int NETSCAPE_SPKI_sign(NETSCAPE_SPKI *spki, EVP_PKEY *pkey,
                                      const EVP_MD *md);

// X509_ATTRIBUTE_dup returns a newly-allocated copy of |xa|, or NULL on error.
// This function works by serializing the structure, so if |xa| is incomplete,
// it may fail.
OPENSSL_EXPORT X509_ATTRIBUTE *X509_ATTRIBUTE_dup(const X509_ATTRIBUTE *xa);

// X509_REVOKED_dup returns a newly-allocated copy of |rev|, or NULL on error.
// This function works by serializing the structure, so if |rev| is incomplete,
// it may fail.
OPENSSL_EXPORT X509_REVOKED *X509_REVOKED_dup(const X509_REVOKED *rev);

// X509_cmp_time compares |s| against |*t|. On success, it returns a negative
// number if |s| <= |*t| and a positive number if |s| > |*t|. On error, it
// returns zero. If |t| is NULL, it uses the current time instead of |*t|.
//
// WARNING: Unlike most comparison functions, this function returns zero on
// error, not equality.
OPENSSL_EXPORT int X509_cmp_time(const ASN1_TIME *s, time_t *t);

// X509_cmp_time_posix compares |s| against |t|. On success, it returns a
// negative number if |s| <= |t| and a positive number if |s| > |t|. On error,
// it returns zero.
//
// WARNING: Unlike most comparison functions, this function returns zero on
// error, not equality.
OPENSSL_EXPORT int X509_cmp_time_posix(const ASN1_TIME *s, int64_t t);

// X509_cmp_current_time behaves like |X509_cmp_time| but compares |s| against
// the current time.
OPENSSL_EXPORT int X509_cmp_current_time(const ASN1_TIME *s);

// X509_time_adj calls |X509_time_adj_ex| with |offset_day| equal to zero.
OPENSSL_EXPORT ASN1_TIME *X509_time_adj(ASN1_TIME *s, long offset_sec,
                                        time_t *t);

// X509_time_adj_ex behaves like |ASN1_TIME_adj|, but adds an offset to |*t|. If
// |t| is NULL, it uses the current time instead of |*t|.
OPENSSL_EXPORT ASN1_TIME *X509_time_adj_ex(ASN1_TIME *s, int offset_day,
                                           long offset_sec, time_t *t);

// X509_gmtime_adj behaves like |X509_time_adj_ex| but adds |offset_sec| to the
// current time.
OPENSSL_EXPORT ASN1_TIME *X509_gmtime_adj(ASN1_TIME *s, long offset_sec);

OPENSSL_EXPORT const char *X509_get_default_cert_area(void);
OPENSSL_EXPORT const char *X509_get_default_cert_dir(void);
OPENSSL_EXPORT const char *X509_get_default_cert_file(void);
OPENSSL_EXPORT const char *X509_get_default_cert_dir_env(void);
OPENSSL_EXPORT const char *X509_get_default_cert_file_env(void);
OPENSSL_EXPORT const char *X509_get_default_private_dir(void);

DECLARE_ASN1_FUNCTIONS_const(X509_PUBKEY)

// X509_PUBKEY_set serializes |pkey| into a newly-allocated |X509_PUBKEY|
// structure. On success, it frees |*x|, sets |*x| to the new object, and
// returns one. Otherwise, it returns zero.
OPENSSL_EXPORT int X509_PUBKEY_set(X509_PUBKEY **x, EVP_PKEY *pkey);

// X509_PUBKEY_get decodes the public key in |key| and returns an |EVP_PKEY| on
// success, or NULL on error. The caller must release the result with
// |EVP_PKEY_free| when done. The |EVP_PKEY| is cached in |key|, so callers must
// not mutate the result.
OPENSSL_EXPORT EVP_PKEY *X509_PUBKEY_get(X509_PUBKEY *key);

DECLARE_ASN1_FUNCTIONS_const(X509_SIG)

DECLARE_ASN1_FUNCTIONS_const(X509_ATTRIBUTE)

// X509_ATTRIBUTE_create returns a newly-allocated |X509_ATTRIBUTE|, or NULL on
// error. The attribute has type |nid| and contains a single value determined by
// |attrtype| and |value|, which are interpreted as in |ASN1_TYPE_set|. Note
// this function takes ownership of |value|.
OPENSSL_EXPORT X509_ATTRIBUTE *X509_ATTRIBUTE_create(int nid, int attrtype,
                                                     void *value);

OPENSSL_EXPORT int X509_add1_trust_object(X509 *x, ASN1_OBJECT *obj);
OPENSSL_EXPORT int X509_add1_reject_object(X509 *x, ASN1_OBJECT *obj);
OPENSSL_EXPORT void X509_trust_clear(X509 *x);
OPENSSL_EXPORT void X509_reject_clear(X509 *x);


OPENSSL_EXPORT int X509_TRUST_set(int *t, int trust);

DECLARE_ASN1_FUNCTIONS_const(X509_REVOKED)

OPENSSL_EXPORT int X509_CRL_add0_revoked(X509_CRL *crl, X509_REVOKED *rev);
OPENSSL_EXPORT int X509_CRL_get0_by_serial(X509_CRL *crl, X509_REVOKED **ret,
                                           ASN1_INTEGER *serial);
OPENSSL_EXPORT int X509_CRL_get0_by_cert(X509_CRL *crl, X509_REVOKED **ret,
                                         X509 *x);

OPENSSL_EXPORT X509_PKEY *X509_PKEY_new(void);
OPENSSL_EXPORT void X509_PKEY_free(X509_PKEY *a);

DECLARE_ASN1_FUNCTIONS_const(NETSCAPE_SPKI)
DECLARE_ASN1_FUNCTIONS_const(NETSCAPE_SPKAC)

OPENSSL_EXPORT X509_INFO *X509_INFO_new(void);
OPENSSL_EXPORT void X509_INFO_free(X509_INFO *a);
OPENSSL_EXPORT char *X509_NAME_oneline(const X509_NAME *a, char *buf, int size);

OPENSSL_EXPORT int ASN1_digest(i2d_of_void *i2d, const EVP_MD *type, char *data,
                               unsigned char *md, unsigned int *len);

OPENSSL_EXPORT int ASN1_item_digest(const ASN1_ITEM *it, const EVP_MD *type,
                                    void *data, unsigned char *md,
                                    unsigned int *len);

OPENSSL_EXPORT int ASN1_item_verify(const ASN1_ITEM *it,
                                    const X509_ALGOR *algor1,
                                    const ASN1_BIT_STRING *signature,
                                    void *data, EVP_PKEY *pkey);

OPENSSL_EXPORT int ASN1_item_sign(const ASN1_ITEM *it, X509_ALGOR *algor1,
                                  X509_ALGOR *algor2,
                                  ASN1_BIT_STRING *signature, void *data,
                                  EVP_PKEY *pkey, const EVP_MD *type);
OPENSSL_EXPORT int ASN1_item_sign_ctx(const ASN1_ITEM *it, X509_ALGOR *algor1,
                                      X509_ALGOR *algor2,
                                      ASN1_BIT_STRING *signature, void *asn,
                                      EVP_MD_CTX *ctx);

// X509_REQ_extension_nid returns one if |nid| is a supported CSR attribute type
// for carrying extensions and zero otherwise. The supported types are
// |NID_ext_req| (pkcs-9-at-extensionRequest from RFC 2985) and |NID_ms_ext_req|
// (a Microsoft szOID_CERT_EXTENSIONS variant).
OPENSSL_EXPORT int X509_REQ_extension_nid(int nid);

// X509_REQ_get_extensions decodes the list of requested extensions in |req| and
// returns a newly-allocated |STACK_OF(X509_EXTENSION)| containing the result.
// It returns NULL on error, or if |req| did not request extensions.
//
// This function supports both pkcs-9-at-extensionRequest from RFC 2985 and the
// Microsoft szOID_CERT_EXTENSIONS variant.
OPENSSL_EXPORT STACK_OF(X509_EXTENSION) *X509_REQ_get_extensions(X509_REQ *req);

// X509_REQ_add_extensions_nid adds an attribute to |req| of type |nid|, to
// request the certificate extensions in |exts|. It returns one on success and
// zero on error. |nid| should be |NID_ext_req| or |NID_ms_ext_req|.
OPENSSL_EXPORT int X509_REQ_add_extensions_nid(
    X509_REQ *req, const STACK_OF(X509_EXTENSION) *exts, int nid);

// X509_REQ_add_extensions behaves like |X509_REQ_add_extensions_nid|, using the
// standard |NID_ext_req| for the attribute type.
OPENSSL_EXPORT int X509_REQ_add_extensions(
    X509_REQ *req, const STACK_OF(X509_EXTENSION) *exts);

// X509_REQ_get_attr_count returns the number of attributes in |req|.
OPENSSL_EXPORT int X509_REQ_get_attr_count(const X509_REQ *req);

// X509_REQ_get_attr_by_NID returns the index of the attribute in |req| of type
// |nid|, or a negative number if not found. If found, callers can use
// |X509_REQ_get_attr| to look up the attribute by index.
//
// If |lastpos| is non-negative, it begins searching at |lastpos| + 1. Callers
// can thus loop over all matching attributes by first passing -1 and then
// passing the previously-returned value until no match is returned.
OPENSSL_EXPORT int X509_REQ_get_attr_by_NID(const X509_REQ *req, int nid,
                                            int lastpos);

// X509_REQ_get_attr_by_OBJ behaves like |X509_REQ_get_attr_by_NID| but looks
// for attributes of type |obj|.
OPENSSL_EXPORT int X509_REQ_get_attr_by_OBJ(const X509_REQ *req,
                                            const ASN1_OBJECT *obj,
                                            int lastpos);

// X509_REQ_get_attr returns the attribute at index |loc| in |req|, or NULL if
// out of bounds.
OPENSSL_EXPORT X509_ATTRIBUTE *X509_REQ_get_attr(const X509_REQ *req, int loc);

// X509_REQ_delete_attr removes the attribute at index |loc| in |req|. It
// returns the removed attribute to the caller, or NULL if |loc| was out of
// bounds. If non-NULL, the caller must release the result with
// |X509_ATTRIBUTE_free| when done. It is also safe, but not necessary, to call
// |X509_ATTRIBUTE_free| if the result is NULL.
OPENSSL_EXPORT X509_ATTRIBUTE *X509_REQ_delete_attr(X509_REQ *req, int loc);

// X509_REQ_add1_attr appends a copy of |attr| to |req|'s list of attributes. It
// returns one on success and zero on error.
//
// TODO(https://crbug.com/boringssl/407): |attr| should be const.
OPENSSL_EXPORT int X509_REQ_add1_attr(X509_REQ *req, X509_ATTRIBUTE *attr);

// X509_REQ_add1_attr_by_OBJ appends a new attribute to |req| with type |obj|.
// It returns one on success and zero on error. The value is determined by
// |X509_ATTRIBUTE_set1_data|.
//
// WARNING: The interpretation of |attrtype|, |data|, and |len| is complex and
// error-prone. See |X509_ATTRIBUTE_set1_data| for details.
OPENSSL_EXPORT int X509_REQ_add1_attr_by_OBJ(X509_REQ *req,
                                             const ASN1_OBJECT *obj,
                                             int attrtype,
                                             const unsigned char *data,
                                             int len);

// X509_REQ_add1_attr_by_NID behaves like |X509_REQ_add1_attr_by_OBJ| except the
// attribute type is determined by |nid|.
OPENSSL_EXPORT int X509_REQ_add1_attr_by_NID(X509_REQ *req, int nid,
                                             int attrtype,
                                             const unsigned char *data,
                                             int len);

// X509_REQ_add1_attr_by_txt behaves like |X509_REQ_add1_attr_by_OBJ| except the
// attribute type is determined by calling |OBJ_txt2obj| with |attrname|.
OPENSSL_EXPORT int X509_REQ_add1_attr_by_txt(X509_REQ *req,
                                             const char *attrname, int attrtype,
                                             const unsigned char *data,
                                             int len);

OPENSSL_EXPORT int X509_CRL_sort(X509_CRL *crl);

// X509_REVOKED_get0_serialNumber returns the serial number of the certificate
// revoked by |revoked|.
OPENSSL_EXPORT const ASN1_INTEGER *X509_REVOKED_get0_serialNumber(
    const X509_REVOKED *revoked);

// X509_REVOKED_set_serialNumber sets |revoked|'s serial number to |serial|. It
// returns one on success or zero on error.
OPENSSL_EXPORT int X509_REVOKED_set_serialNumber(X509_REVOKED *revoked,
                                                 const ASN1_INTEGER *serial);

// X509_REVOKED_get0_revocationDate returns the revocation time of the
// certificate revoked by |revoked|.
OPENSSL_EXPORT const ASN1_TIME *X509_REVOKED_get0_revocationDate(
    const X509_REVOKED *revoked);

// X509_REVOKED_set_revocationDate sets |revoked|'s revocation time to |tm|. It
// returns one on success or zero on error.
OPENSSL_EXPORT int X509_REVOKED_set_revocationDate(X509_REVOKED *revoked,
                                                   const ASN1_TIME *tm);

// X509_REVOKED_get0_extensions returns |r|'s extensions list, or NULL if |r|
// omits it.
OPENSSL_EXPORT const STACK_OF(X509_EXTENSION) *X509_REVOKED_get0_extensions(
    const X509_REVOKED *r);

OPENSSL_EXPORT X509_CRL *X509_CRL_diff(X509_CRL *base, X509_CRL *newer,
                                       EVP_PKEY *skey, const EVP_MD *md,
                                       unsigned int flags);

OPENSSL_EXPORT int X509_REQ_check_private_key(X509_REQ *x509, EVP_PKEY *pkey);

OPENSSL_EXPORT int X509_check_private_key(X509 *x509, const EVP_PKEY *pkey);

OPENSSL_EXPORT int X509_issuer_name_cmp(const X509 *a, const X509 *b);
OPENSSL_EXPORT unsigned long X509_issuer_name_hash(X509 *a);

OPENSSL_EXPORT int X509_subject_name_cmp(const X509 *a, const X509 *b);
OPENSSL_EXPORT unsigned long X509_subject_name_hash(X509 *x);

OPENSSL_EXPORT unsigned long X509_issuer_name_hash_old(X509 *a);
OPENSSL_EXPORT unsigned long X509_subject_name_hash_old(X509 *x);

OPENSSL_EXPORT int X509_cmp(const X509 *a, const X509 *b);
OPENSSL_EXPORT int X509_NAME_cmp(const X509_NAME *a, const X509_NAME *b);
OPENSSL_EXPORT unsigned long X509_NAME_hash(X509_NAME *x);
OPENSSL_EXPORT unsigned long X509_NAME_hash_old(X509_NAME *x);

OPENSSL_EXPORT int X509_CRL_cmp(const X509_CRL *a, const X509_CRL *b);
OPENSSL_EXPORT int X509_CRL_match(const X509_CRL *a, const X509_CRL *b);
OPENSSL_EXPORT int X509_print_ex_fp(FILE *bp, X509 *x, unsigned long nmflag,
                                    unsigned long cflag);
OPENSSL_EXPORT int X509_print_fp(FILE *bp, X509 *x);
OPENSSL_EXPORT int X509_CRL_print_fp(FILE *bp, X509_CRL *x);
OPENSSL_EXPORT int X509_REQ_print_fp(FILE *bp, X509_REQ *req);
OPENSSL_EXPORT int X509_NAME_print_ex_fp(FILE *fp, const X509_NAME *nm,
                                         int indent, unsigned long flags);

OPENSSL_EXPORT int X509_NAME_print(BIO *bp, const X509_NAME *name, int obase);
OPENSSL_EXPORT int X509_NAME_print_ex(BIO *out, const X509_NAME *nm, int indent,
                                      unsigned long flags);
OPENSSL_EXPORT int X509_print_ex(BIO *bp, X509 *x, unsigned long nmflag,
                                 unsigned long cflag);
OPENSSL_EXPORT int X509_print(BIO *bp, X509 *x);
OPENSSL_EXPORT int X509_CRL_print(BIO *bp, X509_CRL *x);
OPENSSL_EXPORT int X509_REQ_print_ex(BIO *bp, X509_REQ *x, unsigned long nmflag,
                                     unsigned long cflag);
OPENSSL_EXPORT int X509_REQ_print(BIO *bp, X509_REQ *req);

// X509_get_ext_d2i behaves like |X509V3_get_d2i| but looks for the extension in
// |x509|'s extension list.
//
// WARNING: This function is difficult to use correctly. See the documentation
// for |X509V3_get_d2i| for details.
OPENSSL_EXPORT void *X509_get_ext_d2i(const X509 *x509, int nid,
                                      int *out_critical, int *out_idx);

// X509_add1_ext_i2d behaves like |X509V3_add1_i2d| but adds the extension to
// |x|'s extension list.
//
// WARNING: This function may return zero or -1 on error. The caller must also
// ensure |value|'s type matches |nid|. See the documentation for
// |X509V3_add1_i2d| for details.
OPENSSL_EXPORT int X509_add1_ext_i2d(X509 *x, int nid, void *value, int crit,
                                     unsigned long flags);

// X509_CRL_get_ext_d2i behaves like |X509V3_get_d2i| but looks for the
// extension in |crl|'s extension list.
//
// WARNING: This function is difficult to use correctly. See the documentation
// for |X509V3_get_d2i| for details.
OPENSSL_EXPORT void *X509_CRL_get_ext_d2i(const X509_CRL *crl, int nid,
                                          int *out_critical, int *out_idx);

// X509_CRL_add1_ext_i2d behaves like |X509V3_add1_i2d| but adds the extension
// to |x|'s extension list.
//
// WARNING: This function may return zero or -1 on error. The caller must also
// ensure |value|'s type matches |nid|. See the documentation for
// |X509V3_add1_i2d| for details.
OPENSSL_EXPORT int X509_CRL_add1_ext_i2d(X509_CRL *x, int nid, void *value,
                                         int crit, unsigned long flags);

// X509_REVOKED_get_ext_count returns the number of extensions in |x|.
OPENSSL_EXPORT int X509_REVOKED_get_ext_count(const X509_REVOKED *x);

// X509_REVOKED_get_ext_by_NID behaves like |X509v3_get_ext_by_NID| but searches
// for extensions in |x|.
OPENSSL_EXPORT int X509_REVOKED_get_ext_by_NID(const X509_REVOKED *x, int nid,
                                               int lastpos);

// X509_REVOKED_get_ext_by_OBJ behaves like |X509v3_get_ext_by_OBJ| but searches
// for extensions in |x|.
OPENSSL_EXPORT int X509_REVOKED_get_ext_by_OBJ(const X509_REVOKED *x,
                                               const ASN1_OBJECT *obj,
                                               int lastpos);

// X509_REVOKED_get_ext_by_critical behaves like |X509v3_get_ext_by_critical|
// but searches for extensions in |x|.
OPENSSL_EXPORT int X509_REVOKED_get_ext_by_critical(const X509_REVOKED *x,
                                                    int crit, int lastpos);

// X509_REVOKED_get_ext returns the extension in |x| at index |loc|, or NULL if
// |loc| is out of bounds. This function returns a non-const pointer for OpenSSL
// compatibility, but callers should not mutate the result.
OPENSSL_EXPORT X509_EXTENSION *X509_REVOKED_get_ext(const X509_REVOKED *x,
                                                    int loc);

// X509_REVOKED_delete_ext removes the extension in |x| at index |loc| and
// returns the removed extension, or NULL if |loc| was out of bounds. If
// non-NULL, the caller must release the result with |X509_EXTENSION_free|.
OPENSSL_EXPORT X509_EXTENSION *X509_REVOKED_delete_ext(X509_REVOKED *x,
                                                       int loc);

// X509_REVOKED_add_ext adds a copy of |ex| to |x|. It returns one on success
// and zero on failure. The caller retains ownership of |ex| and can release it
// independently of |x|.
//
// The new extension is inserted at index |loc|, shifting extensions to the
// right. If |loc| is -1 or out of bounds, the new extension is appended to the
// list.
OPENSSL_EXPORT int X509_REVOKED_add_ext(X509_REVOKED *x,
                                        const X509_EXTENSION *ex, int loc);

// X509_REVOKED_get_ext_d2i behaves like |X509V3_get_d2i| but looks for the
// extension in |revoked|'s extension list.
//
// WARNING: This function is difficult to use correctly. See the documentation
// for |X509V3_get_d2i| for details.
OPENSSL_EXPORT void *X509_REVOKED_get_ext_d2i(const X509_REVOKED *revoked,
                                              int nid, int *out_critical,
                                              int *out_idx);

// X509_REVOKED_add1_ext_i2d behaves like |X509V3_add1_i2d| but adds the
// extension to |x|'s extension list.
//
// WARNING: This function may return zero or -1 on error. The caller must also
// ensure |value|'s type matches |nid|. See the documentation for
// |X509V3_add1_i2d| for details.
OPENSSL_EXPORT int X509_REVOKED_add1_ext_i2d(X509_REVOKED *x, int nid,
                                             void *value, int crit,
                                             unsigned long flags);

// X509at_get_attr_count returns the number of attributes in |x|.
OPENSSL_EXPORT int X509at_get_attr_count(const STACK_OF(X509_ATTRIBUTE) *x);

// X509at_get_attr_by_NID returns the index of the attribute in |x| of type
// |nid|, or a negative number if not found. If found, callers can use
// |X509at_get_attr| to look up the attribute by index.
//
// If |lastpos| is non-negative, it begins searching at |lastpos| + 1. Callers
// can thus loop over all matching attributes by first passing -1 and then
// passing the previously-returned value until no match is returned.
OPENSSL_EXPORT int X509at_get_attr_by_NID(const STACK_OF(X509_ATTRIBUTE) *x,
                                          int nid, int lastpos);

// X509at_get_attr_by_OBJ behaves like |X509at_get_attr_by_NID| but looks for
// attributes of type |obj|.
OPENSSL_EXPORT int X509at_get_attr_by_OBJ(const STACK_OF(X509_ATTRIBUTE) *sk,
                                          const ASN1_OBJECT *obj, int lastpos);

// X509at_get_attr returns the attribute at index |loc| in |x|, or NULL if
// out of bounds.
OPENSSL_EXPORT X509_ATTRIBUTE *X509at_get_attr(
    const STACK_OF(X509_ATTRIBUTE) *x, int loc);

// X509at_delete_attr removes the attribute at index |loc| in |x|. It returns
// the removed attribute to the caller, or NULL if |loc| was out of bounds. If
// non-NULL, the caller must release the result with |X509_ATTRIBUTE_free| when
// done. It is also safe, but not necessary, to call |X509_ATTRIBUTE_free| if
// the result is NULL.
OPENSSL_EXPORT X509_ATTRIBUTE *X509at_delete_attr(STACK_OF(X509_ATTRIBUTE) *x,
                                                  int loc);

// X509at_add1_attr appends a copy of |attr| to the attribute list in |*x|. If
// |*x| is NULL, it allocates a new |STACK_OF(X509_ATTRIBUTE)| to hold the copy
// and sets |*x| to the new list. It returns |*x| on success and NULL on error.
// The caller retains ownership of |attr| and can release it independently of
// |*x|.
OPENSSL_EXPORT STACK_OF(X509_ATTRIBUTE) *X509at_add1_attr(
    STACK_OF(X509_ATTRIBUTE) **x, X509_ATTRIBUTE *attr);

// X509at_add1_attr_by_OBJ behaves like |X509at_add1_attr|, but adds an
// attribute created by |X509_ATTRIBUTE_create_by_OBJ|.
OPENSSL_EXPORT STACK_OF(X509_ATTRIBUTE) *X509at_add1_attr_by_OBJ(
    STACK_OF(X509_ATTRIBUTE) **x, const ASN1_OBJECT *obj, int type,
    const unsigned char *bytes, int len);

// X509at_add1_attr_by_NID behaves like |X509at_add1_attr|, but adds an
// attribute created by |X509_ATTRIBUTE_create_by_NID|.
OPENSSL_EXPORT STACK_OF(X509_ATTRIBUTE) *X509at_add1_attr_by_NID(
    STACK_OF(X509_ATTRIBUTE) **x, int nid, int type, const unsigned char *bytes,
    int len);

// X509at_add1_attr_by_txt behaves like |X509at_add1_attr|, but adds an
// attribute created by |X509_ATTRIBUTE_create_by_txt|.
OPENSSL_EXPORT STACK_OF(X509_ATTRIBUTE) *X509at_add1_attr_by_txt(
    STACK_OF(X509_ATTRIBUTE) **x, const char *attrname, int type,
    const unsigned char *bytes, int len);

// X509_ATTRIBUTE_create_by_NID returns a newly-allocated |X509_ATTRIBUTE| of
// type |nid|, or NULL on error. The value is determined as in
// |X509_ATTRIBUTE_set1_data|.
//
// If |attr| is non-NULL, the resulting |X509_ATTRIBUTE| is also written to
// |*attr|. If |*attr| was non-NULL when the function was called, |*attr| is
// reused instead of creating a new object.
//
// WARNING: The interpretation of |attrtype|, |data|, and |len| is complex and
// error-prone. See |X509_ATTRIBUTE_set1_data| for details.
//
// WARNING: The object reuse form is deprecated and may be removed in the
// future. It also currently incorrectly appends to the reused object's value
// set rather than overwriting it.
OPENSSL_EXPORT X509_ATTRIBUTE *X509_ATTRIBUTE_create_by_NID(
    X509_ATTRIBUTE **attr, int nid, int attrtype, const void *data, int len);

// X509_ATTRIBUTE_create_by_OBJ behaves like |X509_ATTRIBUTE_create_by_NID|
// except the attribute's type is determined by |obj|.
OPENSSL_EXPORT X509_ATTRIBUTE *X509_ATTRIBUTE_create_by_OBJ(
    X509_ATTRIBUTE **attr, const ASN1_OBJECT *obj, int attrtype,
    const void *data, int len);

// X509_ATTRIBUTE_create_by_txt behaves like |X509_ATTRIBUTE_create_by_NID|
// except the attribute's type is determined by calling |OBJ_txt2obj| with
// |attrname|.
OPENSSL_EXPORT X509_ATTRIBUTE *X509_ATTRIBUTE_create_by_txt(
    X509_ATTRIBUTE **attr, const char *attrname, int type,
    const unsigned char *bytes, int len);

// X509_ATTRIBUTE_set1_object sets |attr|'s type to |obj|. It returns one on
// success and zero on error.
OPENSSL_EXPORT int X509_ATTRIBUTE_set1_object(X509_ATTRIBUTE *attr,
                                              const ASN1_OBJECT *obj);

// X509_ATTRIBUTE_set1_data appends a value to |attr|'s value set and returns
// one on success or zero on error. The value is determined as follows:
//
// If |attrtype| is a |MBSTRING_*| constant, the value is an ASN.1 string. The
// string is determined by decoding |len| bytes from |data| in the encoding
// specified by |attrtype|, and then re-encoding it in a form appropriate for
// |attr|'s type. If |len| is -1, |strlen(data)| is used instead. See
// |ASN1_STRING_set_by_NID| for details.
//
// Otherwise, if |len| is not -1, the value is an ASN.1 string. |attrtype| is an
// |ASN1_STRING| type value and the |len| bytes from |data| are copied as the
// type-specific representation of |ASN1_STRING|. See |ASN1_STRING| for details.
//
// WARNING: If this form is used to construct a negative INTEGER or ENUMERATED,
// |attrtype| includes the |V_ASN1_NEG| flag for |ASN1_STRING|, but the function
// forgets to clear the flag for |ASN1_TYPE|. This matches OpenSSL but is
// probably a bug. For now, do not use this form with negative values.
//
// Otherwise, if |len| is -1, the value is constructed by passing |attrtype| and
// |data| to |ASN1_TYPE_set1|. That is, |attrtype| is an |ASN1_TYPE| type value,
// and |data| is cast to the corresponding pointer type.
//
// WARNING: Despite the name, this function appends to |attr|'s value set,
// rather than overwriting it. To overwrite the value set, create a new
// |X509_ATTRIBUTE| with |X509_ATTRIBUTE_new|.
//
// WARNING: If using the |MBSTRING_*| form, pass a length rather than relying on
// |strlen|. In particular, |strlen| will not behave correctly if the input is
// |MBSTRING_BMP| or |MBSTRING_UNIV|.
//
// WARNING: This function currently misinterprets |V_ASN1_OTHER| as an
// |MBSTRING_*| constant. This matches OpenSSL but means it is impossible to
// construct a value with a non-universal tag.
OPENSSL_EXPORT int X509_ATTRIBUTE_set1_data(X509_ATTRIBUTE *attr, int attrtype,
                                            const void *data, int len);

// X509_ATTRIBUTE_get0_data returns the |idx|th value of |attr| in a
// type-specific representation to |attrtype|, or NULL if out of bounds or the
// type does not match. |attrtype| is one of the type values in |ASN1_TYPE|. On
// match, the return value uses the same representation as |ASN1_TYPE_set0|. See
// |ASN1_TYPE| for details.
OPENSSL_EXPORT void *X509_ATTRIBUTE_get0_data(X509_ATTRIBUTE *attr, int idx,
                                              int attrtype, void *unused);

// X509_ATTRIBUTE_count returns the number of values in |attr|.
OPENSSL_EXPORT int X509_ATTRIBUTE_count(const X509_ATTRIBUTE *attr);

// X509_ATTRIBUTE_get0_object returns the type of |attr|.
OPENSSL_EXPORT ASN1_OBJECT *X509_ATTRIBUTE_get0_object(X509_ATTRIBUTE *attr);

// X509_ATTRIBUTE_get0_type returns the |idx|th value in |attr|, or NULL if out
// of bounds. Note this function returns one of |attr|'s values, not the type.
OPENSSL_EXPORT ASN1_TYPE *X509_ATTRIBUTE_get0_type(X509_ATTRIBUTE *attr,
                                                   int idx);

OPENSSL_EXPORT int X509_verify_cert(X509_STORE_CTX *ctx);

// PKCS#8 utilities

DECLARE_ASN1_FUNCTIONS_const(PKCS8_PRIV_KEY_INFO)

// EVP_PKCS82PKEY returns |p8| as a newly-allocated |EVP_PKEY|, or NULL if the
// key was unsupported or could not be decoded. If non-NULL, the caller must
// release the result with |EVP_PKEY_free| when done.
//
// Use |EVP_parse_private_key| instead.
OPENSSL_EXPORT EVP_PKEY *EVP_PKCS82PKEY(const PKCS8_PRIV_KEY_INFO *p8);

// EVP_PKEY2PKCS8 encodes |pkey| as a PKCS#8 PrivateKeyInfo (RFC 5208),
// represented as a newly-allocated |PKCS8_PRIV_KEY_INFO|, or NULL on error. The
// caller must release the result with |PKCS8_PRIV_KEY_INFO_free| when done.
//
// Use |EVP_marshal_private_key| instead.
OPENSSL_EXPORT PKCS8_PRIV_KEY_INFO *EVP_PKEY2PKCS8(const EVP_PKEY *pkey);

// X509_PUBKEY_set0_param sets |pub| to a key with AlgorithmIdentifier
// determined by |obj|, |param_type|, and |param_value|, and an encoded
// public key of |key|. On success, it takes ownership of all its parameters and
// returns one. Otherwise, it returns zero. |key| must have been allocated by
// |OPENSSL_malloc|.
//
// |obj|, |param_type|, and |param_value| are interpreted as in
// |X509_ALGOR_set0|. See |X509_ALGOR_set0| for details.
OPENSSL_EXPORT int X509_PUBKEY_set0_param(X509_PUBKEY *pub, ASN1_OBJECT *obj,
                                          int param_type, void *param_value,
                                          uint8_t *key, int key_len);

// X509_PUBKEY_get0_param outputs fields of |pub| and returns one. If |out_obj|
// is not NULL, it sets |*out_obj| to AlgorithmIdentifier's OID. If |out_key|
// is not NULL, it sets |*out_key| and |*out_key_len| to the encoded public key.
// If |out_alg| is not NULL, it sets |*out_alg| to the AlgorithmIdentifier.
//
// Note: X.509 SubjectPublicKeyInfo structures store the encoded public key as a
// BIT STRING. |*out_key| and |*out_key_len| will silently pad the key with zero
// bits if |pub| did not contain a whole number of bytes. Use
// |X509_PUBKEY_get0_public_key| to preserve this information.
OPENSSL_EXPORT int X509_PUBKEY_get0_param(ASN1_OBJECT **out_obj,
                                          const uint8_t **out_key,
                                          int *out_key_len,
                                          X509_ALGOR **out_alg,
                                          X509_PUBKEY *pub);

// X509_PUBKEY_get0_public_key returns |pub|'s encoded public key.
OPENSSL_EXPORT const ASN1_BIT_STRING *X509_PUBKEY_get0_public_key(
    const X509_PUBKEY *pub);

OPENSSL_EXPORT int X509_check_trust(X509 *x, int id, int flags);
OPENSSL_EXPORT int X509_TRUST_get_count(void);
OPENSSL_EXPORT X509_TRUST *X509_TRUST_get0(int idx);
OPENSSL_EXPORT int X509_TRUST_get_by_id(int id);
OPENSSL_EXPORT int X509_TRUST_add(int id, int flags,
                                  int (*ck)(X509_TRUST *, X509 *, int),
                                  char *name, int arg1, void *arg2);
OPENSSL_EXPORT void X509_TRUST_cleanup(void);
OPENSSL_EXPORT int X509_TRUST_get_flags(const X509_TRUST *xp);
OPENSSL_EXPORT char *X509_TRUST_get0_name(const X509_TRUST *xp);
OPENSSL_EXPORT int X509_TRUST_get_trust(const X509_TRUST *xp);


struct rsa_pss_params_st {
  X509_ALGOR *hashAlgorithm;
  X509_ALGOR *maskGenAlgorithm;
  ASN1_INTEGER *saltLength;
  ASN1_INTEGER *trailerField;
  // OpenSSL caches the MGF hash on |RSA_PSS_PARAMS| in some cases. None of the
  // cases apply to BoringSSL, so this is always NULL, but Node expects the
  // field to be present.
  X509_ALGOR *maskHash;
} /* RSA_PSS_PARAMS */;

DECLARE_ASN1_FUNCTIONS_const(RSA_PSS_PARAMS)

/*
SSL_CTX -> X509_STORE
                -> X509_LOOKUP
                        ->X509_LOOKUP_METHOD
                -> X509_LOOKUP
                        ->X509_LOOKUP_METHOD

SSL	-> X509_STORE_CTX
                ->X509_STORE

The X509_STORE holds the tables etc for verification stuff.
A X509_STORE_CTX is used while validating a single certificate.
The X509_STORE has X509_LOOKUPs for looking up certs.
The X509_STORE then calls a function to actually verify the
certificate chain.
*/

#define X509_LU_X509 1
#define X509_LU_CRL 2
#define X509_LU_PKEY 3

DEFINE_STACK_OF(X509_LOOKUP)
DEFINE_STACK_OF(X509_OBJECT)
DEFINE_STACK_OF(X509_VERIFY_PARAM)

typedef int (*X509_STORE_CTX_verify_cb)(int, X509_STORE_CTX *);
typedef int (*X509_STORE_CTX_verify_fn)(X509_STORE_CTX *);
typedef int (*X509_STORE_CTX_get_issuer_fn)(X509 **issuer, X509_STORE_CTX *ctx,
                                            X509 *x);
typedef int (*X509_STORE_CTX_check_issued_fn)(X509_STORE_CTX *ctx, X509 *x,
                                              X509 *issuer);
typedef int (*X509_STORE_CTX_check_revocation_fn)(X509_STORE_CTX *ctx);
typedef int (*X509_STORE_CTX_get_crl_fn)(X509_STORE_CTX *ctx, X509_CRL **crl,
                                         X509 *x);
typedef int (*X509_STORE_CTX_check_crl_fn)(X509_STORE_CTX *ctx, X509_CRL *crl);
typedef int (*X509_STORE_CTX_cert_crl_fn)(X509_STORE_CTX *ctx, X509_CRL *crl,
                                          X509 *x);
typedef int (*X509_STORE_CTX_check_policy_fn)(X509_STORE_CTX *ctx);
typedef STACK_OF(X509) *(*X509_STORE_CTX_lookup_certs_fn)(X509_STORE_CTX *ctx,
                                                          X509_NAME *nm);
typedef STACK_OF(X509_CRL) *(*X509_STORE_CTX_lookup_crls_fn)(
    X509_STORE_CTX *ctx, X509_NAME *nm);
typedef int (*X509_STORE_CTX_cleanup_fn)(X509_STORE_CTX *ctx);

OPENSSL_EXPORT int X509_STORE_set_depth(X509_STORE *store, int depth);

OPENSSL_EXPORT void X509_STORE_CTX_set_depth(X509_STORE_CTX *ctx, int depth);

#define X509_STORE_CTX_set_app_data(ctx, data) \
  X509_STORE_CTX_set_ex_data(ctx, 0, data)
#define X509_STORE_CTX_get_app_data(ctx) X509_STORE_CTX_get_ex_data(ctx, 0)

#define X509_L_FILE_LOAD 1
#define X509_L_ADD_DIR 2

#define X509_LOOKUP_load_file(x, name, type) \
  X509_LOOKUP_ctrl((x), X509_L_FILE_LOAD, (name), (long)(type), NULL)

#define X509_LOOKUP_add_dir(x, name, type) \
  X509_LOOKUP_ctrl((x), X509_L_ADD_DIR, (name), (long)(type), NULL)

#define X509_V_OK 0
#define X509_V_ERR_UNSPECIFIED 1

#define X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT 2
#define X509_V_ERR_UNABLE_TO_GET_CRL 3
#define X509_V_ERR_UNABLE_TO_DECRYPT_CERT_SIGNATURE 4
#define X509_V_ERR_UNABLE_TO_DECRYPT_CRL_SIGNATURE 5
#define X509_V_ERR_UNABLE_TO_DECODE_ISSUER_PUBLIC_KEY 6
#define X509_V_ERR_CERT_SIGNATURE_FAILURE 7
#define X509_V_ERR_CRL_SIGNATURE_FAILURE 8
#define X509_V_ERR_CERT_NOT_YET_VALID 9
#define X509_V_ERR_CERT_HAS_EXPIRED 10
#define X509_V_ERR_CRL_NOT_YET_VALID 11
#define X509_V_ERR_CRL_HAS_EXPIRED 12
#define X509_V_ERR_ERROR_IN_CERT_NOT_BEFORE_FIELD 13
#define X509_V_ERR_ERROR_IN_CERT_NOT_AFTER_FIELD 14
#define X509_V_ERR_ERROR_IN_CRL_LAST_UPDATE_FIELD 15
#define X509_V_ERR_ERROR_IN_CRL_NEXT_UPDATE_FIELD 16
#define X509_V_ERR_OUT_OF_MEM 17
#define X509_V_ERR_DEPTH_ZERO_SELF_SIGNED_CERT 18
#define X509_V_ERR_SELF_SIGNED_CERT_IN_CHAIN 19
#define X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY 20
#define X509_V_ERR_UNABLE_TO_VERIFY_LEAF_SIGNATURE 21
#define X509_V_ERR_CERT_CHAIN_TOO_LONG 22
#define X509_V_ERR_CERT_REVOKED 23
#define X509_V_ERR_INVALID_CA 24
#define X509_V_ERR_PATH_LENGTH_EXCEEDED 25
#define X509_V_ERR_INVALID_PURPOSE 26
#define X509_V_ERR_CERT_UNTRUSTED 27
#define X509_V_ERR_CERT_REJECTED 28
// These are 'informational' when looking for issuer cert
#define X509_V_ERR_SUBJECT_ISSUER_MISMATCH 29
#define X509_V_ERR_AKID_SKID_MISMATCH 30
#define X509_V_ERR_AKID_ISSUER_SERIAL_MISMATCH 31
#define X509_V_ERR_KEYUSAGE_NO_CERTSIGN 32

#define X509_V_ERR_UNABLE_TO_GET_CRL_ISSUER 33
#define X509_V_ERR_UNHANDLED_CRITICAL_EXTENSION 34
#define X509_V_ERR_KEYUSAGE_NO_CRL_SIGN 35
#define X509_V_ERR_UNHANDLED_CRITICAL_CRL_EXTENSION 36
#define X509_V_ERR_INVALID_NON_CA 37
#define X509_V_ERR_PROXY_PATH_LENGTH_EXCEEDED 38
#define X509_V_ERR_KEYUSAGE_NO_DIGITAL_SIGNATURE 39
#define X509_V_ERR_PROXY_CERTIFICATES_NOT_ALLOWED 40

#define X509_V_ERR_INVALID_EXTENSION 41
#define X509_V_ERR_INVALID_POLICY_EXTENSION 42
#define X509_V_ERR_NO_EXPLICIT_POLICY 43
#define X509_V_ERR_DIFFERENT_CRL_SCOPE 44
#define X509_V_ERR_UNSUPPORTED_EXTENSION_FEATURE 45

#define X509_V_ERR_UNNESTED_RESOURCE 46

#define X509_V_ERR_PERMITTED_VIOLATION 47
#define X509_V_ERR_EXCLUDED_VIOLATION 48
#define X509_V_ERR_SUBTREE_MINMAX 49
#define X509_V_ERR_APPLICATION_VERIFICATION 50
#define X509_V_ERR_UNSUPPORTED_CONSTRAINT_TYPE 51
#define X509_V_ERR_UNSUPPORTED_CONSTRAINT_SYNTAX 52
#define X509_V_ERR_UNSUPPORTED_NAME_SYNTAX 53
#define X509_V_ERR_CRL_PATH_VALIDATION_ERROR 54

// Host, email and IP check errors
#define X509_V_ERR_HOSTNAME_MISMATCH 62
#define X509_V_ERR_EMAIL_MISMATCH 63
#define X509_V_ERR_IP_ADDRESS_MISMATCH 64

// Caller error
#define X509_V_ERR_INVALID_CALL 65
// Issuer lookup error
#define X509_V_ERR_STORE_LOOKUP 66

#define X509_V_ERR_NAME_CONSTRAINTS_WITHOUT_SANS 67

// Certificate verify flags

// Send issuer+subject checks to verify_cb
#define X509_V_FLAG_CB_ISSUER_CHECK 0x1
// Use check time instead of current time
#define X509_V_FLAG_USE_CHECK_TIME 0x2
// Lookup CRLs
#define X509_V_FLAG_CRL_CHECK 0x4
// Lookup CRLs for whole chain
#define X509_V_FLAG_CRL_CHECK_ALL 0x8
// Ignore unhandled critical extensions
#define X509_V_FLAG_IGNORE_CRITICAL 0x10
// Does nothing as its functionality has been enabled by default.
#define X509_V_FLAG_X509_STRICT 0x00
// Enable proxy certificate validation
#define X509_V_FLAG_ALLOW_PROXY_CERTS 0x40
// Enable policy checking
#define X509_V_FLAG_POLICY_CHECK 0x80
// Policy variable require-explicit-policy
#define X509_V_FLAG_EXPLICIT_POLICY 0x100
// Policy variable inhibit-any-policy
#define X509_V_FLAG_INHIBIT_ANY 0x200
// Policy variable inhibit-policy-mapping
#define X509_V_FLAG_INHIBIT_MAP 0x400
// Notify callback that policy is OK
#define X509_V_FLAG_NOTIFY_POLICY 0x800
// Extended CRL features such as indirect CRLs, alternate CRL signing keys
#define X509_V_FLAG_EXTENDED_CRL_SUPPORT 0x1000
// Delta CRL support
#define X509_V_FLAG_USE_DELTAS 0x2000
// Check selfsigned CA signature
#define X509_V_FLAG_CHECK_SS_SIGNATURE 0x4000
// Use trusted store first
#define X509_V_FLAG_TRUSTED_FIRST 0x8000

// Allow partial chains if at least one certificate is in trusted store
#define X509_V_FLAG_PARTIAL_CHAIN 0x80000

// If the initial chain is not trusted, do not attempt to build an alternative
// chain. Alternate chain checking was introduced in 1.0.2b. Setting this flag
// will force the behaviour to match that of previous versions.
#define X509_V_FLAG_NO_ALT_CHAINS 0x100000

// X509_V_FLAG_NO_CHECK_TIME disables all time checks in certificate
// verification.
#define X509_V_FLAG_NO_CHECK_TIME 0x200000

#define X509_VP_FLAG_DEFAULT 0x1
#define X509_VP_FLAG_OVERWRITE 0x2
#define X509_VP_FLAG_RESET_FLAGS 0x4
#define X509_VP_FLAG_LOCKED 0x8
#define X509_VP_FLAG_ONCE 0x10

// Internal use: mask of policy related options
#define X509_V_FLAG_POLICY_MASK                             \
  (X509_V_FLAG_POLICY_CHECK | X509_V_FLAG_EXPLICIT_POLICY | \
   X509_V_FLAG_INHIBIT_ANY | X509_V_FLAG_INHIBIT_MAP)

OPENSSL_EXPORT int X509_OBJECT_idx_by_subject(STACK_OF(X509_OBJECT) *h,
                                              int type, X509_NAME *name);
OPENSSL_EXPORT X509_OBJECT *X509_OBJECT_retrieve_by_subject(
    STACK_OF(X509_OBJECT) *h, int type, X509_NAME *name);
OPENSSL_EXPORT X509_OBJECT *X509_OBJECT_retrieve_match(STACK_OF(X509_OBJECT) *h,
                                                       X509_OBJECT *x);
OPENSSL_EXPORT int X509_OBJECT_up_ref_count(X509_OBJECT *a);
OPENSSL_EXPORT void X509_OBJECT_free_contents(X509_OBJECT *a);
OPENSSL_EXPORT int X509_OBJECT_get_type(const X509_OBJECT *a);
OPENSSL_EXPORT X509 *X509_OBJECT_get0_X509(const X509_OBJECT *a);
OPENSSL_EXPORT X509_STORE *X509_STORE_new(void);
OPENSSL_EXPORT int X509_STORE_up_ref(X509_STORE *store);
OPENSSL_EXPORT void X509_STORE_free(X509_STORE *v);

OPENSSL_EXPORT STACK_OF(X509_OBJECT) *X509_STORE_get0_objects(X509_STORE *st);
OPENSSL_EXPORT STACK_OF(X509) *X509_STORE_get1_certs(X509_STORE_CTX *st,
                                                     X509_NAME *nm);
OPENSSL_EXPORT STACK_OF(X509_CRL) *X509_STORE_get1_crls(X509_STORE_CTX *st,
                                                        X509_NAME *nm);
OPENSSL_EXPORT int X509_STORE_set_flags(X509_STORE *ctx, unsigned long flags);
OPENSSL_EXPORT int X509_STORE_set_purpose(X509_STORE *ctx, int purpose);
OPENSSL_EXPORT int X509_STORE_set_trust(X509_STORE *ctx, int trust);
OPENSSL_EXPORT int X509_STORE_set1_param(X509_STORE *ctx,
                                         X509_VERIFY_PARAM *pm);
OPENSSL_EXPORT X509_VERIFY_PARAM *X509_STORE_get0_param(X509_STORE *ctx);

OPENSSL_EXPORT void X509_STORE_set_verify(X509_STORE *ctx,
                                          X509_STORE_CTX_verify_fn verify);
#define X509_STORE_set_verify_func(ctx, func) \
  X509_STORE_set_verify((ctx), (func))
OPENSSL_EXPORT void X509_STORE_CTX_set_verify(X509_STORE_CTX *ctx,
                                              X509_STORE_CTX_verify_fn verify);
OPENSSL_EXPORT X509_STORE_CTX_verify_fn X509_STORE_get_verify(X509_STORE *ctx);
OPENSSL_EXPORT void X509_STORE_set_verify_cb(
    X509_STORE *ctx, X509_STORE_CTX_verify_cb verify_cb);
#define X509_STORE_set_verify_cb_func(ctx, func) \
  X509_STORE_set_verify_cb((ctx), (func))
OPENSSL_EXPORT X509_STORE_CTX_verify_cb
X509_STORE_get_verify_cb(X509_STORE *ctx);
OPENSSL_EXPORT void X509_STORE_set_get_issuer(
    X509_STORE *ctx, X509_STORE_CTX_get_issuer_fn get_issuer);
OPENSSL_EXPORT X509_STORE_CTX_get_issuer_fn
X509_STORE_get_get_issuer(X509_STORE *ctx);
OPENSSL_EXPORT void X509_STORE_set_check_issued(
    X509_STORE *ctx, X509_STORE_CTX_check_issued_fn check_issued);
OPENSSL_EXPORT X509_STORE_CTX_check_issued_fn
X509_STORE_get_check_issued(X509_STORE *ctx);
OPENSSL_EXPORT void X509_STORE_set_check_revocation(
    X509_STORE *ctx, X509_STORE_CTX_check_revocation_fn check_revocation);
OPENSSL_EXPORT X509_STORE_CTX_check_revocation_fn
X509_STORE_get_check_revocation(X509_STORE *ctx);
OPENSSL_EXPORT void X509_STORE_set_get_crl(X509_STORE *ctx,
                                           X509_STORE_CTX_get_crl_fn get_crl);
OPENSSL_EXPORT X509_STORE_CTX_get_crl_fn
X509_STORE_get_get_crl(X509_STORE *ctx);
OPENSSL_EXPORT void X509_STORE_set_check_crl(
    X509_STORE *ctx, X509_STORE_CTX_check_crl_fn check_crl);
OPENSSL_EXPORT X509_STORE_CTX_check_crl_fn
X509_STORE_get_check_crl(X509_STORE *ctx);
OPENSSL_EXPORT void X509_STORE_set_cert_crl(
    X509_STORE *ctx, X509_STORE_CTX_cert_crl_fn cert_crl);
OPENSSL_EXPORT X509_STORE_CTX_cert_crl_fn
X509_STORE_get_cert_crl(X509_STORE *ctx);
OPENSSL_EXPORT void X509_STORE_set_lookup_certs(
    X509_STORE *ctx, X509_STORE_CTX_lookup_certs_fn lookup_certs);
OPENSSL_EXPORT X509_STORE_CTX_lookup_certs_fn
X509_STORE_get_lookup_certs(X509_STORE *ctx);
OPENSSL_EXPORT void X509_STORE_set_lookup_crls(
    X509_STORE *ctx, X509_STORE_CTX_lookup_crls_fn lookup_crls);
#define X509_STORE_set_lookup_crls_cb(ctx, func) \
  X509_STORE_set_lookup_crls((ctx), (func))
OPENSSL_EXPORT X509_STORE_CTX_lookup_crls_fn
X509_STORE_get_lookup_crls(X509_STORE *ctx);
OPENSSL_EXPORT void X509_STORE_set_cleanup(X509_STORE *ctx,
                                           X509_STORE_CTX_cleanup_fn cleanup);
OPENSSL_EXPORT X509_STORE_CTX_cleanup_fn
X509_STORE_get_cleanup(X509_STORE *ctx);

OPENSSL_EXPORT X509_STORE_CTX *X509_STORE_CTX_new(void);

OPENSSL_EXPORT int X509_STORE_CTX_get1_issuer(X509 **issuer,
                                              X509_STORE_CTX *ctx, X509 *x);

OPENSSL_EXPORT void X509_STORE_CTX_zero(X509_STORE_CTX *ctx);
OPENSSL_EXPORT void X509_STORE_CTX_free(X509_STORE_CTX *ctx);
OPENSSL_EXPORT int X509_STORE_CTX_init(X509_STORE_CTX *ctx, X509_STORE *store,
                                       X509 *x509, STACK_OF(X509) *chain);

// X509_STORE_CTX_set0_trusted_stack configures |ctx| to trust the certificates
// in |sk|. |sk| must remain valid for the duration of |ctx|.
//
// WARNING: This function differs from most |set0| functions in that it does not
// take ownership of its input. The caller is required to ensure the lifetimes
// are consistent.
OPENSSL_EXPORT void X509_STORE_CTX_set0_trusted_stack(X509_STORE_CTX *ctx,
                                                      STACK_OF(X509) *sk);

// X509_STORE_CTX_trusted_stack is a deprecated alias for
// |X509_STORE_CTX_set0_trusted_stack|.
OPENSSL_EXPORT void X509_STORE_CTX_trusted_stack(X509_STORE_CTX *ctx,
                                                 STACK_OF(X509) *sk);

OPENSSL_EXPORT void X509_STORE_CTX_cleanup(X509_STORE_CTX *ctx);

OPENSSL_EXPORT X509_STORE *X509_STORE_CTX_get0_store(X509_STORE_CTX *ctx);
OPENSSL_EXPORT X509 *X509_STORE_CTX_get0_cert(X509_STORE_CTX *ctx);

OPENSSL_EXPORT X509_LOOKUP *X509_STORE_add_lookup(X509_STORE *v,
                                                  X509_LOOKUP_METHOD *m);

OPENSSL_EXPORT X509_LOOKUP_METHOD *X509_LOOKUP_hash_dir(void);
OPENSSL_EXPORT X509_LOOKUP_METHOD *X509_LOOKUP_file(void);

OPENSSL_EXPORT int X509_STORE_add_cert(X509_STORE *ctx, X509 *x);
OPENSSL_EXPORT int X509_STORE_add_crl(X509_STORE *ctx, X509_CRL *x);

OPENSSL_EXPORT int X509_STORE_get_by_subject(X509_STORE_CTX *vs, int type,
                                             X509_NAME *name, X509_OBJECT *ret);

OPENSSL_EXPORT int X509_LOOKUP_ctrl(X509_LOOKUP *ctx, int cmd, const char *argc,
                                    long argl, char **ret);

#ifndef OPENSSL_NO_STDIO
OPENSSL_EXPORT int X509_load_cert_file(X509_LOOKUP *ctx, const char *file,
                                       int type);
OPENSSL_EXPORT int X509_load_crl_file(X509_LOOKUP *ctx, const char *file,
                                      int type);
OPENSSL_EXPORT int X509_load_cert_crl_file(X509_LOOKUP *ctx, const char *file,
                                           int type);
#endif

OPENSSL_EXPORT X509_LOOKUP *X509_LOOKUP_new(X509_LOOKUP_METHOD *method);
OPENSSL_EXPORT void X509_LOOKUP_free(X509_LOOKUP *ctx);
OPENSSL_EXPORT int X509_LOOKUP_init(X509_LOOKUP *ctx);
OPENSSL_EXPORT int X509_LOOKUP_by_subject(X509_LOOKUP *ctx, int type,
                                          X509_NAME *name, X509_OBJECT *ret);
OPENSSL_EXPORT int X509_LOOKUP_shutdown(X509_LOOKUP *ctx);

#ifndef OPENSSL_NO_STDIO
OPENSSL_EXPORT int X509_STORE_load_locations(X509_STORE *ctx, const char *file,
                                             const char *dir);
OPENSSL_EXPORT int X509_STORE_set_default_paths(X509_STORE *ctx);
#endif
OPENSSL_EXPORT int X509_STORE_CTX_get_error(X509_STORE_CTX *ctx);
OPENSSL_EXPORT void X509_STORE_CTX_set_error(X509_STORE_CTX *ctx, int s);
OPENSSL_EXPORT int X509_STORE_CTX_get_error_depth(X509_STORE_CTX *ctx);
OPENSSL_EXPORT X509 *X509_STORE_CTX_get_current_cert(X509_STORE_CTX *ctx);
OPENSSL_EXPORT X509 *X509_STORE_CTX_get0_current_issuer(X509_STORE_CTX *ctx);
OPENSSL_EXPORT X509_CRL *X509_STORE_CTX_get0_current_crl(X509_STORE_CTX *ctx);
OPENSSL_EXPORT X509_STORE_CTX *X509_STORE_CTX_get0_parent_ctx(
    X509_STORE_CTX *ctx);
OPENSSL_EXPORT STACK_OF(X509) *X509_STORE_CTX_get_chain(X509_STORE_CTX *ctx);
OPENSSL_EXPORT STACK_OF(X509) *X509_STORE_CTX_get0_chain(X509_STORE_CTX *ctx);
OPENSSL_EXPORT STACK_OF(X509) *X509_STORE_CTX_get1_chain(X509_STORE_CTX *ctx);
OPENSSL_EXPORT void X509_STORE_CTX_set_cert(X509_STORE_CTX *c, X509 *x);
OPENSSL_EXPORT void X509_STORE_CTX_set_chain(X509_STORE_CTX *c,
                                             STACK_OF(X509) *sk);
OPENSSL_EXPORT STACK_OF(X509) *X509_STORE_CTX_get0_untrusted(
    X509_STORE_CTX *ctx);
OPENSSL_EXPORT void X509_STORE_CTX_set0_crls(X509_STORE_CTX *c,
                                             STACK_OF(X509_CRL) *sk);
OPENSSL_EXPORT int X509_STORE_CTX_set_purpose(X509_STORE_CTX *ctx, int purpose);
OPENSSL_EXPORT int X509_STORE_CTX_set_trust(X509_STORE_CTX *ctx, int trust);
OPENSSL_EXPORT int X509_STORE_CTX_purpose_inherit(X509_STORE_CTX *ctx,
                                                  int def_purpose, int purpose,
                                                  int trust);
OPENSSL_EXPORT void X509_STORE_CTX_set_flags(X509_STORE_CTX *ctx,
                                             unsigned long flags);
OPENSSL_EXPORT void X509_STORE_CTX_set_time(X509_STORE_CTX *ctx,
                                            unsigned long flags, time_t t);
OPENSSL_EXPORT void X509_STORE_CTX_set_time_posix(X509_STORE_CTX *ctx,
                                                  unsigned long flags,
                                                  int64_t t);
OPENSSL_EXPORT void X509_STORE_CTX_set_verify_cb(
    X509_STORE_CTX *ctx, int (*verify_cb)(int, X509_STORE_CTX *));

OPENSSL_EXPORT X509_VERIFY_PARAM *X509_STORE_CTX_get0_param(
    X509_STORE_CTX *ctx);
OPENSSL_EXPORT void X509_STORE_CTX_set0_param(X509_STORE_CTX *ctx,
                                              X509_VERIFY_PARAM *param);
OPENSSL_EXPORT int X509_STORE_CTX_set_default(X509_STORE_CTX *ctx,
                                              const char *name);

// X509_VERIFY_PARAM functions

OPENSSL_EXPORT X509_VERIFY_PARAM *X509_VERIFY_PARAM_new(void);
OPENSSL_EXPORT void X509_VERIFY_PARAM_free(X509_VERIFY_PARAM *param);
OPENSSL_EXPORT int X509_VERIFY_PARAM_inherit(X509_VERIFY_PARAM *to,
                                             const X509_VERIFY_PARAM *from);
OPENSSL_EXPORT int X509_VERIFY_PARAM_set1(X509_VERIFY_PARAM *to,
                                          const X509_VERIFY_PARAM *from);
OPENSSL_EXPORT int X509_VERIFY_PARAM_set1_name(X509_VERIFY_PARAM *param,
                                               const char *name);
OPENSSL_EXPORT int X509_VERIFY_PARAM_set_flags(X509_VERIFY_PARAM *param,
                                               unsigned long flags);
OPENSSL_EXPORT int X509_VERIFY_PARAM_clear_flags(X509_VERIFY_PARAM *param,
                                                 unsigned long flags);
OPENSSL_EXPORT unsigned long X509_VERIFY_PARAM_get_flags(
    X509_VERIFY_PARAM *param);
OPENSSL_EXPORT int X509_VERIFY_PARAM_set_purpose(X509_VERIFY_PARAM *param,
                                                 int purpose);
OPENSSL_EXPORT int X509_VERIFY_PARAM_set_trust(X509_VERIFY_PARAM *param,
                                               int trust);
OPENSSL_EXPORT void X509_VERIFY_PARAM_set_depth(X509_VERIFY_PARAM *param,
                                                int depth);
OPENSSL_EXPORT void X509_VERIFY_PARAM_set_time(X509_VERIFY_PARAM *param,
                                               time_t t);
OPENSSL_EXPORT void X509_VERIFY_PARAM_set_time_posix(X509_VERIFY_PARAM *param,
                                                     int64_t t);
OPENSSL_EXPORT int X509_VERIFY_PARAM_add0_policy(X509_VERIFY_PARAM *param,
                                                 ASN1_OBJECT *policy);
OPENSSL_EXPORT int X509_VERIFY_PARAM_set1_policies(
    X509_VERIFY_PARAM *param, const STACK_OF(ASN1_OBJECT) *policies);

OPENSSL_EXPORT int X509_VERIFY_PARAM_set1_host(X509_VERIFY_PARAM *param,
                                               const char *name,
                                               size_t namelen);
OPENSSL_EXPORT int X509_VERIFY_PARAM_add1_host(X509_VERIFY_PARAM *param,
                                               const char *name,
                                               size_t namelen);
OPENSSL_EXPORT void X509_VERIFY_PARAM_set_hostflags(X509_VERIFY_PARAM *param,
                                                    unsigned int flags);
OPENSSL_EXPORT char *X509_VERIFY_PARAM_get0_peername(X509_VERIFY_PARAM *);
OPENSSL_EXPORT int X509_VERIFY_PARAM_set1_email(X509_VERIFY_PARAM *param,
                                                const char *email,
                                                size_t emaillen);
OPENSSL_EXPORT int X509_VERIFY_PARAM_set1_ip(X509_VERIFY_PARAM *param,
                                             const unsigned char *ip,
                                             size_t iplen);
OPENSSL_EXPORT int X509_VERIFY_PARAM_set1_ip_asc(X509_VERIFY_PARAM *param,
                                                 const char *ipasc);

OPENSSL_EXPORT int X509_VERIFY_PARAM_get_depth(const X509_VERIFY_PARAM *param);
OPENSSL_EXPORT const char *X509_VERIFY_PARAM_get0_name(
    const X509_VERIFY_PARAM *param);

OPENSSL_EXPORT int X509_VERIFY_PARAM_add0_table(X509_VERIFY_PARAM *param);
OPENSSL_EXPORT int X509_VERIFY_PARAM_get_count(void);
OPENSSL_EXPORT const X509_VERIFY_PARAM *X509_VERIFY_PARAM_get0(int id);
OPENSSL_EXPORT const X509_VERIFY_PARAM *X509_VERIFY_PARAM_lookup(
    const char *name);
OPENSSL_EXPORT void X509_VERIFY_PARAM_table_cleanup(void);


#if defined(__cplusplus)
}  // extern C
#endif

#if !defined(BORINGSSL_NO_CXX)
extern "C++" {

BSSL_NAMESPACE_BEGIN

BORINGSSL_MAKE_DELETER(NETSCAPE_SPKI, NETSCAPE_SPKI_free)
BORINGSSL_MAKE_DELETER(RSA_PSS_PARAMS, RSA_PSS_PARAMS_free)
BORINGSSL_MAKE_DELETER(X509, X509_free)
BORINGSSL_MAKE_UP_REF(X509, X509_up_ref)
BORINGSSL_MAKE_DELETER(X509_ALGOR, X509_ALGOR_free)
BORINGSSL_MAKE_DELETER(X509_ATTRIBUTE, X509_ATTRIBUTE_free)
BORINGSSL_MAKE_DELETER(X509_CRL, X509_CRL_free)
BORINGSSL_MAKE_UP_REF(X509_CRL, X509_CRL_up_ref)
BORINGSSL_MAKE_DELETER(X509_EXTENSION, X509_EXTENSION_free)
BORINGSSL_MAKE_DELETER(X509_INFO, X509_INFO_free)
BORINGSSL_MAKE_DELETER(X509_LOOKUP, X509_LOOKUP_free)
BORINGSSL_MAKE_DELETER(X509_NAME, X509_NAME_free)
BORINGSSL_MAKE_DELETER(X509_NAME_ENTRY, X509_NAME_ENTRY_free)
BORINGSSL_MAKE_DELETER(X509_PKEY, X509_PKEY_free)
BORINGSSL_MAKE_DELETER(X509_PUBKEY, X509_PUBKEY_free)
BORINGSSL_MAKE_DELETER(X509_REQ, X509_REQ_free)
BORINGSSL_MAKE_DELETER(X509_REVOKED, X509_REVOKED_free)
BORINGSSL_MAKE_DELETER(X509_SIG, X509_SIG_free)
BORINGSSL_MAKE_DELETER(X509_STORE, X509_STORE_free)
BORINGSSL_MAKE_UP_REF(X509_STORE, X509_STORE_up_ref)
BORINGSSL_MAKE_DELETER(X509_STORE_CTX, X509_STORE_CTX_free)
BORINGSSL_MAKE_DELETER(X509_VERIFY_PARAM, X509_VERIFY_PARAM_free)

BSSL_NAMESPACE_END

}  // extern C++
#endif  // !BORINGSSL_NO_CXX

#define X509_R_AKID_MISMATCH 100
#define X509_R_BAD_PKCS7_VERSION 101
#define X509_R_BAD_X509_FILETYPE 102
#define X509_R_BASE64_DECODE_ERROR 103
#define X509_R_CANT_CHECK_DH_KEY 104
#define X509_R_CERT_ALREADY_IN_HASH_TABLE 105
#define X509_R_CRL_ALREADY_DELTA 106
#define X509_R_CRL_VERIFY_FAILURE 107
#define X509_R_IDP_MISMATCH 108
#define X509_R_INVALID_BIT_STRING_BITS_LEFT 109
#define X509_R_INVALID_DIRECTORY 110
#define X509_R_INVALID_FIELD_NAME 111
#define X509_R_INVALID_PSS_PARAMETERS 112
#define X509_R_INVALID_TRUST 113
#define X509_R_ISSUER_MISMATCH 114
#define X509_R_KEY_TYPE_MISMATCH 115
#define X509_R_KEY_VALUES_MISMATCH 116
#define X509_R_LOADING_CERT_DIR 117
#define X509_R_LOADING_DEFAULTS 118
#define X509_R_NEWER_CRL_NOT_NEWER 119
#define X509_R_NOT_PKCS7_SIGNED_DATA 120
#define X509_R_NO_CERTIFICATES_INCLUDED 121
#define X509_R_NO_CERT_SET_FOR_US_TO_VERIFY 122
#define X509_R_NO_CRLS_INCLUDED 123
#define X509_R_NO_CRL_NUMBER 124
#define X509_R_PUBLIC_KEY_DECODE_ERROR 125
#define X509_R_PUBLIC_KEY_ENCODE_ERROR 126
#define X509_R_SHOULD_RETRY 127
#define X509_R_UNKNOWN_KEY_TYPE 128
#define X509_R_UNKNOWN_NID 129
#define X509_R_UNKNOWN_PURPOSE_ID 130
#define X509_R_UNKNOWN_TRUST_ID 131
#define X509_R_UNSUPPORTED_ALGORITHM 132
#define X509_R_WRONG_LOOKUP_TYPE 133
#define X509_R_WRONG_TYPE 134
#define X509_R_NAME_TOO_LONG 135
#define X509_R_INVALID_PARAMETER 136
#define X509_R_SIGNATURE_ALGORITHM_MISMATCH 137
#define X509_R_DELTA_CRL_WITHOUT_CRL_NUMBER 138
#define X509_R_INVALID_FIELD_FOR_VERSION 139
#define X509_R_INVALID_VERSION 140
#define X509_R_NO_CERTIFICATE_FOUND 141
#define X509_R_NO_CERTIFICATE_OR_CRL_FOUND 142
#define X509_R_NO_CRL_FOUND 143
#define X509_R_INVALID_POLICY_EXTENSION 144

#endif  // OPENSSL_HEADER_X509_H
