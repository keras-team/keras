/* Written by Dr Stephen N Henson (steve@openssl.org) for the OpenSSL
 * project 1999. */
/* ====================================================================
 * Copyright (c) 1999-2004 The OpenSSL Project.  All rights reserved.
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

#ifndef OPENSSL_HEADER_X509V3_H
#define OPENSSL_HEADER_X509V3_H

#include <openssl/bio.h>
#include <openssl/conf.h>
#include <openssl/lhash.h>
#include <openssl/x509.h>

#if defined(__cplusplus)
extern "C" {
#endif


// Legacy X.509 library.
//
// This header is part of OpenSSL's X.509 implementation. It is retained for
// compatibility but otherwise underdocumented and not actively maintained. In
// the future, a replacement library will be available. Meanwhile, minimize
// dependencies on this header where possible.


// Forward reference
struct v3_ext_method;
struct v3_ext_ctx;

// Useful typedefs

typedef struct v3_ext_method X509V3_EXT_METHOD;

typedef void *(*X509V3_EXT_NEW)(void);
typedef void (*X509V3_EXT_FREE)(void *);
typedef void *(*X509V3_EXT_D2I)(void *, const unsigned char **, long);
typedef int (*X509V3_EXT_I2D)(void *, unsigned char **);
typedef STACK_OF(CONF_VALUE) *(*X509V3_EXT_I2V)(const X509V3_EXT_METHOD *method,
                                                void *ext,
                                                STACK_OF(CONF_VALUE) *extlist);
typedef void *(*X509V3_EXT_V2I)(const X509V3_EXT_METHOD *method,
                                const X509V3_CTX *ctx,
                                const STACK_OF(CONF_VALUE) *values);
typedef char *(*X509V3_EXT_I2S)(const X509V3_EXT_METHOD *method, void *ext);
typedef void *(*X509V3_EXT_S2I)(const X509V3_EXT_METHOD *method,
                                const X509V3_CTX *ctx, const char *str);
typedef int (*X509V3_EXT_I2R)(const X509V3_EXT_METHOD *method, void *ext,
                              BIO *out, int indent);
typedef void *(*X509V3_EXT_R2I)(const X509V3_EXT_METHOD *method,
                                const X509V3_CTX *ctx, const char *str);

// V3 extension structure

struct v3_ext_method {
  int ext_nid;
  int ext_flags;
  // If this is set the following four fields are ignored
  ASN1_ITEM_EXP *it;
  // Old style ASN1 calls
  X509V3_EXT_NEW ext_new;
  X509V3_EXT_FREE ext_free;
  X509V3_EXT_D2I d2i;
  X509V3_EXT_I2D i2d;

  // The following pair is used for string extensions
  X509V3_EXT_I2S i2s;
  X509V3_EXT_S2I s2i;

  // The following pair is used for multi-valued extensions
  X509V3_EXT_I2V i2v;
  X509V3_EXT_V2I v2i;

  // The following are used for raw extensions
  X509V3_EXT_I2R i2r;
  X509V3_EXT_R2I r2i;

  void *usr_data;  // Any extension specific data
};

DEFINE_STACK_OF(X509V3_EXT_METHOD)

// ext_flags values
#define X509V3_EXT_DYNAMIC 0x1
#define X509V3_EXT_CTX_DEP 0x2
#define X509V3_EXT_MULTILINE 0x4

struct BASIC_CONSTRAINTS_st {
  int ca;
  ASN1_INTEGER *pathlen;
};


typedef struct otherName_st {
  ASN1_OBJECT *type_id;
  ASN1_TYPE *value;
} OTHERNAME;

typedef struct EDIPartyName_st {
  ASN1_STRING *nameAssigner;
  ASN1_STRING *partyName;
} EDIPARTYNAME;

typedef struct GENERAL_NAME_st {
#define GEN_OTHERNAME 0
#define GEN_EMAIL 1
#define GEN_DNS 2
#define GEN_X400 3
#define GEN_DIRNAME 4
#define GEN_EDIPARTY 5
#define GEN_URI 6
#define GEN_IPADD 7
#define GEN_RID 8

  int type;
  union {
    char *ptr;
    OTHERNAME *otherName;  // otherName
    ASN1_IA5STRING *rfc822Name;
    ASN1_IA5STRING *dNSName;
    ASN1_STRING *x400Address;
    X509_NAME *directoryName;
    EDIPARTYNAME *ediPartyName;
    ASN1_IA5STRING *uniformResourceIdentifier;
    ASN1_OCTET_STRING *iPAddress;
    ASN1_OBJECT *registeredID;

    // Old names
    ASN1_OCTET_STRING *ip;  // iPAddress
    X509_NAME *dirn;        // dirn
    ASN1_IA5STRING *ia5;    // rfc822Name, dNSName, uniformResourceIdentifier
    ASN1_OBJECT *rid;       // registeredID
  } d;
} GENERAL_NAME;

DEFINE_STACK_OF(GENERAL_NAME)

typedef STACK_OF(GENERAL_NAME) GENERAL_NAMES;

DEFINE_STACK_OF(GENERAL_NAMES)

typedef struct ACCESS_DESCRIPTION_st {
  ASN1_OBJECT *method;
  GENERAL_NAME *location;
} ACCESS_DESCRIPTION;

DEFINE_STACK_OF(ACCESS_DESCRIPTION)

typedef STACK_OF(ACCESS_DESCRIPTION) AUTHORITY_INFO_ACCESS;

typedef STACK_OF(ASN1_OBJECT) EXTENDED_KEY_USAGE;

typedef struct DIST_POINT_NAME_st {
  int type;
  union {
    GENERAL_NAMES *fullname;
    STACK_OF(X509_NAME_ENTRY) *relativename;
  } name;
  // If relativename then this contains the full distribution point name
  X509_NAME *dpname;
} DIST_POINT_NAME;
// All existing reasons
#define CRLDP_ALL_REASONS 0x807f

#define CRL_REASON_NONE (-1)
#define CRL_REASON_UNSPECIFIED 0
#define CRL_REASON_KEY_COMPROMISE 1
#define CRL_REASON_CA_COMPROMISE 2
#define CRL_REASON_AFFILIATION_CHANGED 3
#define CRL_REASON_SUPERSEDED 4
#define CRL_REASON_CESSATION_OF_OPERATION 5
#define CRL_REASON_CERTIFICATE_HOLD 6
#define CRL_REASON_REMOVE_FROM_CRL 8
#define CRL_REASON_PRIVILEGE_WITHDRAWN 9
#define CRL_REASON_AA_COMPROMISE 10

struct DIST_POINT_st {
  DIST_POINT_NAME *distpoint;
  ASN1_BIT_STRING *reasons;
  GENERAL_NAMES *CRLissuer;
  int dp_reasons;
};

typedef STACK_OF(DIST_POINT) CRL_DIST_POINTS;

DEFINE_STACK_OF(DIST_POINT)

struct AUTHORITY_KEYID_st {
  ASN1_OCTET_STRING *keyid;
  GENERAL_NAMES *issuer;
  ASN1_INTEGER *serial;
};

typedef struct NOTICEREF_st {
  ASN1_STRING *organization;
  STACK_OF(ASN1_INTEGER) *noticenos;
} NOTICEREF;

typedef struct USERNOTICE_st {
  NOTICEREF *noticeref;
  ASN1_STRING *exptext;
} USERNOTICE;

typedef struct POLICYQUALINFO_st {
  ASN1_OBJECT *pqualid;
  union {
    ASN1_IA5STRING *cpsuri;
    USERNOTICE *usernotice;
    ASN1_TYPE *other;
  } d;
} POLICYQUALINFO;

DEFINE_STACK_OF(POLICYQUALINFO)

typedef struct POLICYINFO_st {
  ASN1_OBJECT *policyid;
  STACK_OF(POLICYQUALINFO) *qualifiers;
} POLICYINFO;

typedef STACK_OF(POLICYINFO) CERTIFICATEPOLICIES;

DEFINE_STACK_OF(POLICYINFO)

typedef struct POLICY_MAPPING_st {
  ASN1_OBJECT *issuerDomainPolicy;
  ASN1_OBJECT *subjectDomainPolicy;
} POLICY_MAPPING;

DEFINE_STACK_OF(POLICY_MAPPING)

typedef STACK_OF(POLICY_MAPPING) POLICY_MAPPINGS;

typedef struct GENERAL_SUBTREE_st {
  GENERAL_NAME *base;
  ASN1_INTEGER *minimum;
  ASN1_INTEGER *maximum;
} GENERAL_SUBTREE;

DEFINE_STACK_OF(GENERAL_SUBTREE)

struct NAME_CONSTRAINTS_st {
  STACK_OF(GENERAL_SUBTREE) *permittedSubtrees;
  STACK_OF(GENERAL_SUBTREE) *excludedSubtrees;
};

typedef struct POLICY_CONSTRAINTS_st {
  ASN1_INTEGER *requireExplicitPolicy;
  ASN1_INTEGER *inhibitPolicyMapping;
} POLICY_CONSTRAINTS;

// Proxy certificate structures, see RFC 3820
typedef struct PROXY_POLICY_st {
  ASN1_OBJECT *policyLanguage;
  ASN1_OCTET_STRING *policy;
} PROXY_POLICY;

typedef struct PROXY_CERT_INFO_EXTENSION_st {
  ASN1_INTEGER *pcPathLengthConstraint;
  PROXY_POLICY *proxyPolicy;
} PROXY_CERT_INFO_EXTENSION;

DECLARE_ASN1_FUNCTIONS_const(PROXY_POLICY)
DECLARE_ASN1_FUNCTIONS_const(PROXY_CERT_INFO_EXTENSION)

struct ISSUING_DIST_POINT_st {
  DIST_POINT_NAME *distpoint;
  int onlyuser;
  int onlyCA;
  ASN1_BIT_STRING *onlysomereasons;
  int indirectCRL;
  int onlyattr;
};

// Values in idp_flags field
// IDP present
#define IDP_PRESENT 0x1
// IDP values inconsistent
#define IDP_INVALID 0x2
// onlyuser true
#define IDP_ONLYUSER 0x4
// onlyCA true
#define IDP_ONLYCA 0x8
// onlyattr true
#define IDP_ONLYATTR 0x10
// indirectCRL true
#define IDP_INDIRECT 0x20
// onlysomereasons present
#define IDP_REASONS 0x40



// X509_PURPOSE stuff

#define EXFLAG_BCONS 0x1
#define EXFLAG_KUSAGE 0x2
#define EXFLAG_XKUSAGE 0x4
#define EXFLAG_NSCERT 0x8

#define EXFLAG_CA 0x10
// Really self issued not necessarily self signed
#define EXFLAG_SI 0x20
#define EXFLAG_V1 0x40
#define EXFLAG_INVALID 0x80
#define EXFLAG_SET 0x100
#define EXFLAG_CRITICAL 0x200
#define EXFLAG_PROXY 0x400

#define EXFLAG_FRESHEST 0x1000
// Self signed
#define EXFLAG_SS 0x2000

#define KU_DIGITAL_SIGNATURE 0x0080
#define KU_NON_REPUDIATION 0x0040
#define KU_KEY_ENCIPHERMENT 0x0020
#define KU_DATA_ENCIPHERMENT 0x0010
#define KU_KEY_AGREEMENT 0x0008
#define KU_KEY_CERT_SIGN 0x0004
#define KU_CRL_SIGN 0x0002
#define KU_ENCIPHER_ONLY 0x0001
#define KU_DECIPHER_ONLY 0x8000

#define NS_SSL_CLIENT 0x80
#define NS_SSL_SERVER 0x40
#define NS_SMIME 0x20
#define NS_OBJSIGN 0x10
#define NS_SSL_CA 0x04
#define NS_SMIME_CA 0x02
#define NS_OBJSIGN_CA 0x01
#define NS_ANY_CA (NS_SSL_CA | NS_SMIME_CA | NS_OBJSIGN_CA)

#define XKU_SSL_SERVER 0x1
#define XKU_SSL_CLIENT 0x2
#define XKU_SMIME 0x4
#define XKU_CODE_SIGN 0x8
#define XKU_SGC 0x10
#define XKU_OCSP_SIGN 0x20
#define XKU_TIMESTAMP 0x40
#define XKU_DVCS 0x80
#define XKU_ANYEKU 0x100

#define X509_PURPOSE_DYNAMIC 0x1
#define X509_PURPOSE_DYNAMIC_NAME 0x2

typedef struct x509_purpose_st {
  int purpose;
  int trust;  // Default trust ID
  int flags;
  int (*check_purpose)(const struct x509_purpose_st *, const X509 *, int);
  char *name;
  char *sname;
  void *usr_data;
} X509_PURPOSE;

#define X509_PURPOSE_SSL_CLIENT 1
#define X509_PURPOSE_SSL_SERVER 2
#define X509_PURPOSE_NS_SSL_SERVER 3
#define X509_PURPOSE_SMIME_SIGN 4
#define X509_PURPOSE_SMIME_ENCRYPT 5
#define X509_PURPOSE_CRL_SIGN 6
#define X509_PURPOSE_ANY 7
#define X509_PURPOSE_OCSP_HELPER 8
#define X509_PURPOSE_TIMESTAMP_SIGN 9

#define X509_PURPOSE_MIN 1
#define X509_PURPOSE_MAX 9

DEFINE_STACK_OF(X509_PURPOSE)

DECLARE_ASN1_FUNCTIONS_const(BASIC_CONSTRAINTS)

// TODO(https://crbug.com/boringssl/407): This is not const because it contains
// an |X509_NAME|.
DECLARE_ASN1_FUNCTIONS(AUTHORITY_KEYID)

// TODO(https://crbug.com/boringssl/407): This is not const because it contains
// an |X509_NAME|.
DECLARE_ASN1_FUNCTIONS(GENERAL_NAME)
OPENSSL_EXPORT GENERAL_NAME *GENERAL_NAME_dup(GENERAL_NAME *a);

// i2v_GENERAL_NAME serializes |gen| as a |CONF_VALUE|. If |ret| is non-NULL, it
// appends the value to |ret| and returns |ret| on success or NULL on error. If
// it returns NULL, the caller is still responsible for freeing |ret|. If |ret|
// is NULL, it returns a newly-allocated |STACK_OF(CONF_VALUE)| containing the
// result. |method| is ignored.
//
// Do not use this function. This is an internal implementation detail of the
// human-readable print functions. If extracting a SAN list from a certificate,
// look at |gen| directly.
OPENSSL_EXPORT STACK_OF(CONF_VALUE) *i2v_GENERAL_NAME(
    const X509V3_EXT_METHOD *method, GENERAL_NAME *gen,
    STACK_OF(CONF_VALUE) *ret);
OPENSSL_EXPORT int GENERAL_NAME_print(BIO *out, GENERAL_NAME *gen);

// TODO(https://crbug.com/boringssl/407): This is not const because it contains
// an |X509_NAME|.
DECLARE_ASN1_FUNCTIONS(GENERAL_NAMES)

// i2v_GENERAL_NAMES serializes |gen| as a list of |CONF_VALUE|s. If |ret| is
// non-NULL, it appends the values to |ret| and returns |ret| on success or NULL
// on error. If it returns NULL, the caller is still responsible for freeing
// |ret|. If |ret| is NULL, it returns a newly-allocated |STACK_OF(CONF_VALUE)|
// containing the results. |method| is ignored.
//
// Do not use this function. This is an internal implementation detail of the
// human-readable print functions. If extracting a SAN list from a certificate,
// look at |gen| directly.
OPENSSL_EXPORT STACK_OF(CONF_VALUE) *i2v_GENERAL_NAMES(
    const X509V3_EXT_METHOD *method, GENERAL_NAMES *gen,
    STACK_OF(CONF_VALUE) *extlist);
OPENSSL_EXPORT GENERAL_NAMES *v2i_GENERAL_NAMES(
    const X509V3_EXT_METHOD *method, const X509V3_CTX *ctx,
    const STACK_OF(CONF_VALUE) *nval);

DECLARE_ASN1_FUNCTIONS_const(OTHERNAME)
DECLARE_ASN1_FUNCTIONS_const(EDIPARTYNAME)
OPENSSL_EXPORT void GENERAL_NAME_set0_value(GENERAL_NAME *a, int type,
                                            void *value);
OPENSSL_EXPORT void *GENERAL_NAME_get0_value(const GENERAL_NAME *a, int *ptype);
OPENSSL_EXPORT int GENERAL_NAME_set0_othername(GENERAL_NAME *gen,
                                               ASN1_OBJECT *oid,
                                               ASN1_TYPE *value);
OPENSSL_EXPORT int GENERAL_NAME_get0_otherName(const GENERAL_NAME *gen,
                                               ASN1_OBJECT **poid,
                                               ASN1_TYPE **pvalue);

OPENSSL_EXPORT char *i2s_ASN1_OCTET_STRING(const X509V3_EXT_METHOD *method,
                                           const ASN1_OCTET_STRING *ia5);
OPENSSL_EXPORT ASN1_OCTET_STRING *s2i_ASN1_OCTET_STRING(
    const X509V3_EXT_METHOD *method, const X509V3_CTX *ctx, const char *str);

DECLARE_ASN1_FUNCTIONS_const(EXTENDED_KEY_USAGE)
OPENSSL_EXPORT int i2a_ACCESS_DESCRIPTION(BIO *bp, const ACCESS_DESCRIPTION *a);

DECLARE_ASN1_FUNCTIONS_const(CERTIFICATEPOLICIES)
DECLARE_ASN1_FUNCTIONS_const(POLICYINFO)
DECLARE_ASN1_FUNCTIONS_const(POLICYQUALINFO)
DECLARE_ASN1_FUNCTIONS_const(USERNOTICE)
DECLARE_ASN1_FUNCTIONS_const(NOTICEREF)

// TODO(https://crbug.com/boringssl/407): This is not const because it contains
// an |X509_NAME|.
DECLARE_ASN1_FUNCTIONS(CRL_DIST_POINTS)
// TODO(https://crbug.com/boringssl/407): This is not const because it contains
// an |X509_NAME|.
DECLARE_ASN1_FUNCTIONS(DIST_POINT)
// TODO(https://crbug.com/boringssl/407): This is not const because it contains
// an |X509_NAME|.
DECLARE_ASN1_FUNCTIONS(DIST_POINT_NAME)
// TODO(https://crbug.com/boringssl/407): This is not const because it contains
// an |X509_NAME|.
DECLARE_ASN1_FUNCTIONS(ISSUING_DIST_POINT)

OPENSSL_EXPORT int DIST_POINT_set_dpname(DIST_POINT_NAME *dpn,
                                         X509_NAME *iname);

OPENSSL_EXPORT int NAME_CONSTRAINTS_check(X509 *x, NAME_CONSTRAINTS *nc);

// TODO(https://crbug.com/boringssl/407): This is not const because it contains
// an |X509_NAME|.
DECLARE_ASN1_FUNCTIONS(ACCESS_DESCRIPTION)
// TODO(https://crbug.com/boringssl/407): This is not const because it contains
// an |X509_NAME|.
DECLARE_ASN1_FUNCTIONS(AUTHORITY_INFO_ACCESS)

DECLARE_ASN1_ITEM(POLICY_MAPPING)
DECLARE_ASN1_ALLOC_FUNCTIONS(POLICY_MAPPING)
DECLARE_ASN1_ITEM(POLICY_MAPPINGS)

DECLARE_ASN1_ITEM(GENERAL_SUBTREE)
DECLARE_ASN1_ALLOC_FUNCTIONS(GENERAL_SUBTREE)

DECLARE_ASN1_ITEM(NAME_CONSTRAINTS)
DECLARE_ASN1_ALLOC_FUNCTIONS(NAME_CONSTRAINTS)

DECLARE_ASN1_ALLOC_FUNCTIONS(POLICY_CONSTRAINTS)
DECLARE_ASN1_ITEM(POLICY_CONSTRAINTS)

OPENSSL_EXPORT GENERAL_NAME *a2i_GENERAL_NAME(GENERAL_NAME *out,
                                              const X509V3_EXT_METHOD *method,
                                              const X509V3_CTX *ctx, int gen_type,
                                              const char *value, int is_nc);

OPENSSL_EXPORT GENERAL_NAME *v2i_GENERAL_NAME(const X509V3_EXT_METHOD *method,
                                              const X509V3_CTX *ctx,
                                              const CONF_VALUE *cnf);
OPENSSL_EXPORT GENERAL_NAME *v2i_GENERAL_NAME_ex(
    GENERAL_NAME *out, const X509V3_EXT_METHOD *method, const X509V3_CTX *ctx,
    const CONF_VALUE *cnf, int is_nc);
OPENSSL_EXPORT void X509V3_conf_free(CONF_VALUE *val);


// Deprecated config-based extension creation.
//
// The following functions allow specifying X.509 extensions using OpenSSL's
// config file syntax, from the OpenSSL command-line tool. They are retained,
// for now, for compatibility with legacy software but may be removed in the
// future. Construct the extensions using the typed C APIs instead.
//
// Callers should especially avoid these functions if passing in non-constant
// values. They use ad-hoc, string-based formats which are prone to injection
// vulnerabilities. For a CA, this means using them risks misissuance.
//
// These functions are not safe to use with untrusted inputs. The string formats
// may implicitly reference context information and, in OpenSSL (though not
// BoringSSL), one even allows reading arbitrary files. They additionally see
// much less testing and review than most of the library and may have bugs
// including memory leaks or crashes.

// v3_ext_ctx, aka |X509V3_CTX|, contains additional context information for
// constructing extensions. Some string formats reference additional values in
// these objects. It must be initialized with |X509V3_set_ctx| or
// |X509V3_set_ctx_test| before use.
struct v3_ext_ctx {
  int flags;
  const X509 *issuer_cert;
  const X509 *subject_cert;
  const X509_REQ *subject_req;
  const X509_CRL *crl;
  const CONF *db;
};

#define X509V3_CTX_TEST 0x1

// X509V3_set_ctx initializes |ctx| with the specified objects. Some string
// formats will reference fields in these objects. Each object may be NULL to
// omit it, in which case those formats cannot be used. |flags| should be zero,
// unless called via |X509V3_set_ctx_test|.
//
// |issuer|, |subject|, |req|, and |crl|, if non-NULL, must outlive |ctx|.
OPENSSL_EXPORT void X509V3_set_ctx(X509V3_CTX *ctx, const X509 *issuer,
                                   const X509 *subject, const X509_REQ *req,
                                   const X509_CRL *crl, int flags);

// X509V3_set_ctx_test calls |X509V3_set_ctx| without any reference objects and
// mocks out some features that use them. The resulting extensions may be
// incomplete and should be discarded. This can be used to partially validate
// syntax.
//
// TODO(davidben): Can we remove this?
#define X509V3_set_ctx_test(ctx) \
  X509V3_set_ctx(ctx, NULL, NULL, NULL, NULL, X509V3_CTX_TEST)

// X509V3_set_nconf sets |ctx| to use |conf| as the config database. |ctx| must
// have previously been initialized by |X509V3_set_ctx| or
// |X509V3_set_ctx_test|. Some string formats will reference sections in |conf|.
// |conf| may be NULL, in which case these formats cannot be used. If non-NULL,
// |conf| must outlive |ctx|.
OPENSSL_EXPORT void X509V3_set_nconf(X509V3_CTX *ctx, const CONF *conf);

// X509V3_set_ctx_nodb calls |X509V3_set_nconf| with no config database.
#define X509V3_set_ctx_nodb(ctx) X509V3_set_nconf(ctx, NULL)

// X509V3_EXT_nconf constructs an extension of type specified by |name|, and
// value specified by |value|. It returns a newly-allocated |X509_EXTENSION|
// object on success, or NULL on error. |conf| and |ctx| specify additional
// information referenced by some formats. Either |conf| or |ctx| may be NULL,
// in which case features which use it will be disabled.
//
// If non-NULL, |ctx| must be initialized with |X509V3_set_ctx| or
// |X509V3_set_ctx_test|.
//
// Both |conf| and |ctx| provide a |CONF| object. When |ctx| is non-NULL, most
// features use the |ctx| copy, configured with |X509V3_set_ctx|, but some use
// |conf|. Callers should ensure the two match to avoid surprisingly behavior.
OPENSSL_EXPORT X509_EXTENSION *X509V3_EXT_nconf(const CONF *conf,
                                                const X509V3_CTX *ctx,
                                                const char *name,
                                                const char *value);

// X509V3_EXT_nconf_nid behaves like |X509V3_EXT_nconf|, except the extension
// type is specified as a NID.
OPENSSL_EXPORT X509_EXTENSION *X509V3_EXT_nconf_nid(const CONF *conf,
                                                    const X509V3_CTX *ctx,
                                                    int ext_nid,
                                                    const char *value);

// X509V3_EXT_conf_nid calls |X509V3_EXT_nconf_nid|. |conf| must be NULL.
//
// TODO(davidben): This is the only exposed instance of an LHASH in our public
// headers. cryptography.io wraps this function so we cannot, yet, replace the
// type with a dummy struct.
OPENSSL_EXPORT X509_EXTENSION *X509V3_EXT_conf_nid(LHASH_OF(CONF_VALUE) *conf,
                                                   const X509V3_CTX *ctx,
                                                   int ext_nid,
                                                   const char *value);

// X509V3_EXT_add_nconf_sk looks up the section named |section| in |conf|. For
// each |CONF_VALUE| in the section, it constructs an extension as in
// |X509V3_EXT_nconf|, taking |name| and |value| from the |CONF_VALUE|. Each new
// extension is appended to |*sk|. If |*sk| is non-NULL, and at least one
// extension is added, it sets |*sk| to a newly-allocated
// |STACK_OF(X509_EXTENSION)|. It returns one on success and zero on error.
OPENSSL_EXPORT int X509V3_EXT_add_nconf_sk(const CONF *conf,
                                           const X509V3_CTX *ctx,
                                           const char *section,
                                           STACK_OF(X509_EXTENSION) **sk);

// X509V3_EXT_add_nconf adds extensions to |cert| as in
// |X509V3_EXT_add_nconf_sk|. It returns one on success and zero on error.
OPENSSL_EXPORT int X509V3_EXT_add_nconf(const CONF *conf, const X509V3_CTX *ctx,
                                        const char *section, X509 *cert);

// X509V3_EXT_REQ_add_nconf adds extensions to |req| as in
// |X509V3_EXT_add_nconf_sk|. It returns one on success and zero on error.
OPENSSL_EXPORT int X509V3_EXT_REQ_add_nconf(const CONF *conf,
                                            const X509V3_CTX *ctx,
                                            const char *section, X509_REQ *req);

// X509V3_EXT_CRL_add_nconf adds extensions to |crl| as in
// |X509V3_EXT_add_nconf_sk|. It returns one on success and zero on error.
OPENSSL_EXPORT int X509V3_EXT_CRL_add_nconf(const CONF *conf,
                                            const X509V3_CTX *ctx,
                                            const char *section, X509_CRL *crl);


OPENSSL_EXPORT char *i2s_ASN1_INTEGER(const X509V3_EXT_METHOD *meth,
                                      const ASN1_INTEGER *aint);
OPENSSL_EXPORT ASN1_INTEGER *s2i_ASN1_INTEGER(const X509V3_EXT_METHOD *meth,
                                              const char *value);
OPENSSL_EXPORT char *i2s_ASN1_ENUMERATED(const X509V3_EXT_METHOD *meth,
                                         const ASN1_ENUMERATED *aint);
OPENSSL_EXPORT int X509V3_EXT_add(X509V3_EXT_METHOD *ext);
OPENSSL_EXPORT int X509V3_EXT_add_list(X509V3_EXT_METHOD *extlist);
OPENSSL_EXPORT int X509V3_EXT_add_alias(int nid_to, int nid_from);
OPENSSL_EXPORT void X509V3_EXT_cleanup(void);

OPENSSL_EXPORT const X509V3_EXT_METHOD *X509V3_EXT_get(
    const X509_EXTENSION *ext);
OPENSSL_EXPORT const X509V3_EXT_METHOD *X509V3_EXT_get_nid(int nid);
OPENSSL_EXPORT int X509V3_add_standard_extensions(void);

// X509V3_EXT_d2i decodes |ext| and returns a pointer to a newly-allocated
// structure, with type dependent on the type of the extension. It returns NULL
// if |ext| is an unsupported extension or if there was a syntax error in the
// extension. The caller should cast the return value to the expected type and
// free the structure when done.
//
// WARNING: Casting the return value to the wrong type is a potentially
// exploitable memory error, so callers must not use this function before
// checking |ext| is of a known type.
OPENSSL_EXPORT void *X509V3_EXT_d2i(const X509_EXTENSION *ext);

// X509V3_get_d2i finds and decodes the extension in |extensions| of type |nid|.
// If found, it decodes it and returns a newly-allocated structure, with type
// dependent on |nid|. If the extension is not found or on error, it returns
// NULL. The caller may distinguish these cases using the |out_critical| value.
//
// If |out_critical| is not NULL, this function sets |*out_critical| to one if
// the extension is found and critical, zero if it is found and not critical, -1
// if it is not found, and -2 if there is an invalid duplicate extension. Note
// this function may set |*out_critical| to one or zero and still return NULL if
// the extension is found but has a syntax error.
//
// If |out_idx| is not NULL, this function looks for the first occurrence of the
// extension after |*out_idx|. It then sets |*out_idx| to the index of the
// extension, or -1 if not found. If |out_idx| is non-NULL, duplicate extensions
// are not treated as an error. Callers, however, should not rely on this
// behavior as it may be removed in the future. Duplicate extensions are
// forbidden in RFC 5280.
//
// WARNING: This function is difficult to use correctly. Callers should pass a
// non-NULL |out_critical| and check both the return value and |*out_critical|
// to handle errors. If the return value is NULL and |*out_critical| is not -1,
// there was an error. Otherwise, the function succeeded and but may return NULL
// for a missing extension. Callers should pass NULL to |out_idx| so that
// duplicate extensions are handled correctly.
//
// Additionally, casting the return value to the wrong type is a potentially
// exploitable memory error, so callers must ensure the cast and |nid| match.
OPENSSL_EXPORT void *X509V3_get_d2i(const STACK_OF(X509_EXTENSION) *extensions,
                                    int nid, int *out_critical, int *out_idx);

// X509V3_EXT_free casts |ext_data| into the type that corresponds to |nid| and
// releases memory associated with it. It returns one on success and zero if
// |nid| is not a known extension.
//
// WARNING: Casting |ext_data| to the wrong type is a potentially exploitable
// memory error, so callers must ensure |ext_data|'s type matches |nid|.
//
// TODO(davidben): OpenSSL upstream no longer exposes this function. Remove it?
OPENSSL_EXPORT int X509V3_EXT_free(int nid, void *ext_data);

// X509V3_EXT_i2d casts |ext_struc| into the type that corresponds to
// |ext_nid|, serializes it, and returns a newly-allocated |X509_EXTENSION|
// object containing the serialization, or NULL on error. The |X509_EXTENSION|
// has OID |ext_nid| and is critical if |crit| is one.
//
// WARNING: Casting |ext_struc| to the wrong type is a potentially exploitable
// memory error, so callers must ensure |ext_struct|'s type matches |ext_nid|.
OPENSSL_EXPORT X509_EXTENSION *X509V3_EXT_i2d(int ext_nid, int crit,
                                              void *ext_struc);

// The following constants control the behavior of |X509V3_add1_i2d| and related
// functions.

// X509V3_ADD_OP_MASK can be ANDed with the flags to determine how duplicate
// extensions are processed.
#define X509V3_ADD_OP_MASK 0xfL

// X509V3_ADD_DEFAULT causes the function to fail if the extension was already
// present.
#define X509V3_ADD_DEFAULT 0L

// X509V3_ADD_APPEND causes the function to unconditionally appended the new
// extension to to the extensions list, even if there is a duplicate.
#define X509V3_ADD_APPEND 1L

// X509V3_ADD_REPLACE causes the function to replace the existing extension, or
// append if it is not present.
#define X509V3_ADD_REPLACE 2L

// X509V3_ADD_REPLACE causes the function to replace the existing extension and
// fail if it is not present.
#define X509V3_ADD_REPLACE_EXISTING 3L

// X509V3_ADD_KEEP_EXISTING causes the function to succeed without replacing the
// extension if already present.
#define X509V3_ADD_KEEP_EXISTING 4L

// X509V3_ADD_DELETE causes the function to remove the matching extension. No
// new extension is added. If there is no matching extension, the function
// fails. The |value| parameter is ignored in this mode.
#define X509V3_ADD_DELETE 5L

// X509V3_ADD_SILENT may be ORed into one of the values above to indicate the
// function should not add to the error queue on duplicate or missing extension.
// The function will continue to return zero in those cases, and it will
// continue to return -1 and add to the error queue on other errors.
#define X509V3_ADD_SILENT 0x10

// X509V3_add1_i2d casts |value| to the type that corresponds to |nid|,
// serializes it, and appends it to the extension list in |*x|. If |*x| is NULL,
// it will set |*x| to a newly-allocated |STACK_OF(X509_EXTENSION)| as needed.
// The |crit| parameter determines whether the new extension is critical.
// |flags| may be some combination of the |X509V3_ADD_*| constants to control
// the function's behavior on duplicate extension.
//
// This function returns one on success, zero if the operation failed due to a
// missing or duplicate extension, and -1 on other errors.
//
// WARNING: Casting |value| to the wrong type is a potentially exploitable
// memory error, so callers must ensure |value|'s type matches |nid|.
OPENSSL_EXPORT int X509V3_add1_i2d(STACK_OF(X509_EXTENSION) **x, int nid,
                                   void *value, int crit, unsigned long flags);

#define X509V3_EXT_UNKNOWN_MASK (0xfL << 16)

// X509V3_EXT_DEFAULT causes unknown extensions or syntax errors to return
// failure.
#define X509V3_EXT_DEFAULT 0
// X509V3_EXT_ERROR_UNKNOWN causes unknown extensions or syntax errors to print
// as "<Not Supported>" or "<Parse Error>", respectively.
#define X509V3_EXT_ERROR_UNKNOWN (1L << 16)
// X509V3_EXT_PARSE_UNKNOWN is deprecated and behaves like
// |X509V3_EXT_DUMP_UNKNOWN|.
#define X509V3_EXT_PARSE_UNKNOWN (2L << 16)
// X509V3_EXT_DUMP_UNKNOWN causes unknown extensions to be displayed as a
// hexdump.
#define X509V3_EXT_DUMP_UNKNOWN (3L << 16)

OPENSSL_EXPORT void X509V3_EXT_val_prn(BIO *out,
                                       const STACK_OF(CONF_VALUE) *val,
                                       int indent, int ml);
OPENSSL_EXPORT int X509V3_EXT_print(BIO *out, const X509_EXTENSION *ext,
                                    unsigned long flag, int indent);
OPENSSL_EXPORT int X509V3_EXT_print_fp(FILE *out, const X509_EXTENSION *ext,
                                       int flag, int indent);

// X509V3_extensions_print prints |title|, followed by a human-readable
// representation of |exts| to |out|. It returns one on success and zero on
// error. The output is indented by |indent| spaces. |flag| is one of the
// |X509V3_EXT_*| constants and controls printing of unknown extensions and
// syntax errors.
OPENSSL_EXPORT int X509V3_extensions_print(BIO *out, const char *title,
                                           const STACK_OF(X509_EXTENSION) *exts,
                                           unsigned long flag, int indent);

OPENSSL_EXPORT int X509_check_ca(X509 *x);
OPENSSL_EXPORT int X509_check_purpose(X509 *x, int id, int ca);
OPENSSL_EXPORT int X509_supported_extension(const X509_EXTENSION *ex);
OPENSSL_EXPORT int X509_PURPOSE_set(int *p, int purpose);
OPENSSL_EXPORT int X509_check_issued(X509 *issuer, X509 *subject);
OPENSSL_EXPORT int X509_check_akid(X509 *issuer, AUTHORITY_KEYID *akid);

OPENSSL_EXPORT uint32_t X509_get_extension_flags(X509 *x);
OPENSSL_EXPORT uint32_t X509_get_key_usage(X509 *x);
OPENSSL_EXPORT uint32_t X509_get_extended_key_usage(X509 *x);

// X509_get0_subject_key_id returns |x509|'s subject key identifier, if present.
// (See RFC 5280, section 4.2.1.2.) It returns NULL if the extension is not
// present or if some extension in |x509| was invalid.
//
// Note that decoding an |X509| object will not check for invalid extensions. To
// detect the error case, call |X509_get_extensions_flags| and check the
// |EXFLAG_INVALID| bit.
OPENSSL_EXPORT const ASN1_OCTET_STRING *X509_get0_subject_key_id(X509 *x509);

// X509_get0_authority_key_id returns keyIdentifier of |x509|'s authority key
// identifier, if the extension and field are present. (See RFC 5280,
// section 4.2.1.1.) It returns NULL if the extension is not present, if it is
// present but lacks a keyIdentifier field, or if some extension in |x509| was
// invalid.
//
// Note that decoding an |X509| object will not check for invalid extensions. To
// detect the error case, call |X509_get_extensions_flags| and check the
// |EXFLAG_INVALID| bit.
OPENSSL_EXPORT const ASN1_OCTET_STRING *X509_get0_authority_key_id(X509 *x509);

// X509_get0_authority_issuer returns the authorityCertIssuer of |x509|'s
// authority key identifier, if the extension and field are present. (See
// RFC 5280, section 4.2.1.1.) It returns NULL if the extension is not present,
// if it is present but lacks a authorityCertIssuer field, or if some extension
// in |x509| was invalid.
//
// Note that decoding an |X509| object will not check for invalid extensions. To
// detect the error case, call |X509_get_extensions_flags| and check the
// |EXFLAG_INVALID| bit.
OPENSSL_EXPORT const GENERAL_NAMES *X509_get0_authority_issuer(X509 *x509);

// X509_get0_authority_serial returns the authorityCertSerialNumber of |x509|'s
// authority key identifier, if the extension and field are present. (See
// RFC 5280, section 4.2.1.1.) It returns NULL if the extension is not present,
// if it is present but lacks a authorityCertSerialNumber field, or if some
// extension in |x509| was invalid.
//
// Note that decoding an |X509| object will not check for invalid extensions. To
// detect the error case, call |X509_get_extensions_flags| and check the
// |EXFLAG_INVALID| bit.
OPENSSL_EXPORT const ASN1_INTEGER *X509_get0_authority_serial(X509 *x509);

OPENSSL_EXPORT int X509_PURPOSE_get_count(void);
OPENSSL_EXPORT X509_PURPOSE *X509_PURPOSE_get0(int idx);
OPENSSL_EXPORT int X509_PURPOSE_get_by_sname(char *sname);
OPENSSL_EXPORT int X509_PURPOSE_get_by_id(int id);
OPENSSL_EXPORT int X509_PURPOSE_add(int id, int trust, int flags,
                                    int (*ck)(const X509_PURPOSE *,
                                              const X509 *, int),
                                    char *name, char *sname, void *arg);
OPENSSL_EXPORT char *X509_PURPOSE_get0_name(const X509_PURPOSE *xp);
OPENSSL_EXPORT char *X509_PURPOSE_get0_sname(const X509_PURPOSE *xp);
OPENSSL_EXPORT int X509_PURPOSE_get_trust(const X509_PURPOSE *xp);
OPENSSL_EXPORT void X509_PURPOSE_cleanup(void);
OPENSSL_EXPORT int X509_PURPOSE_get_id(const X509_PURPOSE *);

OPENSSL_EXPORT STACK_OF(OPENSSL_STRING) *X509_get1_email(X509 *x);
OPENSSL_EXPORT STACK_OF(OPENSSL_STRING) *X509_REQ_get1_email(X509_REQ *x);
OPENSSL_EXPORT void X509_email_free(STACK_OF(OPENSSL_STRING) *sk);
OPENSSL_EXPORT STACK_OF(OPENSSL_STRING) *X509_get1_ocsp(X509 *x);
// Flags for X509_check_* functions

// Deprecated: this flag does nothing
#define X509_CHECK_FLAG_ALWAYS_CHECK_SUBJECT 0
// Disable wildcard matching for dnsName fields and common name.
#define X509_CHECK_FLAG_NO_WILDCARDS 0x2
// X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS does nothing, but is necessary in
// OpenSSL to enable standard wildcard matching. In BoringSSL, this behavior is
// always enabled.
#define X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS 0
// Deprecated: this flag does nothing
#define X509_CHECK_FLAG_MULTI_LABEL_WILDCARDS 0
// Deprecated: this flag does nothing
#define X509_CHECK_FLAG_SINGLE_LABEL_SUBDOMAINS 0
// Skip the subject common name fallback if subjectAltNames is missing.
#define X509_CHECK_FLAG_NEVER_CHECK_SUBJECT 0x20

OPENSSL_EXPORT int X509_check_host(X509 *x, const char *chk, size_t chklen,
                                   unsigned int flags, char **peername);
OPENSSL_EXPORT int X509_check_email(X509 *x, const char *chk, size_t chklen,
                                    unsigned int flags);
OPENSSL_EXPORT int X509_check_ip(X509 *x, const unsigned char *chk,
                                 size_t chklen, unsigned int flags);
OPENSSL_EXPORT int X509_check_ip_asc(X509 *x, const char *ipasc,
                                     unsigned int flags);

OPENSSL_EXPORT ASN1_OCTET_STRING *a2i_IPADDRESS(const char *ipasc);
OPENSSL_EXPORT ASN1_OCTET_STRING *a2i_IPADDRESS_NC(const char *ipasc);

// BEGIN ERROR CODES
// The following lines are auto generated by the script mkerr.pl. Any changes
// made after this point may be overwritten when the script is next run.


#if defined(__cplusplus)
}  // extern C

extern "C++" {

BSSL_NAMESPACE_BEGIN

BORINGSSL_MAKE_DELETER(ACCESS_DESCRIPTION, ACCESS_DESCRIPTION_free)
BORINGSSL_MAKE_DELETER(AUTHORITY_KEYID, AUTHORITY_KEYID_free)
BORINGSSL_MAKE_DELETER(BASIC_CONSTRAINTS, BASIC_CONSTRAINTS_free)
// TODO(davidben): Move this to conf.h and rename to CONF_VALUE_free.
BORINGSSL_MAKE_DELETER(CONF_VALUE, X509V3_conf_free)
BORINGSSL_MAKE_DELETER(DIST_POINT, DIST_POINT_free)
BORINGSSL_MAKE_DELETER(GENERAL_NAME, GENERAL_NAME_free)
BORINGSSL_MAKE_DELETER(GENERAL_SUBTREE, GENERAL_SUBTREE_free)
BORINGSSL_MAKE_DELETER(NAME_CONSTRAINTS, NAME_CONSTRAINTS_free)
BORINGSSL_MAKE_DELETER(POLICY_MAPPING, POLICY_MAPPING_free)
BORINGSSL_MAKE_DELETER(POLICYINFO, POLICYINFO_free)

BSSL_NAMESPACE_END

}  // extern C++
#endif

#define X509V3_R_BAD_IP_ADDRESS 100
#define X509V3_R_BAD_OBJECT 101
#define X509V3_R_BN_DEC2BN_ERROR 102
#define X509V3_R_BN_TO_ASN1_INTEGER_ERROR 103
#define X509V3_R_CANNOT_FIND_FREE_FUNCTION 104
#define X509V3_R_DIRNAME_ERROR 105
#define X509V3_R_DISTPOINT_ALREADY_SET 106
#define X509V3_R_DUPLICATE_ZONE_ID 107
#define X509V3_R_ERROR_CONVERTING_ZONE 108
#define X509V3_R_ERROR_CREATING_EXTENSION 109
#define X509V3_R_ERROR_IN_EXTENSION 110
#define X509V3_R_EXPECTED_A_SECTION_NAME 111
#define X509V3_R_EXTENSION_EXISTS 112
#define X509V3_R_EXTENSION_NAME_ERROR 113
#define X509V3_R_EXTENSION_NOT_FOUND 114
#define X509V3_R_EXTENSION_SETTING_NOT_SUPPORTED 115
#define X509V3_R_EXTENSION_VALUE_ERROR 116
#define X509V3_R_ILLEGAL_EMPTY_EXTENSION 117
#define X509V3_R_ILLEGAL_HEX_DIGIT 118
#define X509V3_R_INCORRECT_POLICY_SYNTAX_TAG 119
#define X509V3_R_INVALID_BOOLEAN_STRING 120
#define X509V3_R_INVALID_EXTENSION_STRING 121
#define X509V3_R_INVALID_MULTIPLE_RDNS 122
#define X509V3_R_INVALID_NAME 123
#define X509V3_R_INVALID_NULL_ARGUMENT 124
#define X509V3_R_INVALID_NULL_NAME 125
#define X509V3_R_INVALID_NULL_VALUE 126
#define X509V3_R_INVALID_NUMBER 127
#define X509V3_R_INVALID_NUMBERS 128
#define X509V3_R_INVALID_OBJECT_IDENTIFIER 129
#define X509V3_R_INVALID_OPTION 130
#define X509V3_R_INVALID_POLICY_IDENTIFIER 131
#define X509V3_R_INVALID_PROXY_POLICY_SETTING 132
#define X509V3_R_INVALID_PURPOSE 133
#define X509V3_R_INVALID_SECTION 134
#define X509V3_R_INVALID_SYNTAX 135
#define X509V3_R_ISSUER_DECODE_ERROR 136
#define X509V3_R_MISSING_VALUE 137
#define X509V3_R_NEED_ORGANIZATION_AND_NUMBERS 138
#define X509V3_R_NO_CONFIG_DATABASE 139
#define X509V3_R_NO_ISSUER_CERTIFICATE 140
#define X509V3_R_NO_ISSUER_DETAILS 141
#define X509V3_R_NO_POLICY_IDENTIFIER 142
#define X509V3_R_NO_PROXY_CERT_POLICY_LANGUAGE_DEFINED 143
#define X509V3_R_NO_PUBLIC_KEY 144
#define X509V3_R_NO_SUBJECT_DETAILS 145
#define X509V3_R_ODD_NUMBER_OF_DIGITS 146
#define X509V3_R_OPERATION_NOT_DEFINED 147
#define X509V3_R_OTHERNAME_ERROR 148
#define X509V3_R_POLICY_LANGUAGE_ALREADY_DEFINED 149
#define X509V3_R_POLICY_PATH_LENGTH 150
#define X509V3_R_POLICY_PATH_LENGTH_ALREADY_DEFINED 151
#define X509V3_R_POLICY_WHEN_PROXY_LANGUAGE_REQUIRES_NO_POLICY 152
#define X509V3_R_SECTION_NOT_FOUND 153
#define X509V3_R_UNABLE_TO_GET_ISSUER_DETAILS 154
#define X509V3_R_UNABLE_TO_GET_ISSUER_KEYID 155
#define X509V3_R_UNKNOWN_BIT_STRING_ARGUMENT 156
#define X509V3_R_UNKNOWN_EXTENSION 157
#define X509V3_R_UNKNOWN_EXTENSION_NAME 158
#define X509V3_R_UNKNOWN_OPTION 159
#define X509V3_R_UNSUPPORTED_OPTION 160
#define X509V3_R_UNSUPPORTED_TYPE 161
#define X509V3_R_USER_TOO_LONG 162
#define X509V3_R_INVALID_VALUE 163
#define X509V3_R_TRAILING_DATA_IN_EXTENSION 164

#endif  // OPENSSL_HEADER_X509V3_H
