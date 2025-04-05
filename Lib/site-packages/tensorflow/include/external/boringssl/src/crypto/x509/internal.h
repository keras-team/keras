/*
 * Written by Dr Stephen N Henson (steve@openssl.org) for the OpenSSL project
 * 2013.
 */
/* ====================================================================
 * Copyright (c) 2013 The OpenSSL Project.  All rights reserved.
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

#ifndef OPENSSL_HEADER_X509_INTERNAL_H
#define OPENSSL_HEADER_X509_INTERNAL_H

#include <openssl/base.h>
#include <openssl/evp.h>
#include <openssl/x509.h>

#include "../asn1/internal.h"

#if defined(__cplusplus)
extern "C" {
#endif


// Internal structures.

typedef struct X509_val_st {
  ASN1_TIME *notBefore;
  ASN1_TIME *notAfter;
} X509_VAL;

DECLARE_ASN1_FUNCTIONS_const(X509_VAL)

struct X509_pubkey_st {
  X509_ALGOR *algor;
  ASN1_BIT_STRING *public_key;
  EVP_PKEY *pkey;
} /* X509_PUBKEY */;

struct X509_name_entry_st {
  ASN1_OBJECT *object;
  ASN1_STRING *value;
  int set;
} /* X509_NAME_ENTRY */;

// we always keep X509_NAMEs in 2 forms.
struct X509_name_st {
  STACK_OF(X509_NAME_ENTRY) *entries;
  int modified;  // true if 'bytes' needs to be built
  BUF_MEM *bytes;
  // unsigned long hash; Keep the hash around for lookups
  unsigned char *canon_enc;
  int canon_enclen;
} /* X509_NAME */;

struct x509_attributes_st {
  ASN1_OBJECT *object;
  STACK_OF(ASN1_TYPE) *set;
} /* X509_ATTRIBUTE */;

typedef struct x509_cert_aux_st {
  STACK_OF(ASN1_OBJECT) *trust;   // trusted uses
  STACK_OF(ASN1_OBJECT) *reject;  // rejected uses
  ASN1_UTF8STRING *alias;         // "friendly name"
  ASN1_OCTET_STRING *keyid;       // key id of private key
} X509_CERT_AUX;

DECLARE_ASN1_FUNCTIONS_const(X509_CERT_AUX)

struct X509_extension_st {
  ASN1_OBJECT *object;
  ASN1_BOOLEAN critical;
  ASN1_OCTET_STRING *value;
} /* X509_EXTENSION */;

typedef struct {
  ASN1_INTEGER *version;  // [ 0 ] default of v1
  ASN1_INTEGER *serialNumber;
  X509_ALGOR *signature;
  X509_NAME *issuer;
  X509_VAL *validity;
  X509_NAME *subject;
  X509_PUBKEY *key;
  ASN1_BIT_STRING *issuerUID;            // [ 1 ] optional in v2
  ASN1_BIT_STRING *subjectUID;           // [ 2 ] optional in v2
  STACK_OF(X509_EXTENSION) *extensions;  // [ 3 ] optional in v3
  ASN1_ENCODING enc;
} X509_CINF;

// TODO(https://crbug.com/boringssl/407): This is not const because it contains
// an |X509_NAME|.
DECLARE_ASN1_FUNCTIONS(X509_CINF)

struct x509_st {
  X509_CINF *cert_info;
  X509_ALGOR *sig_alg;
  ASN1_BIT_STRING *signature;
  CRYPTO_refcount_t references;
  CRYPTO_EX_DATA ex_data;
  // These contain copies of various extension values
  long ex_pathlen;
  long ex_pcpathlen;
  uint32_t ex_flags;
  uint32_t ex_kusage;
  uint32_t ex_xkusage;
  uint32_t ex_nscert;
  ASN1_OCTET_STRING *skid;
  AUTHORITY_KEYID *akid;
  STACK_OF(DIST_POINT) *crldp;
  STACK_OF(GENERAL_NAME) *altname;
  NAME_CONSTRAINTS *nc;
  unsigned char cert_hash[SHA256_DIGEST_LENGTH];
  X509_CERT_AUX *aux;
  CRYPTO_MUTEX lock;
} /* X509 */;

typedef struct {
  ASN1_ENCODING enc;
  ASN1_INTEGER *version;
  X509_NAME *subject;
  X509_PUBKEY *pubkey;
  //  d=2 hl=2 l=  0 cons: cont: 00
  STACK_OF(X509_ATTRIBUTE) *attributes;  // [ 0 ]
} X509_REQ_INFO;

// TODO(https://crbug.com/boringssl/407): This is not const because it contains
// an |X509_NAME|.
DECLARE_ASN1_FUNCTIONS(X509_REQ_INFO)

struct X509_req_st {
  X509_REQ_INFO *req_info;
  X509_ALGOR *sig_alg;
  ASN1_BIT_STRING *signature;
} /* X509_REQ */;

struct x509_revoked_st {
  ASN1_INTEGER *serialNumber;
  ASN1_TIME *revocationDate;
  STACK_OF(X509_EXTENSION) /* optional */ *extensions;
  // Set up if indirect CRL
  STACK_OF(GENERAL_NAME) *issuer;
  // Revocation reason
  int reason;
} /* X509_REVOKED */;

typedef struct {
  ASN1_INTEGER *version;
  X509_ALGOR *sig_alg;
  X509_NAME *issuer;
  ASN1_TIME *lastUpdate;
  ASN1_TIME *nextUpdate;
  STACK_OF(X509_REVOKED) *revoked;
  STACK_OF(X509_EXTENSION) /* [0] */ *extensions;
  ASN1_ENCODING enc;
} X509_CRL_INFO;

// TODO(https://crbug.com/boringssl/407): This is not const because it contains
// an |X509_NAME|.
DECLARE_ASN1_FUNCTIONS(X509_CRL_INFO)

struct X509_crl_st {
  // actual signature
  X509_CRL_INFO *crl;
  X509_ALGOR *sig_alg;
  ASN1_BIT_STRING *signature;
  CRYPTO_refcount_t references;
  int flags;
  // Copies of various extensions
  AUTHORITY_KEYID *akid;
  ISSUING_DIST_POINT *idp;
  // Convenient breakdown of IDP
  int idp_flags;
  int idp_reasons;
  // CRL and base CRL numbers for delta processing
  ASN1_INTEGER *crl_number;
  ASN1_INTEGER *base_crl_number;
  unsigned char crl_hash[SHA256_DIGEST_LENGTH];
  STACK_OF(GENERAL_NAMES) *issuers;
} /* X509_CRL */;

struct X509_VERIFY_PARAM_st {
  char *name;
  int64_t check_time;               // POSIX time to use
  unsigned long inh_flags;          // Inheritance flags
  unsigned long flags;              // Various verify flags
  int purpose;                      // purpose to check untrusted certificates
  int trust;                        // trust setting to check
  int depth;                        // Verify depth
  STACK_OF(ASN1_OBJECT) *policies;  // Permissible policies
  // The following fields specify acceptable peer identities.
  STACK_OF(OPENSSL_STRING) *hosts;  // Set of acceptable names
  unsigned int hostflags;           // Flags to control matching features
  char *peername;                   // Matching hostname in peer certificate
  char *email;                      // If not NULL email address to match
  size_t emaillen;
  unsigned char *ip;     // If not NULL IP address to match
  size_t iplen;          // Length of IP address
  unsigned char poison;  // Fail all verifications at name checking
} /* X509_VERIFY_PARAM */;

struct x509_object_st {
  // one of the above types
  int type;
  union {
    char *ptr;
    X509 *x509;
    X509_CRL *crl;
    EVP_PKEY *pkey;
  } data;
} /* X509_OBJECT */;

// This is a static that defines the function interface
struct x509_lookup_method_st {
  const char *name;
  int (*new_item)(X509_LOOKUP *ctx);
  void (*free)(X509_LOOKUP *ctx);
  int (*init)(X509_LOOKUP *ctx);
  int (*shutdown)(X509_LOOKUP *ctx);
  int (*ctrl)(X509_LOOKUP *ctx, int cmd, const char *argc, long argl,
              char **ret);
  int (*get_by_subject)(X509_LOOKUP *ctx, int type, X509_NAME *name,
                        X509_OBJECT *ret);
} /* X509_LOOKUP_METHOD */;

// This is used to hold everything.  It is used for all certificate
// validation.  Once we have a certificate chain, the 'verify'
// function is then called to actually check the cert chain.
struct x509_store_st {
  // The following is a cache of trusted certs
  int cache;                    // if true, stash any hits
  STACK_OF(X509_OBJECT) *objs;  // Cache of all objects
  CRYPTO_MUTEX objs_lock;

  // These are external lookup methods
  STACK_OF(X509_LOOKUP) *get_cert_methods;

  X509_VERIFY_PARAM *param;

  // Callbacks for various operations
  X509_STORE_CTX_verify_fn verify;          // called to verify a certificate
  X509_STORE_CTX_verify_cb verify_cb;       // error callback
  X509_STORE_CTX_get_issuer_fn get_issuer;  // get issuers cert from ctx
  X509_STORE_CTX_check_issued_fn check_issued;  // check issued
  X509_STORE_CTX_check_revocation_fn
      check_revocation;                   // Check revocation status of chain
  X509_STORE_CTX_get_crl_fn get_crl;      // retrieve CRL
  X509_STORE_CTX_check_crl_fn check_crl;  // Check CRL validity
  X509_STORE_CTX_cert_crl_fn cert_crl;    // Check certificate against CRL
  X509_STORE_CTX_lookup_certs_fn lookup_certs;
  X509_STORE_CTX_lookup_crls_fn lookup_crls;
  X509_STORE_CTX_cleanup_fn cleanup;

  CRYPTO_refcount_t references;
} /* X509_STORE */;


// This is the functions plus an instance of the local variables.
struct x509_lookup_st {
  int init;                    // have we been started
  int skip;                    // don't use us.
  X509_LOOKUP_METHOD *method;  // the functions
  void *method_data;           // method data

  X509_STORE *store_ctx;  // who owns us
} /* X509_LOOKUP */;

// This is a used when verifying cert chains.  Since the
// gathering of the cert chain can take some time (and have to be
// 'retried', this needs to be kept and passed around.
struct x509_store_ctx_st {
  X509_STORE *ctx;

  // The following are set by the caller
  X509 *cert;                 // The cert to check
  STACK_OF(X509) *untrusted;  // chain of X509s - untrusted - passed in
  STACK_OF(X509_CRL) *crls;   // set of CRLs passed in

  X509_VERIFY_PARAM *param;
  void *other_ctx;  // Other info for use with get_issuer()

  // Callbacks for various operations
  X509_STORE_CTX_verify_fn verify;          // called to verify a certificate
  X509_STORE_CTX_verify_cb verify_cb;       // error callback
  X509_STORE_CTX_get_issuer_fn get_issuer;  // get issuers cert from ctx
  X509_STORE_CTX_check_issued_fn check_issued;  // check issued
  X509_STORE_CTX_check_revocation_fn
      check_revocation;                   // Check revocation status of chain
  X509_STORE_CTX_get_crl_fn get_crl;      // retrieve CRL
  X509_STORE_CTX_check_crl_fn check_crl;  // Check CRL validity
  X509_STORE_CTX_cert_crl_fn cert_crl;    // Check certificate against CRL
  X509_STORE_CTX_check_policy_fn check_policy;
  X509_STORE_CTX_lookup_certs_fn lookup_certs;
  X509_STORE_CTX_lookup_crls_fn lookup_crls;
  X509_STORE_CTX_cleanup_fn cleanup;

  // The following is built up
  int valid;               // if 0, rebuild chain
  int last_untrusted;      // index of last untrusted cert
  STACK_OF(X509) *chain;   // chain of X509s - built up and trusted

  // When something goes wrong, this is why
  int error_depth;
  int error;
  X509 *current_cert;
  X509 *current_issuer;   // cert currently being tested as valid issuer
  X509_CRL *current_crl;  // current CRL

  int current_crl_score;         // score of current CRL
  unsigned int current_reasons;  // Reason mask

  X509_STORE_CTX *parent;  // For CRL path validation: parent context

  CRYPTO_EX_DATA ex_data;
} /* X509_STORE_CTX */;

ASN1_TYPE *ASN1_generate_v3(const char *str, const X509V3_CTX *cnf);

int X509_CERT_AUX_print(BIO *bp, X509_CERT_AUX *x, int indent);


// RSA-PSS functions.

// x509_rsa_pss_to_ctx configures |ctx| for an RSA-PSS operation based on
// signature algorithm parameters in |sigalg| (which must have type
// |NID_rsassaPss|) and key |pkey|. It returns one on success and zero on
// error.
int x509_rsa_pss_to_ctx(EVP_MD_CTX *ctx, const X509_ALGOR *sigalg,
                        EVP_PKEY *pkey);

// x509_rsa_pss_to_ctx sets |algor| to the signature algorithm parameters for
// |ctx|, which must have been configured for an RSA-PSS signing operation. It
// returns one on success and zero on error.
int x509_rsa_ctx_to_pss(EVP_MD_CTX *ctx, X509_ALGOR *algor);

// x509_print_rsa_pss_params prints a human-readable representation of RSA-PSS
// parameters in |sigalg| to |bp|. It returns one on success and zero on
// error.
int x509_print_rsa_pss_params(BIO *bp, const X509_ALGOR *sigalg, int indent,
                              ASN1_PCTX *pctx);


// Signature algorithm functions.

// x509_digest_sign_algorithm encodes the signing parameters of |ctx| as an
// AlgorithmIdentifer and saves the result in |algor|. It returns one on
// success, or zero on error.
int x509_digest_sign_algorithm(EVP_MD_CTX *ctx, X509_ALGOR *algor);

// x509_digest_verify_init sets up |ctx| for a signature verification operation
// with public key |pkey| and parameters from |algor|. The |ctx| argument must
// have been initialised with |EVP_MD_CTX_init|. It returns one on success, or
// zero on error.
int x509_digest_verify_init(EVP_MD_CTX *ctx, const X509_ALGOR *sigalg,
                            EVP_PKEY *pkey);


// Path-building functions.

// X509_policy_check checks certificate policies in |certs|. |user_policies| is
// the user-initial-policy-set. |flags| is a set of |X509_V_FLAG_*| values to
// apply. It returns |X509_V_OK| on success and |X509_V_ERR_*| on error. It
// additionally sets |*out_current_cert| to the certificate where the error
// occurred. If the function succeeded, or the error applies to the entire
// chain, it sets |*out_current_cert| to NULL.
int X509_policy_check(const STACK_OF(X509) *certs,
                      const STACK_OF(ASN1_OBJECT) *user_policies,
                      unsigned long flags, X509 **out_current_cert);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_X509_INTERNAL_H
