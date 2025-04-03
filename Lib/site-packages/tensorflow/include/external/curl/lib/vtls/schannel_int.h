#ifndef HEADER_CURL_SCHANNEL_INT_H
#define HEADER_CURL_SCHANNEL_INT_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Marc Hoersken, <info@marc-hoersken.de>, et al.
 * Copyright (C) Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 * SPDX-License-Identifier: curl
 *
 ***************************************************************************/
#include "curl_setup.h"

#ifdef USE_SCHANNEL

#include "vtls.h"

#if (defined(__MINGW32__) || defined(CERT_CHAIN_REVOCATION_CHECK_CHAIN)) \
  && !defined(CURL_WINDOWS_UWP)
#define HAS_MANUAL_VERIFY_API
#endif

#if defined(CryptStringToBinary) && defined(CRYPT_STRING_HEX)   \
  && !defined(DISABLE_SCHANNEL_CLIENT_CERT)
#define HAS_CLIENT_CERT_PATH
#endif

#ifndef CRYPT_DECODE_NOCOPY_FLAG
#define CRYPT_DECODE_NOCOPY_FLAG 0x1
#endif

#ifndef CRYPT_DECODE_ALLOC_FLAG
#define CRYPT_DECODE_ALLOC_FLAG 0x8000
#endif

#ifndef CERT_ALT_NAME_DNS_NAME
#define CERT_ALT_NAME_DNS_NAME 3
#endif

#ifndef CERT_ALT_NAME_IP_ADDRESS
#define CERT_ALT_NAME_IP_ADDRESS 8
#endif

#if defined(_MSC_VER) && (_MSC_VER <= 1600)
/* Workaround for warning:
   'type cast' : conversion from 'int' to 'LPCSTR' of greater size */
#undef CERT_STORE_PROV_MEMORY
#undef CERT_STORE_PROV_SYSTEM_A
#undef CERT_STORE_PROV_SYSTEM_W
#define CERT_STORE_PROV_MEMORY    ((LPCSTR)(size_t)2)
#define CERT_STORE_PROV_SYSTEM_A  ((LPCSTR)(size_t)9)
#define CERT_STORE_PROV_SYSTEM_W  ((LPCSTR)(size_t)10)
#endif

#ifndef SCH_CREDENTIALS_VERSION

#define SCH_CREDENTIALS_VERSION  0x00000005

typedef enum _eTlsAlgorithmUsage
{
    TlsParametersCngAlgUsageKeyExchange,
    TlsParametersCngAlgUsageSignature,
    TlsParametersCngAlgUsageCipher,
    TlsParametersCngAlgUsageDigest,
    TlsParametersCngAlgUsageCertSig
} eTlsAlgorithmUsage;

typedef struct _CRYPTO_SETTINGS
{
    eTlsAlgorithmUsage  eAlgorithmUsage;
    UNICODE_STRING      strCngAlgId;
    DWORD               cChainingModes;
    PUNICODE_STRING     rgstrChainingModes;
    DWORD               dwMinBitLength;
    DWORD               dwMaxBitLength;
} CRYPTO_SETTINGS, * PCRYPTO_SETTINGS;

typedef struct _TLS_PARAMETERS
{
    DWORD               cAlpnIds;
    PUNICODE_STRING     rgstrAlpnIds;
    DWORD               grbitDisabledProtocols;
    DWORD               cDisabledCrypto;
    PCRYPTO_SETTINGS    pDisabledCrypto;
    DWORD               dwFlags;
} TLS_PARAMETERS, * PTLS_PARAMETERS;

typedef struct _SCH_CREDENTIALS
{
    DWORD               dwVersion;
    DWORD               dwCredFormat;
    DWORD               cCreds;
    PCCERT_CONTEXT* paCred;
    HCERTSTORE          hRootStore;

    DWORD               cMappers;
    struct _HMAPPER **aphMappers;

    DWORD               dwSessionLifespan;
    DWORD               dwFlags;
    DWORD               cTlsParameters;
    PTLS_PARAMETERS     pTlsParameters;
} SCH_CREDENTIALS, * PSCH_CREDENTIALS;

#define SCH_CRED_MAX_SUPPORTED_PARAMETERS 16
#define SCH_CRED_MAX_SUPPORTED_ALPN_IDS 16
#define SCH_CRED_MAX_SUPPORTED_CRYPTO_SETTINGS 16
#define SCH_CRED_MAX_SUPPORTED_CHAINING_MODES 16

#endif /* SCH_CREDENTIALS_VERSION */

struct Curl_schannel_cred {
  CredHandle cred_handle;
  TimeStamp time_stamp;
  TCHAR *sni_hostname;
#ifdef HAS_CLIENT_CERT_PATH
  HCERTSTORE client_cert_store;
#endif
  int refcount;
};

struct Curl_schannel_ctxt {
  CtxtHandle ctxt_handle;
  TimeStamp time_stamp;
};

struct schannel_ssl_backend_data {
  struct Curl_schannel_cred *cred;
  struct Curl_schannel_ctxt *ctxt;
  SecPkgContext_StreamSizes stream_sizes;
  size_t encdata_length, decdata_length;
  size_t encdata_offset, decdata_offset;
  unsigned char *encdata_buffer, *decdata_buffer;
  /* encdata_is_incomplete: if encdata contains only a partial record that
     cannot be decrypted without another recv() (that is, status is
     SEC_E_INCOMPLETE_MESSAGE) then set this true. after an recv() adds
     more bytes into encdata then set this back to false. */
  bool encdata_is_incomplete;
  unsigned long req_flags, ret_flags;
  CURLcode recv_unrecoverable_err; /* schannel_recv had an unrecoverable err */
  bool recv_sspi_close_notify; /* true if connection closed by close_notify */
  bool recv_connection_closed; /* true if connection closed, regardless how */
  bool recv_renegotiating;     /* true if recv is doing renegotiation */
  bool use_alpn; /* true if ALPN is used for this connection */
#ifdef HAS_MANUAL_VERIFY_API
  bool use_manual_cred_validation; /* true if manual cred validation is used */
#endif
  BIT(sent_shutdown);
};

/* key to use at `multi->proto_hash` */
#define MPROTO_SCHANNEL_CERT_SHARE_KEY   "tls:schannel:cert:share"

struct schannel_cert_share {
  unsigned char CAinfo_blob_digest[CURL_SHA256_DIGEST_LENGTH];
  size_t CAinfo_blob_size;           /* CA info blob size */
  char *CAfile;                      /* CAfile path used to generate
                                        certificate store */
  HCERTSTORE cert_store;             /* cached certificate store or
                                        NULL if none */
  struct curltime time;              /* when the cached store was created */
};

/*
* size of the structure: 20 bytes.
*/
struct num_ip_data {
  DWORD size; /* 04 bytes */
  union {
    struct in_addr  ia;  /* 04 bytes */
    struct in6_addr ia6; /* 16 bytes */
  } bData;
};

HCERTSTORE Curl_schannel_get_cached_cert_store(struct Curl_cfilter *cf,
                                               const struct Curl_easy *data);

bool Curl_schannel_set_cached_cert_store(struct Curl_cfilter *cf,
                                         const struct Curl_easy *data,
                                         HCERTSTORE cert_store);

#endif /* USE_SCHANNEL */
#endif /* HEADER_CURL_SCHANNEL_INT_H */
