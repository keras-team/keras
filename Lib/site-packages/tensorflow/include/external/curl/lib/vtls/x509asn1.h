#ifndef HEADER_CURL_X509ASN1_H
#define HEADER_CURL_X509ASN1_H

/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
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

#if defined(USE_GNUTLS) || defined(USE_WOLFSSL) || \
  defined(USE_SCHANNEL) || defined(USE_SECTRANSP) || \
  defined(USE_MBEDTLS)

#include "cfilters.h"
#include "urldata.h"

/*
 * Types.
 */

/* ASN.1 parsed element. */
struct Curl_asn1Element {
  const char *header;         /* Pointer to header byte. */
  const char *beg;            /* Pointer to element data. */
  const char *end;            /* Pointer to 1st byte after element. */
  unsigned char class;        /* ASN.1 element class. */
  unsigned char tag;          /* ASN.1 element tag. */
  bool          constructed;  /* Element is constructed. */
};

/* X509 certificate: RFC 5280. */
struct Curl_X509certificate {
  struct Curl_asn1Element certificate;
  struct Curl_asn1Element version;
  struct Curl_asn1Element serialNumber;
  struct Curl_asn1Element signatureAlgorithm;
  struct Curl_asn1Element signature;
  struct Curl_asn1Element issuer;
  struct Curl_asn1Element notBefore;
  struct Curl_asn1Element notAfter;
  struct Curl_asn1Element subject;
  struct Curl_asn1Element subjectPublicKeyInfo;
  struct Curl_asn1Element subjectPublicKeyAlgorithm;
  struct Curl_asn1Element subjectPublicKey;
  struct Curl_asn1Element issuerUniqueID;
  struct Curl_asn1Element subjectUniqueID;
  struct Curl_asn1Element extensions;
};

/*
 * Prototypes.
 */

int Curl_parseX509(struct Curl_X509certificate *cert,
                   const char *beg, const char *end);
CURLcode Curl_extract_certinfo(struct Curl_easy *data, int certnum,
                               const char *beg, const char *end);
CURLcode Curl_verifyhost(struct Curl_cfilter *cf, struct Curl_easy *data,
                         const char *beg, const char *end);

#ifdef UNITTESTS
#if defined(USE_GNUTLS) || defined(USE_SCHANNEL) || defined(USE_SECTRANSP) || \
  defined(USE_MBEDTLS)

/* used by unit1656.c */
CURLcode Curl_x509_GTime2str(struct dynbuf *store,
                             const char *beg, const char *end);
#endif
#endif

#endif /* USE_GNUTLS or USE_WOLFSSL or USE_SCHANNEL or USE_SECTRANSP */
#endif /* HEADER_CURL_X509ASN1_H */
