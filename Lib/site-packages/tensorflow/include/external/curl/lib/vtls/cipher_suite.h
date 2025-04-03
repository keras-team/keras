#ifndef HEADER_CURL_CIPHER_SUITE_H
#define HEADER_CURL_CIPHER_SUITE_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Jan Venekamp, <jan@venekamp.net>
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

#if defined(USE_SECTRANSP) || defined(USE_MBEDTLS) || \
    defined(USE_BEARSSL) || defined(USE_RUSTLS)
#include <stdint.h>

/* Lookup IANA id for cipher suite string, returns 0 if not recognized */
uint16_t Curl_cipher_suite_lookup_id(const char *cs_str, size_t cs_len);

/* Walk over cipher suite string, update str and end pointers to next
   cipher suite in string, returns IANA id of that suite if recognized */
uint16_t Curl_cipher_suite_walk_str(const char **str, const char **end);

/* Copy openssl or RFC name for cipher suite in supplied buffer.
   Caller is responsible to supply sufficiently large buffer (size
   of 64 should suffice), excess bytes are silently truncated. */
int Curl_cipher_suite_get_str(uint16_t id, char *buf, size_t buf_size,
                              bool prefer_rfc);

#endif /* defined(USE_SECTRANSP) || defined(USE_MBEDTLS) || \
          defined(USE_BEARSSL) || defined(USE_RUSTLS) */
#endif /* HEADER_CURL_CIPHER_SUITE_H */
