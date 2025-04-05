#ifndef HEADER_CURL_IDN_H
#define HEADER_CURL_IDN_H
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

bool Curl_is_ASCII_name(const char *hostname);
CURLcode Curl_idnconvert_hostname(struct hostname *host);
#if defined(USE_LIBIDN2) || defined(USE_WIN32_IDN) || defined(USE_APPLE_IDN)
#define USE_IDN
void Curl_free_idnconverted_hostname(struct hostname *host);
CURLcode Curl_idn_decode(const char *input, char **output);
CURLcode Curl_idn_encode(const char *input, char **output);

#else
#define Curl_free_idnconverted_hostname(x)
#define Curl_idn_decode(x) NULL
#endif
#endif /* HEADER_CURL_IDN_H */
