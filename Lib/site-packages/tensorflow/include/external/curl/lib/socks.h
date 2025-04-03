#ifndef HEADER_CURL_SOCKS_H
#define HEADER_CURL_SOCKS_H
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

#ifdef CURL_DISABLE_PROXY
#define Curl_SOCKS4(a,b,c,d,e) CURLE_NOT_BUILT_IN
#define Curl_SOCKS5(a,b,c,d,e,f) CURLE_NOT_BUILT_IN
#define Curl_SOCKS_getsock(x,y,z) 0
#else
/*
 * Helper read-from-socket functions. Does the same as Curl_read() but it
 * blocks until all bytes amount of buffersize will be read. No more, no less.
 *
 * This is STUPID BLOCKING behavior
 */
int Curl_blockread_all(struct Curl_cfilter *cf,
                       struct Curl_easy *data,
                       char *buf,
                       ssize_t buffersize,
                       ssize_t *n);

#if defined(HAVE_GSSAPI) || defined(USE_WINDOWS_SSPI)
/*
 * This function handles the SOCKS5 GSS-API negotiation and initialization
 */
CURLcode Curl_SOCKS5_gssapi_negotiate(struct Curl_cfilter *cf,
                                      struct Curl_easy *data);
#endif

CURLcode Curl_cf_socks_proxy_insert_after(struct Curl_cfilter *cf_at,
                                          struct Curl_easy *data);

extern struct Curl_cftype Curl_cft_socks_proxy;

#endif /* CURL_DISABLE_PROXY */

#endif  /* HEADER_CURL_SOCKS_H */
