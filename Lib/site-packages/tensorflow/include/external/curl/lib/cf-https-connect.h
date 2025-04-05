#ifndef HEADER_CURL_CF_HTTP_H
#define HEADER_CURL_CF_HTTP_H
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

#if !defined(CURL_DISABLE_HTTP) && !defined(USE_HYPER)

struct Curl_cfilter;
struct Curl_easy;
struct connectdata;
struct Curl_cftype;
struct Curl_dns_entry;

extern struct Curl_cftype Curl_cft_http_connect;

CURLcode Curl_cf_http_connect_add(struct Curl_easy *data,
                                  struct connectdata *conn,
                                  int sockindex,
                                  const struct Curl_dns_entry *remotehost,
                                  bool try_h3, bool try_h21);

CURLcode
Curl_cf_http_connect_insert_after(struct Curl_cfilter *cf_at,
                                  struct Curl_easy *data,
                                  const struct Curl_dns_entry *remotehost,
                                  bool try_h3, bool try_h21);


CURLcode Curl_cf_https_setup(struct Curl_easy *data,
                             struct connectdata *conn,
                             int sockindex,
                             const struct Curl_dns_entry *remotehost);


#endif /* !defined(CURL_DISABLE_HTTP) && !defined(USE_HYPER) */
#endif /* HEADER_CURL_CF_HTTP_H */
