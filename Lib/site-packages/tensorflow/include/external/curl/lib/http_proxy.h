#ifndef HEADER_CURL_HTTP_PROXY_H
#define HEADER_CURL_HTTP_PROXY_H
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

#if !defined(CURL_DISABLE_PROXY) && !defined(CURL_DISABLE_HTTP)

#include "urldata.h"

CURLcode Curl_http_proxy_get_destination(struct Curl_cfilter *cf,
                                         const char **phostname,
                                         int *pport, bool *pipv6_ip);

CURLcode Curl_http_proxy_create_CONNECT(struct httpreq **preq,
                                        struct Curl_cfilter *cf,
                                        struct Curl_easy *data,
                                        int http_version_major);

/* Default proxy timeout in milliseconds */
#define PROXY_TIMEOUT (3600*1000)

void Curl_cf_http_proxy_get_host(struct Curl_cfilter *cf,
                                 struct Curl_easy *data,
                                 const char **phost,
                                 const char **pdisplay_host,
                                 int *pport);

CURLcode Curl_cf_http_proxy_insert_after(struct Curl_cfilter *cf_at,
                                         struct Curl_easy *data);

extern struct Curl_cftype Curl_cft_http_proxy;

#endif /* !CURL_DISABLE_PROXY  && !CURL_DISABLE_HTTP */

#define IS_HTTPS_PROXY(t) (((t) == CURLPROXY_HTTPS) ||  \
                           ((t) == CURLPROXY_HTTPS2))

#endif /* HEADER_CURL_HTTP_PROXY_H */
