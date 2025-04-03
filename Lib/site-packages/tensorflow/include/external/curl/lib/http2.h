#ifndef HEADER_CURL_HTTP2_H
#define HEADER_CURL_HTTP2_H
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

#ifdef USE_NGHTTP2
#include "http.h"

/* value for MAX_CONCURRENT_STREAMS we use until we get an updated setting
   from the peer */
#define DEFAULT_MAX_CONCURRENT_STREAMS 100

/*
 * Store nghttp2 version info in this buffer.
 */
void Curl_http2_ver(char *p, size_t len);

CURLcode Curl_http2_request_upgrade(struct dynbuf *req,
                                    struct Curl_easy *data);

/* returns true if the HTTP/2 stream error was HTTP_1_1_REQUIRED */
bool Curl_h2_http_1_1_error(struct Curl_easy *data);

bool Curl_conn_is_http2(const struct Curl_easy *data,
                        const struct connectdata *conn,
                        int sockindex);
bool Curl_http2_may_switch(struct Curl_easy *data,
                           struct connectdata *conn,
                           int sockindex);

CURLcode Curl_http2_switch(struct Curl_easy *data,
                           struct connectdata *conn, int sockindex);

CURLcode Curl_http2_switch_at(struct Curl_cfilter *cf, struct Curl_easy *data);

CURLcode Curl_http2_upgrade(struct Curl_easy *data,
                            struct connectdata *conn, int sockindex,
                            const char *ptr, size_t nread);

extern struct Curl_cftype Curl_cft_nghttp2;

#else /* USE_NGHTTP2 */

#define Curl_cf_is_http2(a,b) FALSE
#define Curl_conn_is_http2(a,b,c) FALSE
#define Curl_http2_may_switch(a,b,c) FALSE

#define Curl_http2_request_upgrade(x,y) CURLE_UNSUPPORTED_PROTOCOL
#define Curl_http2_switch(a,b,c)        CURLE_UNSUPPORTED_PROTOCOL
#define Curl_http2_upgrade(a,b,c,d,e)   CURLE_UNSUPPORTED_PROTOCOL
#define Curl_h2_http_1_1_error(x) 0
#endif

#endif /* HEADER_CURL_HTTP2_H */
