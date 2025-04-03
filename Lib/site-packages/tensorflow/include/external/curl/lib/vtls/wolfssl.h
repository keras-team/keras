#ifndef HEADER_CURL_WOLFSSL_H
#define HEADER_CURL_WOLFSSL_H
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

#ifdef USE_WOLFSSL

#include "urldata.h"

struct WOLFSSL;
typedef struct WOLFSSL WOLFSSL;
struct WOLFSSL_CTX;
typedef struct WOLFSSL_CTX WOLFSSL_CTX;
struct WOLFSSL_SESSION;
typedef struct WOLFSSL_SESSION WOLFSSL_SESSION;

extern const struct Curl_ssl Curl_ssl_wolfssl;

struct wolfssl_ctx {
  WOLFSSL_CTX *ctx;
  WOLFSSL     *handle;
  CURLcode    io_result;   /* result of last BIO cfilter operation */
  int io_send_blocked_len; /* length of last BIO write that EAGAINed */
  BIT(x509_store_setup);   /* x509 store has been set up */
  BIT(shutting_down);      /* TLS is being shut down */
};

CURLcode Curl_wssl_setup_x509_store(struct Curl_cfilter *cf,
                                    struct Curl_easy *data,
                                    struct wolfssl_ctx *wssl);

CURLcode wssl_setup_session(struct Curl_cfilter *cf,
                            struct Curl_easy *data,
                            struct wolfssl_ctx *wss,
                            struct ssl_peer *peer);

CURLcode wssl_cache_session(struct Curl_cfilter *cf,
                            struct Curl_easy *data,
                            struct ssl_peer *peer,
                            WOLFSSL_SESSION *session);


#endif /* USE_WOLFSSL */
#endif /* HEADER_CURL_WOLFSSL_H */
