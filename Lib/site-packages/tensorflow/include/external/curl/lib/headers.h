#ifndef HEADER_CURL_HEADER_H
#define HEADER_CURL_HEADER_H
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

#if !defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_HEADERS_API)

struct Curl_header_store {
  struct Curl_llist_node node;
  char *name; /* points into 'buffer' */
  char *value; /* points into 'buffer */
  int request; /* 0 is the first request, then 1.. 2.. */
  unsigned char type; /* CURLH_* defines */
  char buffer[1]; /* this is the raw header blob */
};

/*
 * Initialize header collecting for a transfer.
 * Will add a client writer that catches CLIENTWRITE_HEADER writes.
 */
CURLcode Curl_headers_init(struct Curl_easy *data);

/*
 * Curl_headers_push() gets passed a full header to store.
 */
CURLcode Curl_headers_push(struct Curl_easy *data, const char *header,
                           unsigned char type);

/*
 * Curl_headers_cleanup(). Free all stored headers and associated memory.
 */
CURLcode Curl_headers_cleanup(struct Curl_easy *data);

#else
#define Curl_headers_init(x) CURLE_OK
#define Curl_headers_push(x,y,z) CURLE_OK
#define Curl_headers_cleanup(x) Curl_nop_stmt
#endif

#endif /* HEADER_CURL_HEADER_H */
