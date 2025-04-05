#ifndef HEADER_CURL_VQUIC_CURL_OSSLQ_H
#define HEADER_CURL_VQUIC_CURL_OSSLQ_H
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

#if defined(USE_OPENSSL_QUIC) && defined(USE_NGHTTP3)

#ifdef HAVE_NETINET_UDP_H
#include <netinet/udp.h>
#endif

struct Curl_cfilter;

#include "urldata.h"

void Curl_osslq_ver(char *p, size_t len);

CURLcode Curl_cf_osslq_create(struct Curl_cfilter **pcf,
                              struct Curl_easy *data,
                              struct connectdata *conn,
                              const struct Curl_addrinfo *ai);

bool Curl_conn_is_osslq(const struct Curl_easy *data,
                        const struct connectdata *conn,
                        int sockindex);
#endif

#endif /* HEADER_CURL_VQUIC_CURL_OSSLQ_H */
