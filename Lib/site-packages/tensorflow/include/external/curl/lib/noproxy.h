#ifndef HEADER_CURL_NOPROXY_H
#define HEADER_CURL_NOPROXY_H
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

#ifndef CURL_DISABLE_PROXY

#ifdef UNITTESTS

UNITTEST bool Curl_cidr4_match(const char *ipv4,    /* 1.2.3.4 address */
                               const char *network, /* 1.2.3.4 address */
                               unsigned int bits);
UNITTEST bool Curl_cidr6_match(const char *ipv6,
                               const char *network,
                               unsigned int bits);
#endif

bool Curl_check_noproxy(const char *name, const char *no_proxy);
#endif

#endif /* HEADER_CURL_NOPROXY_H */
