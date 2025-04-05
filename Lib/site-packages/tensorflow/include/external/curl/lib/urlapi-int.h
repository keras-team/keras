#ifndef HEADER_CURL_URLAPI_INT_H
#define HEADER_CURL_URLAPI_INT_H
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

size_t Curl_is_absolute_url(const char *url, char *buf, size_t buflen,
                            bool guess_scheme);

CURLUcode Curl_url_set_authority(CURLU *u, const char *authority);

#ifdef UNITTESTS
UNITTEST CURLUcode Curl_parse_port(struct Curl_URL *u, struct dynbuf *host,
                                   bool has_scheme);
#endif

#endif /* HEADER_CURL_URLAPI_INT_H */
