#ifndef HEADER_CURL_SETOPT_H
#define HEADER_CURL_SETOPT_H
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

CURLcode Curl_setstropt(char **charp, const char *s) WARN_UNUSED_RESULT;
CURLcode Curl_setblobopt(struct curl_blob **blobp,
                         const struct curl_blob *blob) WARN_UNUSED_RESULT;
CURLcode Curl_vsetopt(struct Curl_easy *data, CURLoption option, va_list arg)
  WARN_UNUSED_RESULT;

#endif /* HEADER_CURL_SETOPT_H */
