#ifndef HEADER_CURL_RAND_H
#define HEADER_CURL_RAND_H
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

CURLcode Curl_rand_bytes(struct Curl_easy *data,
#ifdef DEBUGBUILD
                         bool allow_env_override,
#endif
                         unsigned char *rnd, size_t num);

#ifdef DEBUGBUILD
#define Curl_rand(a,b,c)   Curl_rand_bytes((a), TRUE, (b), (c))
#else
#define Curl_rand(a,b,c)   Curl_rand_bytes((a), (b), (c))
#endif

/*
 * Curl_rand_hex() fills the 'rnd' buffer with a given 'num' size with random
 * hexadecimal digits PLUS a null-terminating byte. It must be an odd number
 * size.
 */
CURLcode Curl_rand_hex(struct Curl_easy *data, unsigned char *rnd,
                       size_t num);

/*
 * Curl_rand_alnum() fills the 'rnd' buffer with a given 'num' size with random
 * alphanumerical chars PLUS a null-terminating byte.
 */
CURLcode Curl_rand_alnum(struct Curl_easy *data, unsigned char *rnd,
                         size_t num);

#ifdef _WIN32
/* Random generator shared between the Schannel vtls and Curl_rand*()
   functions */
CURLcode Curl_win32_random(unsigned char *entropy, size_t length);
#endif

#endif /* HEADER_CURL_RAND_H */
