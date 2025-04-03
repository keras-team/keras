#ifndef HEADER_CURL_MBEDTLS_THREADLOCK_H
#define HEADER_CURL_MBEDTLS_THREADLOCK_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Daniel Stenberg, <daniel@haxx.se>, et al.
 * Copyright (C) Hoi-Ho Chan, <hoiho.chan@gmail.com>
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

#ifdef USE_MBEDTLS

#if (defined(USE_THREADS_POSIX) && defined(HAVE_PTHREAD_H)) || \
    defined(_WIN32)

int Curl_mbedtlsthreadlock_thread_setup(void);
int Curl_mbedtlsthreadlock_thread_cleanup(void);
int Curl_mbedtlsthreadlock_lock_function(int n);
int Curl_mbedtlsthreadlock_unlock_function(int n);

#else

#define Curl_mbedtlsthreadlock_thread_setup() 1
#define Curl_mbedtlsthreadlock_thread_cleanup() 1
#define Curl_mbedtlsthreadlock_lock_function(x) 1
#define Curl_mbedtlsthreadlock_unlock_function(x) 1

#endif /* (USE_THREADS_POSIX && HAVE_PTHREAD_H) || _WIN32 */

#endif /* USE_MBEDTLS */

#endif /* HEADER_CURL_MBEDTLS_THREADLOCK_H */
