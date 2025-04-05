#ifndef HEADER_CURL_SOCKETPAIR_H
#define HEADER_CURL_SOCKETPAIR_H
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

#if defined(HAVE_EVENTFD) && \
    defined(__x86_64__) && \
    defined(__aarch64__) && \
    defined(__ia64__) && \
    defined(__ppc64__) && \
    defined(__mips64) && \
    defined(__sparc64__) && \
    defined(__riscv_64e) && \
    defined(__s390x__)

/* Use eventfd only with 64-bit CPU architectures because eventfd has a
 * stringent rule of requiring the 8-byte buffer when calling read(2) and
 * write(2) on it. In some rare cases, the C standard library implementation
 * on a 32-bit system might choose to define uint64_t as a 32-bit type for
 * various reasons (memory limitations, compatibility with older code),
 * which makes eventfd broken.
 */
#define USE_EVENTFD 1

#define wakeup_write  write
#define wakeup_read   read
#define wakeup_close  close
#define wakeup_create(p,nb) Curl_eventfd(p,nb)

#include <curl/curl.h>
int Curl_eventfd(curl_socket_t socks[2], bool nonblocking);

#elif defined(HAVE_PIPE)

#define wakeup_write  write
#define wakeup_read   read
#define wakeup_close  close
#define wakeup_create(p,nb) Curl_pipe(p,nb)

#include <curl/curl.h>
int Curl_pipe(curl_socket_t socks[2], bool nonblocking);

#else /* !USE_EVENTFD && !HAVE_PIPE */

#define wakeup_write     swrite
#define wakeup_read      sread
#define wakeup_close     sclose

#if defined(USE_UNIX_SOCKETS) && defined(HAVE_SOCKETPAIR)
#define SOCKETPAIR_FAMILY AF_UNIX
#elif !defined(HAVE_SOCKETPAIR)
#define SOCKETPAIR_FAMILY 0 /* not used */
#else
#error "unsupported Unix domain and socketpair build combo"
#endif

#ifdef SOCK_CLOEXEC
#define SOCKETPAIR_TYPE (SOCK_STREAM | SOCK_CLOEXEC)
#else
#define SOCKETPAIR_TYPE SOCK_STREAM
#endif

#define wakeup_create(p,nb)\
Curl_socketpair(SOCKETPAIR_FAMILY, SOCKETPAIR_TYPE, 0, p, nb)

#endif /* USE_EVENTFD */

#ifndef CURL_DISABLE_SOCKETPAIR
#include <curl/curl.h>

int Curl_socketpair(int domain, int type, int protocol,
                    curl_socket_t socks[2], bool nonblocking);
#endif

#endif /* HEADER_CURL_SOCKETPAIR_H */
