#ifndef HEADER_CURL_CONFIG_MAC_H
#define HEADER_CURL_CONFIG_MAC_H
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

/* =================================================================== */
/*                Hand crafted config file for Mac OS 9                */
/* =================================================================== */
/*   On macOS you must run configure to generate curl_config.h file    */
/* =================================================================== */

#ifndef CURL_OS
#define CURL_OS "mac"
#endif

#include <ConditionalMacros.h>
#if TYPE_LONGLONG
#define HAVE_LONGLONG           1
#endif

/* Define if you want the built-in manual */
#define USE_MANUAL              1

#define HAVE_NETINET_IN_H       1
#define HAVE_SYS_SOCKET_H       1
#define HAVE_NETDB_H            1
#define HAVE_ARPA_INET_H        1
#define HAVE_UNISTD_H           1
#define HAVE_NET_IF_H           1
#define HAVE_SYS_TYPES_H        1
#define HAVE_GETTIMEOFDAY       1
#define HAVE_FCNTL_H            1
#define HAVE_SYS_STAT_H         1
#define HAVE_UTIME_H            1
#define HAVE_SYS_TIME_H         1
#define HAVE_SYS_UTIME_H        1
#define HAVE_SYS_IOCTL_H        1
#define HAVE_ALARM              1
#define HAVE_FTRUNCATE          1
#define HAVE_UTIME              1
#define HAVE_SELECT             1
#define HAVE_SOCKET             1
#define HAVE_STRUCT_TIMEVAL     1

#define HAVE_SIGACTION          1

#ifdef MACOS_SSL_SUPPORT
#  define USE_OPENSSL           1
#endif

#define CURL_DISABLE_LDAP       1

#define HAVE_IOCTL_FIONBIO      1

#define SIZEOF_INT              4
#define SIZEOF_LONG             4
#define SIZEOF_SIZE_T           4
#ifdef HAVE_LONGLONG
#define SIZEOF_CURL_OFF_T       8
#else
#define SIZEOF_CURL_OFF_T       4
#endif

#define HAVE_RECV 1
#define RECV_TYPE_ARG1 int
#define RECV_TYPE_ARG2 void *
#define RECV_TYPE_ARG3 size_t
#define RECV_TYPE_ARG4 int
#define RECV_TYPE_RETV ssize_t

#define HAVE_SEND 1
#define SEND_TYPE_ARG1 int
#define SEND_QUAL_ARG2 const
#define SEND_TYPE_ARG2 void *
#define SEND_TYPE_ARG3 size_t
#define SEND_TYPE_ARG4 int
#define SEND_TYPE_RETV ssize_t

#define HAVE_EXTRA_STRICMP_H 1
#define HAVE_EXTRA_STRDUP_H  1

#endif /* HEADER_CURL_CONFIG_MAC_H */
