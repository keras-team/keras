#ifndef HEADER_CURL_CONFIG_DOS_H
#define HEADER_CURL_CONFIG_DOS_H
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


/* ================================================================ */
/*       lib/config-dos.h - Hand crafted config file for DOS        */
/* ================================================================ */

#ifndef CURL_OS
#if defined(DJGPP)
  #define CURL_OS  "MSDOS/djgpp"
#elif defined(__HIGHC__)
  #define CURL_OS  "MSDOS/HighC"
#else
  #define CURL_OS  "MSDOS/?"
#endif
#endif

#define PACKAGE  "curl"

#define USE_MANUAL 1

#define HAVE_ARPA_INET_H       1
#define HAVE_FCNTL_H           1
#define HAVE_FREEADDRINFO      1
#define HAVE_GETADDRINFO       1
#define HAVE_GETTIMEOFDAY      1
#define HAVE_IO_H              1
#define HAVE_IOCTL_FIONBIO     1
#define HAVE_IOCTLSOCKET       1
#define HAVE_IOCTLSOCKET_FIONBIO   1
#define HAVE_LOCALE_H          1
#define HAVE_LONGLONG          1
#define HAVE_NETDB_H           1
#define HAVE_NETINET_IN_H      1
#define HAVE_NETINET_TCP_H     1
#define HAVE_NET_IF_H          1
#define HAVE_RECV              1
#define HAVE_SELECT            1
#define HAVE_SEND              1
#define HAVE_SETLOCALE         1
#define HAVE_SETMODE           1
#define HAVE_SIGNAL            1
#define HAVE_SOCKET            1
#define HAVE_STRDUP            1
#define HAVE_STRICMP           1
#define HAVE_STRTOLL           1
#define HAVE_STRUCT_TIMEVAL    1
#define HAVE_SYS_IOCTL_H       1
#define HAVE_SYS_SOCKET_H      1
#define HAVE_SYS_STAT_H        1
#define HAVE_SYS_TYPES_H       1
#define HAVE_UNISTD_H          1

#define NEED_MALLOC_H          1

#define SIZEOF_INT             4
#define SIZEOF_LONG            4
#define SIZEOF_SIZE_T          4
#define SIZEOF_CURL_OFF_T      8
#define STDC_HEADERS           1

/* Qualifiers for send() and recv() */

#define SEND_TYPE_ARG1         int
#define SEND_QUAL_ARG2         const
#define SEND_TYPE_ARG2         void *
#define SEND_TYPE_ARG3         int
#define SEND_TYPE_ARG4         int
#define SEND_TYPE_RETV         int

#define RECV_TYPE_ARG1         int
#define RECV_TYPE_ARG2         void *
#define RECV_TYPE_ARG3         int
#define RECV_TYPE_ARG4         int
#define RECV_TYPE_RETV         int

#define BSD

/* CURLDEBUG definition enables memory tracking */
/* #define CURLDEBUG */

/* to disable LDAP */
#define CURL_DISABLE_LDAP        1

#define in_addr_t  u_long

#if defined(__HIGHC__) || \
    (defined(__GNUC__) && (__GNUC__ < 4))
  #define ssize_t  int
#endif

/* Target HAVE_x section */

#if defined(DJGPP)
  #define HAVE_BASENAME   1
  #define HAVE_STRCASECMP 1
  #define HAVE_SIGACTION  1
  #define HAVE_SIGSETJMP  1
  #define HAVE_SYS_TIME_H 1
  #define HAVE_TERMIOS_H  1

#elif defined(__HIGHC__)
  #define HAVE_SYS_TIME_H 1
  #define strerror(e) strerror_s_((e))
#endif

#ifdef MSDOS  /* Watt-32 */
  #define HAVE_CLOSE_S    1
#endif

#undef word
#undef byte

#endif /* HEADER_CURL_CONFIG_DOS_H */
