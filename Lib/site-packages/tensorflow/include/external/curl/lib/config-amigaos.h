#ifndef HEADER_CURL_CONFIG_AMIGAOS_H
#define HEADER_CURL_CONFIG_AMIGAOS_H
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
/*               Hand crafted config file for AmigaOS               */
/* ================================================================ */

#ifdef __AMIGA__ /* Any AmigaOS flavour */

#define HAVE_ARPA_INET_H 1
#define HAVE_CLOSESOCKET_CAMEL 1
#define HAVE_IOCTLSOCKET_CAMEL 1
#define HAVE_IOCTLSOCKET_CAMEL_FIONBIO 1
#define HAVE_LONGLONG 1
#define HAVE_NETDB_H 1
#define HAVE_NETINET_IN_H 1
#define HAVE_NET_IF_H 1
#define HAVE_PWD_H 1
#define HAVE_SELECT 1
#define HAVE_SIGNAL 1
#define HAVE_SOCKET 1
#define HAVE_STRCASECMP 1
#define HAVE_STRDUP 1
#define HAVE_STRICMP 1
#define HAVE_STRINGS_H 1
#define HAVE_STRUCT_TIMEVAL 1
#define HAVE_SYS_PARAM_H 1
#define HAVE_SYS_SOCKET_H 1
#define HAVE_SYS_SOCKIO_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_UNISTD_H 1
#define HAVE_UTIME 1
#define HAVE_UTIME_H 1
#define HAVE_WRITABLE_ARGV 1
#define HAVE_SYS_IOCTL_H 1

#define NEED_MALLOC_H 1

#define SIZEOF_INT 4
#define SIZEOF_SIZE_T 4

#ifndef SIZEOF_CURL_OFF_T
#define SIZEOF_CURL_OFF_T 8
#endif

#define USE_MANUAL 1
#define CURL_DISABLE_LDAP 1

#ifndef CURL_OS
#define CURL_OS "AmigaOS"
#endif

#define PACKAGE "curl"
#define PACKAGE_BUGREPORT "a suitable mailing list: https://curl.se/mail/"
#define PACKAGE_NAME "curl"
#define PACKAGE_STRING "curl -"
#define PACKAGE_TARNAME "curl"
#define PACKAGE_VERSION "-"

#if defined(USE_AMISSL)
#define CURL_CA_PATH "AmiSSL:Certs"
#elif defined(__MORPHOS__)
#define CURL_CA_BUNDLE "MOSSYS:Data/SSL/curl-ca-bundle.crt"
#else
#define CURL_CA_BUNDLE "s:curl-ca-bundle.crt"
#endif

#define STDC_HEADERS 1

#define in_addr_t int

#ifndef F_OK
#  define F_OK 0
#endif

#ifndef O_RDONLY
#  define O_RDONLY 0x0000
#endif

#ifndef LONG_MAX
#  define LONG_MAX 0x7fffffffL
#endif

#ifndef LONG_MIN
#  define LONG_MIN (-0x7fffffffL-1)
#endif

#define HAVE_RECV 1
#define RECV_TYPE_ARG1 long
#define RECV_TYPE_ARG2 char *
#define RECV_TYPE_ARG3 long
#define RECV_TYPE_ARG4 long
#define RECV_TYPE_RETV long

#define HAVE_SEND 1
#define SEND_TYPE_ARG1 int
#define SEND_QUAL_ARG2 const
#define SEND_TYPE_ARG2 char *
#define SEND_TYPE_ARG3 int
#define SEND_TYPE_ARG4 int
#define SEND_TYPE_RETV int

#endif /* __AMIGA__ */
#endif /* HEADER_CURL_CONFIG_AMIGAOS_H */
