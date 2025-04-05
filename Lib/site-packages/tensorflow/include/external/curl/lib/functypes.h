#ifndef HEADER_CURL_FUNCTYPES_H
#define HEADER_CURL_FUNCTYPES_H
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

/* defaults:

   ssize_t recv(int, void *, size_t, int);
   ssize_t send(int, const void *, size_t, int);

   If other argument or return types are needed:

   1. For systems that run configure or cmake, the alternatives are provided
      here.
   2. For systems with config-*.h files, define them there.
*/

#ifdef _WIN32
/* int recv(SOCKET, char *, int, int) */
#define RECV_TYPE_ARG1 SOCKET
#define RECV_TYPE_ARG2 char *
#define RECV_TYPE_ARG3 int
#define RECV_TYPE_RETV int

/* int send(SOCKET, const char *, int, int); */
#define SEND_TYPE_ARG1 SOCKET
#define SEND_TYPE_ARG2 char *
#define SEND_TYPE_ARG3 int
#define SEND_TYPE_RETV int

#elif defined(__AMIGA__) /* Any AmigaOS flavour */

/* long recv(long, char *, long, long); */
#define RECV_TYPE_ARG1 long
#define RECV_TYPE_ARG2 char *
#define RECV_TYPE_ARG3 long
#define RECV_TYPE_ARG4 long
#define RECV_TYPE_RETV long

/* int send(int, const char *, int, int); */
#define SEND_TYPE_ARG1 int
#define SEND_TYPE_ARG2 char *
#define SEND_TYPE_ARG3 int
#define SEND_TYPE_RETV int
#endif


#ifndef RECV_TYPE_ARG1
#define RECV_TYPE_ARG1 int
#endif

#ifndef RECV_TYPE_ARG2
#define RECV_TYPE_ARG2 void *
#endif

#ifndef RECV_TYPE_ARG3
#define RECV_TYPE_ARG3 size_t
#endif

#ifndef RECV_TYPE_ARG4
#define RECV_TYPE_ARG4 int
#endif

#ifndef RECV_TYPE_RETV
#define RECV_TYPE_RETV ssize_t
#endif

#ifndef SEND_QUAL_ARG2
#define SEND_QUAL_ARG2 const
#endif

#ifndef SEND_TYPE_ARG1
#define SEND_TYPE_ARG1 int
#endif

#ifndef SEND_TYPE_ARG2
#define SEND_TYPE_ARG2 void *
#endif

#ifndef SEND_TYPE_ARG3
#define SEND_TYPE_ARG3 size_t
#endif

#ifndef SEND_TYPE_ARG4
#define SEND_TYPE_ARG4 int
#endif

#ifndef SEND_TYPE_RETV
#define SEND_TYPE_RETV ssize_t
#endif

#endif /* HEADER_CURL_FUNCTYPES_H */
