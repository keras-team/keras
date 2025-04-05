#ifndef HEADER_CURL_TIMEDIFF_H
#define HEADER_CURL_TIMEDIFF_H
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

/* Use a larger type even for 32-bit time_t systems so that we can keep
   microsecond accuracy in it */
typedef curl_off_t timediff_t;
#define FMT_TIMEDIFF_T FMT_OFF_T

#define TIMEDIFF_T_MAX CURL_OFF_T_MAX
#define TIMEDIFF_T_MIN CURL_OFF_T_MIN

/*
 * Converts number of milliseconds into a timeval structure.
 *
 * Return values:
 *    NULL IF tv is NULL or ms < 0 (eg. no timeout -> blocking select)
 *    tv with 0 in both fields IF ms == 0 (eg. 0ms timeout -> polling select)
 *    tv with converted fields IF ms > 0 (eg. >0ms timeout -> waiting select)
 */
struct timeval *curlx_mstotv(struct timeval *tv, timediff_t ms);

/*
 * Converts a timeval structure into number of milliseconds.
 */
timediff_t curlx_tvtoms(struct timeval *tv);

#endif /* HEADER_CURL_TIMEDIFF_H */
