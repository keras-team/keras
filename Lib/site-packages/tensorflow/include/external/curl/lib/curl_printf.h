#ifndef HEADER_CURL_PRINTF_H
#define HEADER_CURL_PRINTF_H
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

/*
 * This header should be included by ALL code in libcurl that uses any
 * *rintf() functions.
 */

#ifndef CURL_TEMP_PRINTF
#error "CURL_TEMP_PRINTF must be set before including curl/mprintf.h"
#endif

#include <curl/mprintf.h>

#define MERR_OK        0
#define MERR_MEM       1
#define MERR_TOO_LARGE 2

# undef printf
# undef fprintf
# undef msnprintf
# undef vprintf
# undef vfprintf
# undef mvsnprintf
# undef aprintf
# undef vaprintf
# define printf curl_mprintf
# define fprintf curl_mfprintf
# define msnprintf curl_msnprintf
# define vprintf curl_mvprintf
# define vfprintf curl_mvfprintf
# define mvsnprintf curl_mvsnprintf
# define aprintf curl_maprintf
# define vaprintf curl_mvaprintf
#endif /* HEADER_CURL_PRINTF_H */
