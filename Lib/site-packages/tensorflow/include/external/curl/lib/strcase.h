#ifndef HEADER_CURL_STRCASE_H
#define HEADER_CURL_STRCASE_H
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

#include <curl/curl.h>

/*
 * Only "raw" case insensitive strings. This is meant to be locale independent
 * and only compare strings we know are safe for this.
 *
 * The function is capable of comparing a-z case insensitively.
 *
 * Result is 1 if text matches and 0 if not.
 */

#define strcasecompare(a,b) curl_strequal(a,b)
#define strncasecompare(a,b,c) curl_strnequal(a,b,c)

char Curl_raw_toupper(char in);
char Curl_raw_tolower(char in);

/* checkprefix() is a shorter version of the above, used when the first
   argument is the string literal */
#define checkprefix(a,b)    curl_strnequal(b, STRCONST(a))

void Curl_strntoupper(char *dest, const char *src, size_t n);
void Curl_strntolower(char *dest, const char *src, size_t n);

bool Curl_safecmp(char *a, char *b);
int Curl_timestrcmp(const char *first, const char *second);

#endif /* HEADER_CURL_STRCASE_H */
