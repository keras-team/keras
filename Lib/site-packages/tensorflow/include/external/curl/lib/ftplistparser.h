#ifndef HEADER_CURL_FTPLISTPARSER_H
#define HEADER_CURL_FTPLISTPARSER_H
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

#ifndef CURL_DISABLE_FTP

/* WRITEFUNCTION callback for parsing LIST responses */
size_t Curl_ftp_parselist(char *buffer, size_t size, size_t nmemb,
                          void *connptr);

struct ftp_parselist_data; /* defined inside ftplibparser.c */

CURLcode Curl_ftp_parselist_geterror(struct ftp_parselist_data *pl_data);

struct ftp_parselist_data *Curl_ftp_parselist_data_alloc(void);

void Curl_ftp_parselist_data_free(struct ftp_parselist_data **pl_data);

/* list of wildcard process states */
typedef enum {
  CURLWC_CLEAR = 0,
  CURLWC_INIT = 1,
  CURLWC_MATCHING, /* library is trying to get list of addresses for
                      downloading */
  CURLWC_DOWNLOADING,
  CURLWC_CLEAN, /* deallocate resources and reset settings */
  CURLWC_SKIP,  /* skip over concrete file */
  CURLWC_ERROR, /* error cases */
  CURLWC_DONE   /* if is wildcard->state == CURLWC_DONE wildcard loop
                   will end */
} wildcard_states;

typedef void (*wildcard_dtor)(void *ptr);

/* struct keeping information about wildcard download process */
struct WildcardData {
  char *path; /* path to the directory, where we trying wildcard-match */
  char *pattern; /* wildcard pattern */
  struct Curl_llist filelist; /* llist with struct Curl_fileinfo */
  struct ftp_wc *ftpwc; /* pointer to FTP wildcard data */
  wildcard_dtor dtor;
  unsigned char state; /* wildcard_states */
};

CURLcode Curl_wildcard_init(struct WildcardData *wc);
void Curl_wildcard_dtor(struct WildcardData **wcp);

struct Curl_easy;

#else
/* FTP is disabled */
#define Curl_wildcard_dtor(x)
#endif /* CURL_DISABLE_FTP */
#endif /* HEADER_CURL_FTPLISTPARSER_H */
