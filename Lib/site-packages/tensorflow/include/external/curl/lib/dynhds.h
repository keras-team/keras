#ifndef HEADER_CURL_DYNHDS_H
#define HEADER_CURL_DYNHDS_H
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

#include <curl/curl.h>
#include "dynbuf.h"

struct dynbuf;

/**
 * A single header entry.
 * `name` and `value` are non-NULL and always NUL terminated.
 */
struct dynhds_entry {
  char *name;
  char *value;
  size_t namelen;
  size_t valuelen;
};

struct dynhds {
  struct dynhds_entry **hds;
  size_t hds_len;   /* number of entries in hds */
  size_t hds_allc;  /* size of hds allocation */
  size_t max_entries;   /* size limit number of entries */
  size_t strs_len; /* length of all strings */
  size_t max_strs_size; /* max length of all strings */
  int opts;
};

#define DYNHDS_OPT_NONE          (0)
#define DYNHDS_OPT_LOWERCASE     (1 << 0)

/**
 * Init for use on first time or after a reset.
 * Allow `max_entries` headers to be added, 0 for unlimited.
 * Allow size of all name and values added to not exceed `max_strs_size``
 */
void Curl_dynhds_init(struct dynhds *dynhds, size_t max_entries,
                      size_t max_strs_size);
/**
 * Frees all data held in `dynhds`, but not the struct itself.
 */
void Curl_dynhds_free(struct dynhds *dynhds);

/**
 * Reset `dyndns` to the initial init state. May keep allocations
 * around.
 */
void Curl_dynhds_reset(struct dynhds *dynhds);

/**
 * Return the number of header entries.
 */
size_t Curl_dynhds_count(struct dynhds *dynhds);

/**
 * Set the options to use, replacing any existing ones.
 * This will not have an effect on already existing headers.
 */
void Curl_dynhds_set_opts(struct dynhds *dynhds, int opts);

/**
 * Return the n-th header entry or NULL if it does not exist.
 */
struct dynhds_entry *Curl_dynhds_getn(struct dynhds *dynhds, size_t n);

/**
 * Return the 1st header entry of the name or NULL if none exists.
 */
struct dynhds_entry *Curl_dynhds_get(struct dynhds *dynhds,
                                     const char *name, size_t namelen);
struct dynhds_entry *Curl_dynhds_cget(struct dynhds *dynhds, const char *name);

#ifdef UNITTESTS
/* used by unit2602.c */

/**
 * Return TRUE iff one or more headers with the given name exist.
 */
bool Curl_dynhds_contains(struct dynhds *dynhds,
                          const char *name, size_t namelen);
bool Curl_dynhds_ccontains(struct dynhds *dynhds, const char *name);

/**
 * Return how often the given name appears in `dynhds`.
 * Names are case-insensitive.
 */
size_t Curl_dynhds_count_name(struct dynhds *dynhds,
                              const char *name, size_t namelen);

/**
 * Return how often the given 0-terminated name appears in `dynhds`.
 * Names are case-insensitive.
 */
size_t Curl_dynhds_ccount_name(struct dynhds *dynhds, const char *name);

/**
 * Remove all entries with the given name.
 * Returns number of entries removed.
 */
size_t Curl_dynhds_remove(struct dynhds *dynhds,
                          const char *name, size_t namelen);
size_t Curl_dynhds_cremove(struct dynhds *dynhds, const char *name);


/**
 * Set the give header name and value, replacing any entries with
 * the same name. The header is added at the end of all (remaining)
 * entries.
 */
CURLcode Curl_dynhds_set(struct dynhds *dynhds,
                         const char *name, size_t namelen,
                         const char *value, size_t valuelen);
#endif

CURLcode Curl_dynhds_cset(struct dynhds *dynhds,
                          const char *name, const char *value);

/**
 * Add a header, name + value, to `dynhds` at the end. Does *not*
 * check for duplicate names.
 */
CURLcode Curl_dynhds_add(struct dynhds *dynhds,
                         const char *name, size_t namelen,
                         const char *value, size_t valuelen);

/**
 * Add a header, c-string name + value, to `dynhds` at the end.
 */
CURLcode Curl_dynhds_cadd(struct dynhds *dynhds,
                          const char *name, const char *value);

/**
 * Add a single header from an HTTP/1.1 formatted line at the end. Line
 * may contain a delimiting \r\n or just \n. Any characters after
 * that will be ignored.
 */
CURLcode Curl_dynhds_h1_cadd_line(struct dynhds *dynhds, const char *line);

/**
 * Add a single header from an HTTP/1.1 formatted line at the end. Line
 * may contain a delimiting \r\n or just \n. Any characters after
 * that will be ignored.
 */
CURLcode Curl_dynhds_h1_add_line(struct dynhds *dynhds,
                                 const char *line, size_t line_len);

/**
 * Add the headers to the given `dynbuf` in HTTP/1.1 format with
 * cr+lf line endings. Will NOT output a last empty line.
 */
CURLcode Curl_dynhds_h1_dprint(struct dynhds *dynhds, struct dynbuf *dbuf);

#ifdef USE_NGHTTP2

#include <stdint.h>
#include <nghttp2/nghttp2.h>

nghttp2_nv *Curl_dynhds_to_nva(struct dynhds *dynhds, size_t *pcount);

#endif /* USE_NGHTTP2 */

#endif /* HEADER_CURL_DYNHDS_H */
