#ifndef HEADER_CURL_COOKIE_H
#define HEADER_CURL_COOKIE_H
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

#include "llist.h"

struct Cookie {
  struct Curl_llist_node node; /* for the main cookie list */
  struct Curl_llist_node getnode; /* for getlist */
  char *name;         /* <this> = value */
  char *value;        /* name = <this> */
  char *path;         /* path = <this> which is in Set-Cookie: */
  char *spath;        /* sanitized cookie path */
  char *domain;       /* domain = <this> */
  curl_off_t expires; /* expires = <this> */
  int creationtime;   /* time when the cookie was written */
  BIT(tailmatch);     /* tail-match the domain name */
  BIT(secure);        /* the 'secure' keyword was used */
  BIT(livecookie);    /* updated from a server, not a stored file */
  BIT(httponly);      /* the httponly directive is present */
  BIT(prefix_secure); /* secure prefix is set */
  BIT(prefix_host);   /* host prefix is set */
};

/*
 * Available cookie prefixes, as defined in
 * draft-ietf-httpbis-rfc6265bis-02
 */
#define COOKIE_PREFIX__SECURE (1<<0)
#define COOKIE_PREFIX__HOST (1<<1)

#define COOKIE_HASH_SIZE 63

struct CookieInfo {
  /* linked lists of cookies we know of */
  struct Curl_llist cookielist[COOKIE_HASH_SIZE];
  curl_off_t next_expiration; /* the next time at which expiration happens */
  int numcookies;  /* number of cookies in the "jar" */
  int lastct;      /* last creation-time used in the jar */
  bool running;    /* state info, for cookie adding information */
  bool newsession; /* new session, discard session cookies on load */
};

/* The maximum sizes we accept for cookies. RFC 6265 section 6.1 says
   "general-use user agents SHOULD provide each of the following minimum
   capabilities":

   - At least 4096 bytes per cookie (as measured by the sum of the length of
     the cookie's name, value, and attributes).
   In the 6265bis draft document section 5.4 it is phrased even stronger: "If
   the sum of the lengths of the name string and the value string is more than
   4096 octets, abort these steps and ignore the set-cookie-string entirely."
*/

/** Limits for INCOMING cookies **/

/* The longest we allow a line to be when reading a cookie from an HTTP header
   or from a cookie jar */
#define MAX_COOKIE_LINE 5000

/* Maximum length of an incoming cookie name or content we deal with. Longer
   cookies are ignored. */
#define MAX_NAME 4096

/* Maximum number of Set-Cookie: lines accepted in a single response. If more
   such header lines are received, they are ignored. This value must be less
   than 256 since an unsigned char is used to count. */
#define MAX_SET_COOKIE_AMOUNT 50

/** Limits for OUTGOING cookies **/

/* Maximum size for an outgoing cookie line libcurl will use in an http
   request. This is the default maximum length used in some versions of Apache
   httpd. */
#define MAX_COOKIE_HEADER_LEN 8190

/* Maximum number of cookies libcurl will send in a single request, even if
   there might be more cookies that match. One reason to cap the number is to
   keep the maximum HTTP request within the maximum allowed size. */
#define MAX_COOKIE_SEND_AMOUNT 150

struct Curl_easy;
/*
 * Add a cookie to the internal list of cookies. The domain and path arguments
 * are only used if the header boolean is TRUE.
 */

struct Cookie *Curl_cookie_add(struct Curl_easy *data,
                               struct CookieInfo *c, bool header,
                               bool noexpiry, const char *lineptr,
                               const char *domain, const char *path,
                               bool secure);

int Curl_cookie_getlist(struct Curl_easy *data,
                        struct CookieInfo *c, const char *host,
                        const char *path, bool secure,
                        struct Curl_llist *list);
void Curl_cookie_clearall(struct CookieInfo *cookies);
void Curl_cookie_clearsess(struct CookieInfo *cookies);

#if defined(CURL_DISABLE_HTTP) || defined(CURL_DISABLE_COOKIES)
#define Curl_cookie_list(x) NULL
#define Curl_cookie_loadfiles(x) Curl_nop_stmt
#define Curl_cookie_init(x,y,z,w) NULL
#define Curl_cookie_cleanup(x) Curl_nop_stmt
#define Curl_flush_cookies(x,y) Curl_nop_stmt
#else
void Curl_flush_cookies(struct Curl_easy *data, bool cleanup);
void Curl_cookie_cleanup(struct CookieInfo *c);
struct CookieInfo *Curl_cookie_init(struct Curl_easy *data,
                                    const char *file, struct CookieInfo *inc,
                                    bool newsession);
struct curl_slist *Curl_cookie_list(struct Curl_easy *data);
void Curl_cookie_loadfiles(struct Curl_easy *data);
#endif

#endif /* HEADER_CURL_COOKIE_H */
