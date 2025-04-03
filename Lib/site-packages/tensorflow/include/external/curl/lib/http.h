#ifndef HEADER_CURL_HTTP_H
#define HEADER_CURL_HTTP_H
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

#if defined(USE_MSH3) && !defined(_WIN32)
#include <pthread.h>
#endif

#include "bufq.h"
#include "dynhds.h"
#include "ws.h"

typedef enum {
  HTTPREQ_GET,
  HTTPREQ_POST,
  HTTPREQ_POST_FORM, /* we make a difference internally */
  HTTPREQ_POST_MIME, /* we make a difference internally */
  HTTPREQ_PUT,
  HTTPREQ_HEAD
} Curl_HttpReq;

#ifndef CURL_DISABLE_HTTP

#if defined(USE_HTTP3)
#include <stdint.h>
#endif

extern const struct Curl_handler Curl_handler_http;

#ifdef USE_SSL
extern const struct Curl_handler Curl_handler_https;
#endif

struct dynhds;

CURLcode Curl_bump_headersize(struct Curl_easy *data,
                              size_t delta,
                              bool connect_only);

/* Header specific functions */
bool Curl_compareheader(const char *headerline,  /* line to check */
                        const char *header,   /* header keyword _with_ colon */
                        const size_t hlen,   /* len of the keyword in bytes */
                        const char *content, /* content string to find */
                        const size_t clen);   /* len of the content in bytes */

char *Curl_copy_header_value(const char *header);

char *Curl_checkProxyheaders(struct Curl_easy *data,
                             const struct connectdata *conn,
                             const char *thisheader,
                             const size_t thislen);

CURLcode Curl_add_timecondition(struct Curl_easy *data,
#ifndef USE_HYPER
                                struct dynbuf *req
#else
                                void *headers
#endif
  );
CURLcode Curl_add_custom_headers(struct Curl_easy *data,
                                 bool is_connect,
#ifndef USE_HYPER
                                 struct dynbuf *req
#else
                                 void *headers
#endif
  );
CURLcode Curl_dynhds_add_custom(struct Curl_easy *data,
                                bool is_connect,
                                struct dynhds *hds);

void Curl_http_method(struct Curl_easy *data, struct connectdata *conn,
                      const char **method, Curl_HttpReq *);
CURLcode Curl_http_useragent(struct Curl_easy *data);
CURLcode Curl_http_host(struct Curl_easy *data, struct connectdata *conn);
CURLcode Curl_http_target(struct Curl_easy *data, struct connectdata *conn,
                          struct dynbuf *req);
CURLcode Curl_http_statusline(struct Curl_easy *data,
                              struct connectdata *conn);
CURLcode Curl_http_header(struct Curl_easy *data,
                          const char *hd, size_t hdlen);
CURLcode Curl_transferencode(struct Curl_easy *data);
CURLcode Curl_http_req_set_reader(struct Curl_easy *data,
                                  Curl_HttpReq httpreq,
                                  const char **tep);
CURLcode Curl_http_req_complete(struct Curl_easy *data,
                                struct dynbuf *r, Curl_HttpReq httpreq);
bool Curl_use_http_1_1plus(const struct Curl_easy *data,
                           const struct connectdata *conn);
#ifndef CURL_DISABLE_COOKIES
CURLcode Curl_http_cookies(struct Curl_easy *data,
                           struct connectdata *conn,
                           struct dynbuf *r);
#else
#define Curl_http_cookies(a,b,c) CURLE_OK
#endif
CURLcode Curl_http_range(struct Curl_easy *data,
                         Curl_HttpReq httpreq);
CURLcode Curl_http_firstwrite(struct Curl_easy *data);

/* protocol-specific functions set up to be called by the main engine */
CURLcode Curl_http_setup_conn(struct Curl_easy *data,
                              struct connectdata *conn);
CURLcode Curl_http(struct Curl_easy *data, bool *done);
CURLcode Curl_http_done(struct Curl_easy *data, CURLcode, bool premature);
CURLcode Curl_http_connect(struct Curl_easy *data, bool *done);
int Curl_http_getsock_do(struct Curl_easy *data, struct connectdata *conn,
                         curl_socket_t *socks);
CURLcode Curl_http_write_resp(struct Curl_easy *data,
                              const char *buf, size_t blen,
                              bool is_eos);
CURLcode Curl_http_write_resp_hd(struct Curl_easy *data,
                                 const char *hd, size_t hdlen,
                                 bool is_eos);

/* These functions are in http.c */
CURLcode Curl_http_input_auth(struct Curl_easy *data, bool proxy,
                              const char *auth);
CURLcode Curl_http_auth_act(struct Curl_easy *data);

/* If only the PICKNONE bit is set, there has been a round-trip and we
   selected to use no auth at all. Ie, we actively select no auth, as opposed
   to not having one selected. The other CURLAUTH_* defines are present in the
   public curl/curl.h header. */
#define CURLAUTH_PICKNONE (1<<30) /* do not use auth */

/* MAX_INITIAL_POST_SIZE indicates the number of bytes that will make the POST
   data get included in the initial data chunk sent to the server. If the
   data is larger than this, it will automatically get split up in multiple
   system calls.

   This value used to be fairly big (100K), but we must take into account that
   if the server rejects the POST due for authentication reasons, this data
   will always be unconditionally sent and thus it may not be larger than can
   always be afforded to send twice.

   It must not be greater than 64K to work on VMS.
*/
#ifndef MAX_INITIAL_POST_SIZE
#define MAX_INITIAL_POST_SIZE (64*1024)
#endif

/* EXPECT_100_THRESHOLD is the request body size limit for when libcurl will
 * automatically add an "Expect: 100-continue" header in HTTP requests. When
 * the size is unknown, it will always add it.
 *
 */
#ifndef EXPECT_100_THRESHOLD
#define EXPECT_100_THRESHOLD (1024*1024)
#endif

/* MAX_HTTP_RESP_HEADER_SIZE is the maximum size of all response headers
   combined that libcurl allows for a single HTTP response, any HTTP
   version. This count includes CONNECT response headers. */
#define MAX_HTTP_RESP_HEADER_SIZE (300*1024)

bool Curl_http_exp100_is_selected(struct Curl_easy *data);
void Curl_http_exp100_got100(struct Curl_easy *data);

#endif /* CURL_DISABLE_HTTP */

/****************************************************************************
 * HTTP unique setup
 ***************************************************************************/

CURLcode Curl_http_size(struct Curl_easy *data);

CURLcode Curl_http_write_resp_hds(struct Curl_easy *data,
                                  const char *buf, size_t blen,
                                  size_t *pconsumed);

/**
 * Curl_http_output_auth() setups the authentication headers for the
 * host/proxy and the correct authentication
 * method. data->state.authdone is set to TRUE when authentication is
 * done.
 *
 * @param data all information about the current transfer
 * @param conn all information about the current connection
 * @param request pointer to the request keyword
 * @param httpreq is the request type
 * @param path pointer to the requested path
 * @param proxytunnel boolean if this is the request setting up a "proxy
 * tunnel"
 *
 * @returns CURLcode
 */
CURLcode
Curl_http_output_auth(struct Curl_easy *data,
                      struct connectdata *conn,
                      const char *request,
                      Curl_HttpReq httpreq,
                      const char *path,
                      bool proxytunnel); /* TRUE if this is the request setting
                                            up the proxy tunnel */

/* Decode HTTP status code string. */
CURLcode Curl_http_decode_status(int *pstatus, const char *s, size_t len);


/**
 * All about a core HTTP request, excluding body and trailers
 */
struct httpreq {
  char method[24];
  char *scheme;
  char *authority;
  char *path;
  struct dynhds headers;
  struct dynhds trailers;
};

/**
 * Create an HTTP request struct.
 */
CURLcode Curl_http_req_make(struct httpreq **preq,
                            const char *method, size_t m_len,
                            const char *scheme, size_t s_len,
                            const char *authority, size_t a_len,
                            const char *path, size_t p_len);

CURLcode Curl_http_req_make2(struct httpreq **preq,
                             const char *method, size_t m_len,
                             CURLU *url, const char *scheme_default);

void Curl_http_req_free(struct httpreq *req);

#define HTTP_PSEUDO_METHOD ":method"
#define HTTP_PSEUDO_SCHEME ":scheme"
#define HTTP_PSEUDO_AUTHORITY ":authority"
#define HTTP_PSEUDO_PATH ":path"
#define HTTP_PSEUDO_STATUS ":status"

/**
 * Create the list of HTTP/2 headers which represent the request,
 * using HTTP/2 pseudo headers preceding the `req->headers`.
 *
 * Applies the following transformations:
 * - if `authority` is set, any "Host" header is removed.
 * - if `authority` is unset and a "Host" header is present, use
 *   that as `authority` and remove "Host"
 * - removes and Connection header fields as defined in rfc9113 ch. 8.2.2
 * - lower-cases the header field names
 *
 * @param h2_headers will contain the HTTP/2 headers on success
 * @param req        the request to transform
 * @param data       the handle to lookup defaults like ' :scheme' from
 */
CURLcode Curl_http_req_to_h2(struct dynhds *h2_headers,
                             struct httpreq *req, struct Curl_easy *data);

/**
 * All about a core HTTP response, excluding body and trailers
 */
struct http_resp {
  int status;
  char *description;
  struct dynhds headers;
  struct dynhds trailers;
  struct http_resp *prev;
};

/**
 * Create an HTTP response struct.
 */
CURLcode Curl_http_resp_make(struct http_resp **presp,
                             int status,
                             const char *description);

void Curl_http_resp_free(struct http_resp *resp);

#endif /* HEADER_CURL_HTTP_H */
