#ifndef HEADER_CURL_HTTP1_H
#define HEADER_CURL_HTTP1_H
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

#ifndef CURL_DISABLE_HTTP
#include "bufq.h"
#include "http.h"

#define H1_PARSE_OPT_NONE       (0)
#define H1_PARSE_OPT_STRICT     (1 << 0)

#define H1_PARSE_DEFAULT_MAX_LINE_LEN   DYN_HTTP_REQUEST

struct h1_req_parser {
  struct httpreq *req;
  struct dynbuf scratch;
  size_t scratch_skip;
  const char *line;
  size_t max_line_len;
  size_t line_len;
  bool done;
};

void Curl_h1_req_parse_init(struct h1_req_parser *parser, size_t max_line_len);
void Curl_h1_req_parse_free(struct h1_req_parser *parser);

ssize_t Curl_h1_req_parse_read(struct h1_req_parser *parser,
                               const char *buf, size_t buflen,
                               const char *scheme_default, int options,
                               CURLcode *err);

CURLcode Curl_h1_req_dprint(const struct httpreq *req,
                            struct dynbuf *dbuf);

CURLcode Curl_h1_req_write_head(struct httpreq *req, int http_minor,
                                struct dynbuf *dbuf);

#endif /* !CURL_DISABLE_HTTP */
#endif /* HEADER_CURL_HTTP1_H */
