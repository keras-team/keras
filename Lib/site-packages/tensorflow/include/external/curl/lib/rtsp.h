#ifndef HEADER_CURL_RTSP_H
#define HEADER_CURL_RTSP_H
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
#ifdef USE_HYPER
#define CURL_DISABLE_RTSP 1
#endif

#ifndef CURL_DISABLE_RTSP

extern const struct Curl_handler Curl_handler_rtsp;

CURLcode Curl_rtsp_parseheader(struct Curl_easy *data, const char *header);

#else
/* disabled */
#define Curl_rtsp_parseheader(x,y) CURLE_NOT_BUILT_IN

#endif /* CURL_DISABLE_RTSP */

typedef enum {
  RTP_PARSE_SKIP,
  RTP_PARSE_CHANNEL,
  RTP_PARSE_LEN,
  RTP_PARSE_DATA
} rtp_parse_st;
/*
 * RTSP Connection data
 *
 * Currently, only used for tracking incomplete RTP data reads
 */
struct rtsp_conn {
  struct dynbuf buf;
  int rtp_channel;
  size_t rtp_len;
  rtp_parse_st state;
  BIT(in_header);
};

/****************************************************************************
 * RTSP unique setup
 ***************************************************************************/
struct RTSP {
  long CSeq_sent; /* CSeq of this request */
  long CSeq_recv; /* CSeq received */
};


#endif /* HEADER_CURL_RTSP_H */
