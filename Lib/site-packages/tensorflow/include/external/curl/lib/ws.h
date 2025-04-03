#ifndef HEADER_CURL_WS_H
#define HEADER_CURL_WS_H
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

#if !defined(CURL_DISABLE_WEBSOCKETS) && !defined(CURL_DISABLE_HTTP)

#ifdef USE_HYPER
#define REQTYPE void
#else
#define REQTYPE struct dynbuf
#endif

/* a client-side WS frame decoder, parsing frame headers and
 * payload, keeping track of current position and stats */
enum ws_dec_state {
  WS_DEC_INIT,
  WS_DEC_HEAD,
  WS_DEC_PAYLOAD
};

struct ws_decoder {
  int frame_age;        /* zero */
  int frame_flags;      /* See the CURLWS_* defines */
  curl_off_t payload_offset;   /* the offset parsing is at */
  curl_off_t payload_len;
  unsigned char head[10];
  int head_len, head_total;
  enum ws_dec_state state;
};

/* a client-side WS frame encoder, generating frame headers and
 * converting payloads, tracking remaining data in current frame */
struct ws_encoder {
  curl_off_t payload_len;  /* payload length of current frame */
  curl_off_t payload_remain;  /* remaining payload of current */
  unsigned int xori; /* xor index */
  unsigned char mask[4]; /* 32-bit mask for this connection */
  unsigned char firstbyte; /* first byte of frame we encode */
  bool contfragment; /* set TRUE if the previous fragment sent was not final */
};

/* A websocket connection with en- and decoder that treat frames
 * and keep track of boundaries. */
struct websocket {
  struct Curl_easy *data; /* used for write callback handling */
  struct ws_decoder dec;  /* decode of we frames */
  struct ws_encoder enc;  /* decode of we frames */
  struct bufq recvbuf;    /* raw data from the server */
  struct bufq sendbuf;    /* raw data to be sent to the server */
  struct curl_ws_frame frame;  /* the current WS FRAME received */
};

CURLcode Curl_ws_request(struct Curl_easy *data, REQTYPE *req);
CURLcode Curl_ws_accept(struct Curl_easy *data, const char *mem, size_t len);

extern const struct Curl_handler Curl_handler_ws;
#ifdef USE_SSL
extern const struct Curl_handler Curl_handler_wss;
#endif


#else
#define Curl_ws_request(x,y) CURLE_OK
#define Curl_ws_free(x) Curl_nop_stmt
#endif

#endif /* HEADER_CURL_WS_H */
