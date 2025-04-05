#ifndef HEADER_CURL_VQUIC_QUIC_INT_H
#define HEADER_CURL_VQUIC_QUIC_INT_H
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
#include "bufq.h"

#ifdef USE_HTTP3

#define MAX_PKT_BURST 10
#define MAX_UDP_PAYLOAD_SIZE  1452
/* Default QUIC connection timeout we announce from our side */
#define CURL_QUIC_MAX_IDLE_MS   (120 * 1000)

struct cf_quic_ctx {
  curl_socket_t sockfd; /* connected UDP socket */
  struct sockaddr_storage local_addr; /* address socket is bound to */
  socklen_t local_addrlen; /* length of local address */

  struct bufq sendbuf; /* buffer for sending one or more packets */
  struct curltime first_byte_at;     /* when first byte was recvd */
  struct curltime last_op; /* last (attempted) send/recv operation */
  struct curltime last_io; /* last successful socket IO */
  size_t gsolen; /* length of individual packets in send buf */
  size_t split_len; /* if != 0, buffer length after which GSO differs */
  size_t split_gsolen; /* length of individual packets after split_len */
#ifdef DEBUGBUILD
  int wblock_percent; /* percent of writes doing EAGAIN */
#endif
  BIT(got_first_byte); /* if first byte was received */
  BIT(no_gso); /* do not use gso on sending */
};

CURLcode vquic_ctx_init(struct cf_quic_ctx *qctx);
void vquic_ctx_free(struct cf_quic_ctx *qctx);

void vquic_ctx_update_time(struct cf_quic_ctx *qctx);

void vquic_push_blocked_pkt(struct Curl_cfilter *cf,
                            struct cf_quic_ctx *qctx,
                            const uint8_t *pkt, size_t pktlen, size_t gsolen);

CURLcode vquic_send_blocked_pkts(struct Curl_cfilter *cf,
                                 struct Curl_easy *data,
                                 struct cf_quic_ctx *qctx);

CURLcode vquic_send(struct Curl_cfilter *cf, struct Curl_easy *data,
                        struct cf_quic_ctx *qctx, size_t gsolen);

CURLcode vquic_send_tail_split(struct Curl_cfilter *cf, struct Curl_easy *data,
                               struct cf_quic_ctx *qctx, size_t gsolen,
                               size_t tail_len, size_t tail_gsolen);

CURLcode vquic_flush(struct Curl_cfilter *cf, struct Curl_easy *data,
                     struct cf_quic_ctx *qctx);


typedef CURLcode vquic_recv_pkt_cb(const unsigned char *pkt, size_t pktlen,
                                   struct sockaddr_storage *remote_addr,
                                   socklen_t remote_addrlen, int ecn,
                                   void *userp);

CURLcode vquic_recv_packets(struct Curl_cfilter *cf,
                            struct Curl_easy *data,
                            struct cf_quic_ctx *qctx,
                            size_t max_pkts,
                            vquic_recv_pkt_cb *recv_cb, void *userp);

#endif /* !USE_HTTP3 */

#endif /* HEADER_CURL_VQUIC_QUIC_INT_H */
