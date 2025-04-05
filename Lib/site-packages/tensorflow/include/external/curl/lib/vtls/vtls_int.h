#ifndef HEADER_CURL_VTLS_INT_H
#define HEADER_CURL_VTLS_INT_H
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
#include "cfilters.h"
#include "urldata.h"

#ifdef USE_SSL

struct ssl_connect_data;

/* see https://www.iana.org/assignments/tls-extensiontype-values/ */
#define ALPN_HTTP_1_1_LENGTH 8
#define ALPN_HTTP_1_1 "http/1.1"
#define ALPN_H2_LENGTH 2
#define ALPN_H2 "h2"
#define ALPN_H3_LENGTH 2
#define ALPN_H3 "h3"

/* conservative sizes on the ALPN entries and count we are handling,
 * we can increase these if we ever feel the need or have to accommodate
 * ALPN strings from the "outside". */
#define ALPN_NAME_MAX     10
#define ALPN_ENTRIES_MAX  3
#define ALPN_PROTO_BUF_MAX   (ALPN_ENTRIES_MAX * (ALPN_NAME_MAX + 1))

struct alpn_spec {
  const char entries[ALPN_ENTRIES_MAX][ALPN_NAME_MAX];
  size_t count; /* number of entries */
};

struct alpn_proto_buf {
  unsigned char data[ALPN_PROTO_BUF_MAX];
  int len;
};

CURLcode Curl_alpn_to_proto_buf(struct alpn_proto_buf *buf,
                                const struct alpn_spec *spec);
CURLcode Curl_alpn_to_proto_str(struct alpn_proto_buf *buf,
                                const struct alpn_spec *spec);

CURLcode Curl_alpn_set_negotiated(struct Curl_cfilter *cf,
                                  struct Curl_easy *data,
                                  struct ssl_connect_data *connssl,
                                  const unsigned char *proto,
                                  size_t proto_len);

bool Curl_alpn_contains_proto(const struct alpn_spec *spec,
                              const char *proto);

/* enum for the nonblocking SSL connection state machine */
typedef enum {
  ssl_connect_1,
  ssl_connect_2,
  ssl_connect_3,
  ssl_connect_done
} ssl_connect_state;

typedef enum {
  ssl_connection_none,
  ssl_connection_deferred,
  ssl_connection_negotiating,
  ssl_connection_complete
} ssl_connection_state;

typedef enum {
  ssl_earlydata_none,
  ssl_earlydata_use,
  ssl_earlydata_sending,
  ssl_earlydata_sent,
  ssl_earlydata_accepted,
  ssl_earlydata_rejected
} ssl_earlydata_state;

#define CURL_SSL_IO_NEED_NONE   (0)
#define CURL_SSL_IO_NEED_RECV   (1<<0)
#define CURL_SSL_IO_NEED_SEND   (1<<1)

/* Max earlydata payload we want to send */
#define CURL_SSL_EARLY_MAX       (64*1024)

/* Information in each SSL cfilter context: cf->ctx */
struct ssl_connect_data {
  struct ssl_peer peer;
  const struct alpn_spec *alpn;     /* ALPN to use or NULL for none */
  void *backend;                    /* vtls backend specific props */
  struct cf_call_data call_data;    /* data handle used in current call */
  struct curltime handshake_done;   /* time when handshake finished */
  char *alpn_negotiated;            /* negotiated ALPN value or NULL */
  struct bufq earlydata;            /* earlydata to be send to peer */
  size_t earlydata_max;             /* max earlydata allowed by peer */
  size_t earlydata_skip;            /* sending bytes to skip when earlydata
                                     * is accepted by peer */
  ssl_connection_state state;
  ssl_connect_state connecting_state;
  ssl_earlydata_state earlydata_state;
  int io_need;                      /* TLS signals special SEND/RECV needs */
  BIT(use_alpn);                    /* if ALPN shall be used in handshake */
  BIT(peer_closed);                 /* peer has closed connection */
};


#undef CF_CTX_CALL_DATA
#define CF_CTX_CALL_DATA(cf)  \
  ((struct ssl_connect_data *)(cf)->ctx)->call_data


/* Definitions for SSL Implementations */

struct Curl_ssl {
  /*
   * This *must* be the first entry to allow returning the list of available
   * backends in curl_global_sslset().
   */
  curl_ssl_backend info;
  unsigned int supports; /* bitfield, see above */
  size_t sizeof_ssl_backend_data;

  int (*init)(void);
  void (*cleanup)(void);

  size_t (*version)(char *buffer, size_t size);
  int (*check_cxn)(struct Curl_cfilter *cf, struct Curl_easy *data);
  CURLcode (*shut_down)(struct Curl_cfilter *cf, struct Curl_easy *data,
                        bool send_shutdown, bool *done);
  bool (*data_pending)(struct Curl_cfilter *cf,
                       const struct Curl_easy *data);

  /* return 0 if a find random is filled in */
  CURLcode (*random)(struct Curl_easy *data, unsigned char *entropy,
                     size_t length);
  bool (*cert_status_request)(void);

  CURLcode (*connect_blocking)(struct Curl_cfilter *cf,
                               struct Curl_easy *data);
  CURLcode (*connect_nonblocking)(struct Curl_cfilter *cf,
                                  struct Curl_easy *data,
                                  bool *done);

  /* During handshake/shutdown, adjust the pollset to include the socket
   * for POLLOUT or POLLIN as needed. Mandatory. */
  void (*adjust_pollset)(struct Curl_cfilter *cf, struct Curl_easy *data,
                          struct easy_pollset *ps);
  void *(*get_internals)(struct ssl_connect_data *connssl, CURLINFO info);
  void (*close)(struct Curl_cfilter *cf, struct Curl_easy *data);
  void (*close_all)(struct Curl_easy *data);

  CURLcode (*set_engine)(struct Curl_easy *data, const char *engine);
  CURLcode (*set_engine_default)(struct Curl_easy *data);
  struct curl_slist *(*engines_list)(struct Curl_easy *data);

  bool (*false_start)(void);
  CURLcode (*sha256sum)(const unsigned char *input, size_t inputlen,
                    unsigned char *sha256sum, size_t sha256sumlen);

  bool (*attach_data)(struct Curl_cfilter *cf, struct Curl_easy *data);
  void (*detach_data)(struct Curl_cfilter *cf, struct Curl_easy *data);

  ssize_t (*recv_plain)(struct Curl_cfilter *cf, struct Curl_easy *data,
                        char *buf, size_t len, CURLcode *code);
  ssize_t (*send_plain)(struct Curl_cfilter *cf, struct Curl_easy *data,
                        const void *mem, size_t len, CURLcode *code);

  CURLcode (*get_channel_binding)(struct Curl_easy *data, int sockindex,
                                  struct dynbuf *binding);

};

extern const struct Curl_ssl *Curl_ssl;


int Curl_none_init(void);
void Curl_none_cleanup(void);
CURLcode Curl_none_shutdown(struct Curl_cfilter *cf, struct Curl_easy *data,
                            bool send_shutdown, bool *done);
int Curl_none_check_cxn(struct Curl_cfilter *cf, struct Curl_easy *data);
void Curl_none_close_all(struct Curl_easy *data);
void Curl_none_session_free(void *ptr);
bool Curl_none_data_pending(struct Curl_cfilter *cf,
                            const struct Curl_easy *data);
bool Curl_none_cert_status_request(void);
CURLcode Curl_none_set_engine(struct Curl_easy *data, const char *engine);
CURLcode Curl_none_set_engine_default(struct Curl_easy *data);
struct curl_slist *Curl_none_engines_list(struct Curl_easy *data);
bool Curl_none_false_start(void);
void Curl_ssl_adjust_pollset(struct Curl_cfilter *cf, struct Curl_easy *data,
                              struct easy_pollset *ps);

/**
 * Get the SSL filter below the given one or NULL if there is none.
 */
bool Curl_ssl_cf_is_proxy(struct Curl_cfilter *cf);

/* extract a session ID
 * Sessionid mutex must be locked (see Curl_ssl_sessionid_lock).
 * Caller must make sure that the ownership of returned sessionid object
 * is properly taken (e.g. its refcount is incremented
 * under sessionid mutex).
 * @param cf      the connection filter wanting to use it
 * @param data    the transfer involved
 * @param peer    the peer the filter wants to talk to
 * @param sessionid on return the TLS session
 * @param idsize  on return the size of the TLS session data
 * @param palpn   on return the ALPN string used by the session,
 *                set to NULL when not interested
 */
bool Curl_ssl_getsessionid(struct Curl_cfilter *cf,
                           struct Curl_easy *data,
                           const struct ssl_peer *peer,
                           void **ssl_sessionid,
                           size_t *idsize, /* set 0 if unknown */
                           char **palpn);

/* Set a TLS session ID for `peer`. Replaces an existing session ID if
 * not already the same.
 * Sessionid mutex must be locked (see Curl_ssl_sessionid_lock).
 * Call takes ownership of `ssl_sessionid`, using `sessionid_free_cb`
 * to deallocate it. Is called in all outcomes, either right away or
 * later when the session cache is cleaned up.
 * Caller must ensure that it has properly shared ownership of this sessionid
 * object with cache (e.g. incrementing refcount on success)
 */
CURLcode Curl_ssl_set_sessionid(struct Curl_cfilter *cf,
                                struct Curl_easy *data,
                                const struct ssl_peer *peer,
                                const char *alpn,
                                void *sessionid,
                                size_t sessionid_size,
                                Curl_ssl_sessionid_dtor *sessionid_free_cb);

#endif /* USE_SSL */

#endif /* HEADER_CURL_VTLS_INT_H */
