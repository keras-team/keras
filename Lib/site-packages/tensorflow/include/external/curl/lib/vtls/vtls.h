#ifndef HEADER_CURL_VTLS_H
#define HEADER_CURL_VTLS_H
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

struct connectdata;
struct ssl_config_data;
struct ssl_primary_config;
struct Curl_ssl_session;

#define SSLSUPP_CA_PATH      (1<<0) /* supports CAPATH */
#define SSLSUPP_CERTINFO     (1<<1) /* supports CURLOPT_CERTINFO */
#define SSLSUPP_PINNEDPUBKEY (1<<2) /* supports CURLOPT_PINNEDPUBLICKEY */
#define SSLSUPP_SSL_CTX      (1<<3) /* supports CURLOPT_SSL_CTX */
#define SSLSUPP_HTTPS_PROXY  (1<<4) /* supports access via HTTPS proxies */
#define SSLSUPP_TLS13_CIPHERSUITES (1<<5) /* supports TLS 1.3 ciphersuites */
#define SSLSUPP_CAINFO_BLOB  (1<<6)
#define SSLSUPP_ECH          (1<<7)
#define SSLSUPP_CA_CACHE     (1<<8)
#define SSLSUPP_CIPHER_LIST  (1<<9) /* supports TLS 1.0-1.2 ciphersuites */

#define ALPN_ACCEPTED "ALPN: server accepted "

#define VTLS_INFOF_NO_ALPN            \
  "ALPN: server did not agree on a protocol. Uses default."
#define VTLS_INFOF_ALPN_OFFER_1STR    \
  "ALPN: curl offers %s"
#define VTLS_INFOF_ALPN_ACCEPTED      \
  ALPN_ACCEPTED "%.*s"

#define VTLS_INFOF_NO_ALPN_DEFERRED   \
  "ALPN: deferred handshake for early data without specific protocol."
#define VTLS_INFOF_ALPN_DEFERRED      \
  "ALPN: deferred handshake for early data using '%.*s'."

/* Curl_multi SSL backend-specific data; declared differently by each SSL
   backend */
struct Curl_cfilter;

CURLsslset Curl_init_sslset_nolock(curl_sslbackend id, const char *name,
                                   const curl_ssl_backend ***avail);

#ifndef MAX_PINNED_PUBKEY_SIZE
#define MAX_PINNED_PUBKEY_SIZE 1048576 /* 1MB */
#endif

#ifndef CURL_SHA256_DIGEST_LENGTH
#define CURL_SHA256_DIGEST_LENGTH 32 /* fixed size */
#endif

curl_sslbackend Curl_ssl_backend(void);

/**
 * Init ssl config for a new easy handle.
 */
void Curl_ssl_easy_config_init(struct Curl_easy *data);

/**
 * Init the `data->set.ssl` and `data->set.proxy_ssl` for
 * connection matching use.
 */
CURLcode Curl_ssl_easy_config_complete(struct Curl_easy *data);

/**
 * Init SSL configs (main + proxy) for a new connection from the easy handle.
 */
CURLcode Curl_ssl_conn_config_init(struct Curl_easy *data,
                                   struct connectdata *conn);

/**
 * Free allocated resources in SSL configs (main + proxy) for
 * the given connection.
 */
void Curl_ssl_conn_config_cleanup(struct connectdata *conn);

/**
 * Return TRUE iff SSL configuration from `data` is functionally the
 * same as the one on `candidate`.
 * @param proxy   match the proxy SSL config or the main one
 */
bool Curl_ssl_conn_config_match(struct Curl_easy *data,
                                struct connectdata *candidate,
                                bool proxy);

/* Update certain connection SSL config flags after they have
 * been changed on the easy handle. Will work for `verifypeer`,
 * `verifyhost` and `verifystatus`. */
void Curl_ssl_conn_config_update(struct Curl_easy *data, bool for_proxy);

/**
 * Init SSL peer information for filter. Can be called repeatedly.
 */
CURLcode Curl_ssl_peer_init(struct ssl_peer *peer,
                            struct Curl_cfilter *cf, int transport);
/**
 * Free all allocated data and reset peer information.
 */
void Curl_ssl_peer_cleanup(struct ssl_peer *peer);

#ifdef USE_SSL
int Curl_ssl_init(void);
void Curl_ssl_cleanup(void);
/* tell the SSL stuff to close down all open information regarding
   connections (and thus session ID caching etc) */
void Curl_ssl_close_all(struct Curl_easy *data);
CURLcode Curl_ssl_set_engine(struct Curl_easy *data, const char *engine);
/* Sets engine as default for all SSL operations */
CURLcode Curl_ssl_set_engine_default(struct Curl_easy *data);
struct curl_slist *Curl_ssl_engines_list(struct Curl_easy *data);

/* init the SSL session ID cache */
CURLcode Curl_ssl_initsessions(struct Curl_easy *, size_t);
void Curl_ssl_version(char *buffer, size_t size);

/* Certificate information list handling. */
#define CURL_X509_STR_MAX  100000

void Curl_ssl_free_certinfo(struct Curl_easy *data);
CURLcode Curl_ssl_init_certinfo(struct Curl_easy *data, int num);
CURLcode Curl_ssl_push_certinfo_len(struct Curl_easy *data, int certnum,
                                    const char *label, const char *value,
                                    size_t valuelen);
CURLcode Curl_ssl_push_certinfo(struct Curl_easy *data, int certnum,
                                const char *label, const char *value);

/* Functions to be used by SSL library adaptation functions */

/* Lock session cache mutex.
 * Call this before calling other Curl_ssl_*session* functions
 * Caller should unlock this mutex as soon as possible, as it may block
 * other SSL connection from making progress.
 * The purpose of explicitly locking SSL session cache data is to allow
 * individual SSL engines to manage session lifetime in their specific way.
 */
void Curl_ssl_sessionid_lock(struct Curl_easy *data);

/* Unlock session cache mutex */
void Curl_ssl_sessionid_unlock(struct Curl_easy *data);

/* Kill a single session ID entry in the cache
 * Sessionid mutex must be locked (see Curl_ssl_sessionid_lock).
 * This will call engine-specific curlssl_session_free function, which must
 * take sessionid object ownership from sessionid cache
 * (e.g. decrement refcount).
 */
void Curl_ssl_kill_session(struct Curl_ssl_session *session);
/* delete a session from the cache
 * Sessionid mutex must be locked (see Curl_ssl_sessionid_lock).
 * This will call engine-specific curlssl_session_free function, which must
 * take sessionid object ownership from sessionid cache
 * (e.g. decrement refcount).
 */
void Curl_ssl_delsessionid(struct Curl_easy *data, void *ssl_sessionid);

/* get N random bytes into the buffer */
CURLcode Curl_ssl_random(struct Curl_easy *data, unsigned char *buffer,
                         size_t length);
/* Check pinned public key. */
CURLcode Curl_pin_peer_pubkey(struct Curl_easy *data,
                              const char *pinnedpubkey,
                              const unsigned char *pubkey, size_t pubkeylen);

bool Curl_ssl_cert_status_request(void);

bool Curl_ssl_false_start(struct Curl_easy *data);

/* The maximum size of the SSL channel binding is 85 bytes, as defined in
 * RFC 5929, Section 4.1. The 'tls-server-end-point:' prefix is 21 bytes long,
 * and SHA-512 is the longest supported hash algorithm, with a digest length of
 * 64 bytes.
 * The maximum size of the channel binding is therefore 21 + 64 = 85 bytes.
 */
#define SSL_CB_MAX_SIZE 85

/* Return the tls-server-end-point channel binding, including the
 * 'tls-server-end-point:' prefix.
 * If successful, the data is written to the dynbuf, and CURLE_OK is returned.
 * The dynbuf MUST HAVE a minimum toobig size of SSL_CB_MAX_SIZE.
 * If the dynbuf is too small, CURLE_OUT_OF_MEMORY is returned.
 * If channel binding is not supported, binding stays empty and CURLE_OK is
 * returned.
 */
CURLcode Curl_ssl_get_channel_binding(struct Curl_easy *data, int sockindex,
                                       struct dynbuf *binding);

#define SSL_SHUTDOWN_TIMEOUT 10000 /* ms */

CURLcode Curl_ssl_cfilter_add(struct Curl_easy *data,
                              struct connectdata *conn,
                              int sockindex);

CURLcode Curl_cf_ssl_insert_after(struct Curl_cfilter *cf_at,
                                  struct Curl_easy *data);

CURLcode Curl_ssl_cfilter_remove(struct Curl_easy *data,
                                 int sockindex, bool send_shutdown);

#ifndef CURL_DISABLE_PROXY
CURLcode Curl_cf_ssl_proxy_insert_after(struct Curl_cfilter *cf_at,
                                        struct Curl_easy *data);
#endif /* !CURL_DISABLE_PROXY */

/**
 * True iff the underlying SSL implementation supports the option.
 * Option is one of the defined SSLSUPP_* values.
 * `data` maybe NULL for the features of the default implementation.
 */
bool Curl_ssl_supports(struct Curl_easy *data, unsigned int ssl_option);

/**
 * Get the internal ssl instance (like OpenSSL's SSL*) from the filter
 * chain at `sockindex` of type specified by `info`.
 * For `n` == 0, the first active (top down) instance is returned.
 * 1 gives the second active, etc.
 * NULL is returned when no active SSL filter is present.
 */
void *Curl_ssl_get_internals(struct Curl_easy *data, int sockindex,
                             CURLINFO info, int n);

/**
 * Get the ssl_config_data in `data` that is relevant for cfilter `cf`.
 */
struct ssl_config_data *Curl_ssl_cf_get_config(struct Curl_cfilter *cf,
                                               struct Curl_easy *data);

/**
 * Get the primary config relevant for the filter from its connection.
 */
struct ssl_primary_config *
  Curl_ssl_cf_get_primary_config(struct Curl_cfilter *cf);

extern struct Curl_cftype Curl_cft_ssl;
#ifndef CURL_DISABLE_PROXY
extern struct Curl_cftype Curl_cft_ssl_proxy;
#endif

#else /* if not USE_SSL */

/* When SSL support is not present, just define away these function calls */
#define Curl_ssl_init() 1
#define Curl_ssl_cleanup() Curl_nop_stmt
#define Curl_ssl_close_all(x) Curl_nop_stmt
#define Curl_ssl_set_engine(x,y) CURLE_NOT_BUILT_IN
#define Curl_ssl_set_engine_default(x) CURLE_NOT_BUILT_IN
#define Curl_ssl_engines_list(x) NULL
#define Curl_ssl_initsessions(x,y) CURLE_OK
#define Curl_ssl_free_certinfo(x) Curl_nop_stmt
#define Curl_ssl_kill_session(x) Curl_nop_stmt
#define Curl_ssl_random(x,y,z) ((void)x, CURLE_NOT_BUILT_IN)
#define Curl_ssl_cert_status_request() FALSE
#define Curl_ssl_false_start(a) FALSE
#define Curl_ssl_get_internals(a,b,c,d) NULL
#define Curl_ssl_supports(a,b) FALSE
#define Curl_ssl_cfilter_add(a,b,c) CURLE_NOT_BUILT_IN
#define Curl_ssl_cfilter_remove(a,b,c) CURLE_OK
#define Curl_ssl_cf_get_config(a,b) NULL
#define Curl_ssl_cf_get_primary_config(a) NULL
#endif

#endif /* HEADER_CURL_VTLS_H */
