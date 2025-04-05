#ifndef HEADER_CURL_CONNCACHE_H
#define HEADER_CURL_CONNCACHE_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Daniel Stenberg, <daniel@haxx.se>, et al.
 * Copyright (C) Linus Nielsen Feltzing, <linus@haxx.se>
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

#include <curl/curl.h>
#include "timeval.h"

struct connectdata;
struct Curl_easy;
struct curl_pollfds;
struct curl_waitfds;
struct Curl_multi;
struct Curl_share;

/**
 * Callback invoked when disconnecting connections.
 * @param data    transfer last handling the connection, not attached
 * @param conn    the connection to discard
 * @param aborted if the connection is being aborted
 * @return if the connection is being aborted, e.g. should NOT perform
 *         a shutdown and just close.
 **/
typedef bool Curl_cpool_disconnect_cb(struct Curl_easy *data,
                                      struct connectdata *conn,
                                      bool aborted);

struct cpool {
   /* the pooled connections, bundled per destination */
  struct Curl_hash dest2bundle;
  size_t num_conn;
  curl_off_t next_connection_id;
  curl_off_t next_easy_id;
  struct curltime last_cleanup;
  struct Curl_llist shutdowns;  /* The connections being shut down */
  struct Curl_easy *idata; /* internal handle used for discard */
  struct Curl_multi *multi; /* != NULL iff pool belongs to multi */
  struct Curl_share *share; /* != NULL iff pool belongs to share */
  Curl_cpool_disconnect_cb *disconnect_cb;
  BIT(locked);
};

/* Init the pool, pass multi only if pool is owned by it.
 * returns 1 on error, 0 is fine.
 */
int Curl_cpool_init(struct cpool *cpool,
                    Curl_cpool_disconnect_cb *disconnect_cb,
                    struct Curl_multi *multi,
                    struct Curl_share *share,
                    size_t size);

/* Destroy all connections and free all members */
void Curl_cpool_destroy(struct cpool *connc);

/* Init the transfer to be used within its connection pool.
 * Assigns `data->id`. */
void Curl_cpool_xfer_init(struct Curl_easy *data);

/**
 * Get the connection with the given id from the transfer's pool.
 */
struct connectdata *Curl_cpool_get_conn(struct Curl_easy *data,
                                        curl_off_t conn_id);

CURLcode Curl_cpool_add_conn(struct Curl_easy *data,
                             struct connectdata *conn) WARN_UNUSED_RESULT;

/**
 * Return if the pool has reached its configured limits for adding
 * the given connection. Will try to discard the oldest, idle
 * connections to make space.
 */
#define CPOOL_LIMIT_OK     0
#define CPOOL_LIMIT_DEST   1
#define CPOOL_LIMIT_TOTAL  2
int Curl_cpool_check_limits(struct Curl_easy *data,
                            struct connectdata *conn);

/* Return of conn is suitable. If so, stops iteration. */
typedef bool Curl_cpool_conn_match_cb(struct connectdata *conn,
                                      void *userdata);

/* Act on the result of the find, may override it. */
typedef bool Curl_cpool_done_match_cb(bool result, void *userdata);

/**
 * Find a connection in the pool matching `destination`.
 * All callbacks are invoked while the pool's lock is held.
 * @param data        current transfer
 * @param destination match agaonst `conn->destination` in pool
 * @param dest_len    destination length, including terminating NUL
 * @param conn_cb     must be present, called for each connection in the
 *                    bundle until it returns TRUE
 * @param result_cb   if not NULL, is called at the end with the result
 *                    of the `conn_cb` or FALSE if never called.
 * @return combined result of last conn_db and result_cb or FALSE if no
                      connections were present.
 */
bool Curl_cpool_find(struct Curl_easy *data,
                     const char *destination, size_t dest_len,
                     Curl_cpool_conn_match_cb *conn_cb,
                     Curl_cpool_done_match_cb *done_cb,
                     void *userdata);

/*
 * A connection (already in the pool) is now idle. Do any
 * cleanups in regard to the pool's limits.
 *
 * Return TRUE if idle connection kept in pool, FALSE if closed.
 */
bool Curl_cpool_conn_now_idle(struct Curl_easy *data,
                              struct connectdata *conn);

/**
 * Remove the connection from the pool and tear it down.
 * If `aborted` is FALSE, the connection will be shut down first
 * before closing and destroying it.
 * If the shutdown is not immediately complete, the connection
 * will be placed into the pool's shutdown queue.
 */
void Curl_cpool_disconnect(struct Curl_easy *data,
                           struct connectdata *conn,
                           bool aborted);

/**
 * This function scans the data's connection pool for half-open/dead
 * connections, closes and removes them.
 * The cleanup is done at most once per second.
 *
 * When called, this transfer has no connection attached.
 */
void Curl_cpool_prune_dead(struct Curl_easy *data);

/**
 * Perform upkeep actions on connections in the transfer's pool.
 */
CURLcode Curl_cpool_upkeep(void *data);

typedef void Curl_cpool_conn_do_cb(struct connectdata *conn,
                                   struct Curl_easy *data,
                                   void *cbdata);

/**
 * Invoke the callback on the pool's connection with the
 * given connection id (if it exists).
 */
void Curl_cpool_do_by_id(struct Curl_easy *data,
                         curl_off_t conn_id,
                         Curl_cpool_conn_do_cb *cb, void *cbdata);

/**
 * Invoked the callback for the given data + connection under the
 * connection pool's lock.
 * The callback is always invoked, even if the transfer has no connection
 * pool associated.
 */
void Curl_cpool_do_locked(struct Curl_easy *data,
                          struct connectdata *conn,
                          Curl_cpool_conn_do_cb *cb, void *cbdata);

/**
 * Add sockets and POLLIN/OUT flags for connections handled by the pool.
 */
CURLcode Curl_cpool_add_pollfds(struct cpool *connc,
                                struct curl_pollfds *cpfds);
CURLcode Curl_cpool_add_waitfds(struct cpool *connc,
                                struct curl_waitfds *cwfds);

/**
 * Perform maintenance on connections in the pool. Specifically,
 * progress the shutdown of connections in the queue.
 */
void Curl_cpool_multi_perform(struct Curl_multi *multi);

void Curl_cpool_multi_socket(struct Curl_multi *multi,
                             curl_socket_t s, int ev_bitmask);


#endif /* HEADER_CURL_CONNCACHE_H */
