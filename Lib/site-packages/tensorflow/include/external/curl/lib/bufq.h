#ifndef HEADER_CURL_BUFQ_H
#define HEADER_CURL_BUFQ_H
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

/**
 * A chunk of bytes for reading and writing.
 * The size is fixed a creation with read and write offset
 * for where unread content is.
 */
struct buf_chunk {
  struct buf_chunk *next;  /* to keep it in a list */
  size_t dlen;             /* the amount of allocated x.data[] */
  size_t r_offset;         /* first unread bytes */
  size_t w_offset;         /* one after last written byte */
  union {
    unsigned char data[1]; /* the buffer for `dlen` bytes */
    void *dummy;           /* alignment */
  } x;
};

/**
 * A pool for providing/keeping a number of chunks of the same size
 *
 * The same pool can be shared by many `bufq` instances. However, a pool
 * is not thread safe. All bufqs using it are supposed to operate in the
 * same thread.
 */
struct bufc_pool {
  struct buf_chunk *spare;  /* list of available spare chunks */
  size_t chunk_size;        /* the size of chunks in this pool */
  size_t spare_count;       /* current number of spare chunks in list */
  size_t spare_max;         /* max number of spares to keep */
};

void Curl_bufcp_init(struct bufc_pool *pool,
                     size_t chunk_size, size_t spare_max);

void Curl_bufcp_free(struct bufc_pool *pool);

/**
 * A queue of byte chunks for reading and writing.
 * Reading is done from `head`, writing is done to `tail`.
 *
 * `bufq`s can be empty or full or neither. Its `len` is the number
 * of bytes that can be read. For an empty bufq, `len` will be 0.
 *
 * By default, a bufq can hold up to `max_chunks * chunk_size` number
 * of bytes. When `max_chunks` are used (in the `head` list) and the
 * `tail` chunk is full, the bufq will report that it is full.
 *
 * On a full bufq, `len` may be less than the maximum number of bytes,
 * e.g. when the head chunk is partially read. `len` may also become
 * larger than the max when option `BUFQ_OPT_SOFT_LIMIT` is used.
 *
 * By default, writing to a full bufq will return (-1, CURLE_AGAIN). Same
 * as reading from an empty bufq.
 * With `BUFQ_OPT_SOFT_LIMIT` set, a bufq will allow writing becond this
 * limit and use more than `max_chunks`. However it will report that it
 * is full nevertheless. This is provided for situation where writes
 * preferably never fail (except for memory exhaustion).
 *
 * By default and without a pool, a bufq will keep chunks that read
 * empty in its `spare` list. Option `BUFQ_OPT_NO_SPARES` will
 * disable that and free chunks once they become empty.
 *
 * When providing a pool to a bufq, all chunk creation and spare handling
 * will be delegated to that pool.
 */
struct bufq {
  struct buf_chunk *head;       /* chunk with bytes to read from */
  struct buf_chunk *tail;       /* chunk to write to */
  struct buf_chunk *spare;      /* list of free chunks, unless `pool` */
  struct bufc_pool *pool;       /* optional pool for free chunks */
  size_t chunk_count;           /* current number of chunks in `head+spare` */
  size_t max_chunks;            /* max `head` chunks to use */
  size_t chunk_size;            /* size of chunks to manage */
  int opts;                     /* options for handling queue, see below */
};

/**
 * Default behaviour: chunk limit is "hard", meaning attempts to write
 * more bytes than can be hold in `max_chunks` is refused and will return
 * -1, CURLE_AGAIN. */
#define BUFQ_OPT_NONE        (0)
/**
 * Make `max_chunks` a "soft" limit. A bufq will report that it is "full"
 * when `max_chunks` are used, but allows writing beyond this limit.
 */
#define BUFQ_OPT_SOFT_LIMIT  (1 << 0)
/**
 * Do not keep spare chunks.
 */
#define BUFQ_OPT_NO_SPARES   (1 << 1)

/**
 * Initialize a buffer queue that can hold up to `max_chunks` buffers
 * each of size `chunk_size`. The bufq will not allow writing of
 * more bytes than can be held in `max_chunks`.
 */
void Curl_bufq_init(struct bufq *q, size_t chunk_size, size_t max_chunks);

/**
 * Initialize a buffer queue that can hold up to `max_chunks` buffers
 * each of size `chunk_size` with the given options. See `BUFQ_OPT_*`.
 */
void Curl_bufq_init2(struct bufq *q, size_t chunk_size,
                     size_t max_chunks, int opts);

void Curl_bufq_initp(struct bufq *q, struct bufc_pool *pool,
                     size_t max_chunks, int opts);

/**
 * Reset the buffer queue to be empty. Will keep any allocated buffer
 * chunks around.
 */
void Curl_bufq_reset(struct bufq *q);

/**
 * Free all resources held by the buffer queue.
 */
void Curl_bufq_free(struct bufq *q);

/**
 * Return the total amount of data in the queue.
 */
size_t Curl_bufq_len(const struct bufq *q);

/**
 * Return the total amount of free space in the queue.
 * The returned length is the number of bytes that can
 * be expected to be written successfully to the bufq,
 * providing no memory allocations fail.
 */
size_t Curl_bufq_space(const struct bufq *q);

/**
 * Returns TRUE iff there is no data in the buffer queue.
 */
bool Curl_bufq_is_empty(const struct bufq *q);

/**
 * Returns TRUE iff there is no space left in the buffer queue.
 */
bool Curl_bufq_is_full(const struct bufq *q);

/**
 * Write buf to the end of the buffer queue. The buf is copied
 * and the amount of copied bytes is returned.
 * A return code of -1 indicates an error, setting `err` to the
 * cause. An err of CURLE_AGAIN is returned if the buffer queue is full.
 */
ssize_t Curl_bufq_write(struct bufq *q,
                        const unsigned char *buf, size_t len,
                        CURLcode *err);

CURLcode Curl_bufq_cwrite(struct bufq *q,
                         const char *buf, size_t len,
                         size_t *pnwritten);

/**
 * Remove `len` bytes from the end of the buffer queue again.
 * Returns CURLE_AGAIN if less than `len` bytes were in the queue.
 */
CURLcode Curl_bufq_unwrite(struct bufq *q, size_t len);

/**
 * Read buf from the start of the buffer queue. The buf is copied
 * and the amount of copied bytes is returned.
 * A return code of -1 indicates an error, setting `err` to the
 * cause. An err of CURLE_AGAIN is returned if the buffer queue is empty.
 */
ssize_t Curl_bufq_read(struct bufq *q, unsigned char *buf, size_t len,
                        CURLcode *err);

CURLcode Curl_bufq_cread(struct bufq *q, char *buf, size_t len,
                         size_t *pnread);

/**
 * Peek at the head chunk in the buffer queue. Returns a pointer to
 * the chunk buf (at the current offset) and its length. Does not
 * modify the buffer queue.
 * Returns TRUE iff bytes are available. Sets `pbuf` to NULL and `plen`
 * to 0 when no bytes are available.
 * Repeated calls return the same information until the buffer queue
 * is modified, see `Curl_bufq_skip()``
 */
bool Curl_bufq_peek(struct bufq *q,
                    const unsigned char **pbuf, size_t *plen);

bool Curl_bufq_peek_at(struct bufq *q, size_t offset,
                       const unsigned char **pbuf, size_t *plen);

/**
 * Tell the buffer queue to discard `amount` buf bytes at the head
 * of the queue. Skipping more buf than is currently buffered will
 * just empty the queue.
 */
void Curl_bufq_skip(struct bufq *q, size_t amount);

typedef ssize_t Curl_bufq_writer(void *writer_ctx,
                                 const unsigned char *buf, size_t len,
                                 CURLcode *err);
/**
 * Passes the chunks in the buffer queue to the writer and returns
 * the amount of buf written. A writer may return -1 and CURLE_AGAIN
 * to indicate blocking at which point the queue will stop and return
 * the amount of buf passed so far.
 * -1 is returned on any other errors reported by the writer.
 * Note that in case of a -1 chunks may have been written and
 * the buffer queue will have different length than before.
 */
ssize_t Curl_bufq_pass(struct bufq *q, Curl_bufq_writer *writer,
                       void *writer_ctx, CURLcode *err);

typedef ssize_t Curl_bufq_reader(void *reader_ctx,
                                 unsigned char *buf, size_t len,
                                 CURLcode *err);

/**
 * Read date and append it to the end of the buffer queue until the
 * reader returns blocking or the queue is full. A reader returns
 * -1 and CURLE_AGAIN to indicate blocking.
 * Returns the total amount of buf read (may be 0) or -1 on other
 * reader errors.
 * Note that in case of a -1 chunks may have been read and
 * the buffer queue will have different length than before.
 */
ssize_t Curl_bufq_slurp(struct bufq *q, Curl_bufq_reader *reader,
                        void *reader_ctx, CURLcode *err);

/**
 * Read *once* up to `max_len` bytes and append it to the buffer.
 * if `max_len` is 0, no limit is imposed besides the chunk space.
 * Returns the total amount of buf read (may be 0) or -1 on other
 * reader errors.
 */
ssize_t Curl_bufq_sipn(struct bufq *q, size_t max_len,
                       Curl_bufq_reader *reader, void *reader_ctx,
                       CURLcode *err);

/**
 * Write buf to the end of the buffer queue.
 * Will write bufq content or passed `buf` directly using the `writer`
 * callback when it sees fit. 'buf' might get passed directly
 * on or is placed into the buffer, depending on `len` and current
 * amount buffered, chunk size, etc.
 */
ssize_t Curl_bufq_write_pass(struct bufq *q,
                             const unsigned char *buf, size_t len,
                             Curl_bufq_writer *writer, void *writer_ctx,
                             CURLcode *err);

#endif /* HEADER_CURL_BUFQ_H */
