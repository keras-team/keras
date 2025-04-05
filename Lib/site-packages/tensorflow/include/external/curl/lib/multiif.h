#ifndef HEADER_CURL_MULTIIF_H
#define HEADER_CURL_MULTIIF_H
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

/*
 * Prototypes for library-wide functions provided by multi.c
 */

CURLcode Curl_updatesocket(struct Curl_easy *data);
void Curl_expire(struct Curl_easy *data, timediff_t milli, expire_id);
bool Curl_expire_clear(struct Curl_easy *data);
void Curl_expire_done(struct Curl_easy *data, expire_id id);
CURLMcode Curl_update_timer(struct Curl_multi *multi) WARN_UNUSED_RESULT;
void Curl_attach_connection(struct Curl_easy *data,
                             struct connectdata *conn);
void Curl_detach_connection(struct Curl_easy *data);
bool Curl_multiplex_wanted(const struct Curl_multi *multi);
void Curl_set_in_callback(struct Curl_easy *data, bool value);
bool Curl_is_in_callback(struct Curl_easy *data);
CURLcode Curl_preconnect(struct Curl_easy *data);

void Curl_multi_connchanged(struct Curl_multi *multi);

/* Internal version of curl_multi_init() accepts size parameters for the
   socket, connection and dns hashes */
struct Curl_multi *Curl_multi_handle(size_t hashsize,
                                     size_t chashsize,
                                     size_t dnssize);

/* the write bits start at bit 16 for the *getsock() bitmap */
#define GETSOCK_WRITEBITSTART 16

#define GETSOCK_BLANK 0 /* no bits set */

/* set the bit for the given sock number to make the bitmap for writable */
#define GETSOCK_WRITESOCK(x) (1 << (GETSOCK_WRITEBITSTART + (x)))

/* set the bit for the given sock number to make the bitmap for readable */
#define GETSOCK_READSOCK(x) (1 << (x))

/* mask for checking if read and/or write is set for index x */
#define GETSOCK_MASK_RW(x) (GETSOCK_READSOCK(x)|GETSOCK_WRITESOCK(x))

/*
 * Curl_multi_closed()
 *
 * Used by the connect code to tell the multi_socket code that one of the
 * sockets we were using is about to be closed. This function will then
 * remove it from the sockethash for this handle to make the multi_socket API
 * behave properly, especially for the case when libcurl will create another
 * socket again and it gets the same file descriptor number.
 */

void Curl_multi_closed(struct Curl_easy *data, curl_socket_t s);

/* Compare the two pollsets to notify the multi_socket API of changes
 * in socket polling, e.g calling multi->socket_cb() with the changes if
 * differences are seen.
 */
CURLMcode Curl_multi_pollset_ev(struct Curl_multi *multi,
                                struct Curl_easy *data,
                                struct easy_pollset *ps,
                                struct easy_pollset *last_ps);

/*
 * Add a handle and move it into PERFORM state at once. For pushed streams.
 */
CURLMcode Curl_multi_add_perform(struct Curl_multi *multi,
                                 struct Curl_easy *data,
                                 struct connectdata *conn);


/* Return the value of the CURLMOPT_MAX_CONCURRENT_STREAMS option */
unsigned int Curl_multi_max_concurrent_streams(struct Curl_multi *multi);

/**
 * Borrow the transfer buffer from the multi, suitable
 * for the given transfer `data`. The buffer may only be used in one
 * multi processing of the easy handle. It MUST be returned to the
 * multi before it can be borrowed again.
 * Pointers into the buffer remain only valid as long as it is borrowed.
 *
 * @param data    the easy handle
 * @param pbuf    on return, the buffer to use or NULL on error
 * @param pbuflen on return, the size of *pbuf or 0 on error
 * @return CURLE_OK when buffer is available and is returned.
 *         CURLE_OUT_OF_MEMORy on failure to allocate the buffer,
 *         CURLE_FAILED_INIT if the easy handle is without multi.
 *         CURLE_AGAIN if the buffer is borrowed already.
 */
CURLcode Curl_multi_xfer_buf_borrow(struct Curl_easy *data,
                                   char **pbuf, size_t *pbuflen);
/**
 * Release the borrowed buffer. All references into the buffer become
 * invalid after this.
 * @param buf the buffer pointer borrowed for coding error checks.
 */
void Curl_multi_xfer_buf_release(struct Curl_easy *data, char *buf);

/**
 * Borrow the upload buffer from the multi, suitable
 * for the given transfer `data`. The buffer may only be used in one
 * multi processing of the easy handle. It MUST be returned to the
 * multi before it can be borrowed again.
 * Pointers into the buffer remain only valid as long as it is borrowed.
 *
 * @param data    the easy handle
 * @param pbuf    on return, the buffer to use or NULL on error
 * @param pbuflen on return, the size of *pbuf or 0 on error
 * @return CURLE_OK when buffer is available and is returned.
 *         CURLE_OUT_OF_MEMORy on failure to allocate the buffer,
 *         CURLE_FAILED_INIT if the easy handle is without multi.
 *         CURLE_AGAIN if the buffer is borrowed already.
 */
CURLcode Curl_multi_xfer_ulbuf_borrow(struct Curl_easy *data,
                                      char **pbuf, size_t *pbuflen);

/**
 * Release the borrowed upload buffer. All references into the buffer become
 * invalid after this.
 * @param buf the upload buffer pointer borrowed for coding error checks.
 */
void Curl_multi_xfer_ulbuf_release(struct Curl_easy *data, char *buf);

/**
 * Borrow the socket scratch buffer from the multi, suitable
 * for the given transfer `data`. The buffer may only be used for
 * direct socket I/O operation by one connection at a time and MUST be
 * returned to the multi before the I/O call returns.
 * Pointers into the buffer remain only valid as long as it is borrowed.
 *
 * @param data    the easy handle
 * @param blen    requested length of the buffer
 * @param pbuf    on return, the buffer to use or NULL on error
 * @return CURLE_OK when buffer is available and is returned.
 *         CURLE_OUT_OF_MEMORy on failure to allocate the buffer,
 *         CURLE_FAILED_INIT if the easy handle is without multi.
 *         CURLE_AGAIN if the buffer is borrowed already.
 */
CURLcode Curl_multi_xfer_sockbuf_borrow(struct Curl_easy *data,
                                        size_t blen, char **pbuf);
/**
 * Release the borrowed buffer. All references into the buffer become
 * invalid after this.
 * @param buf the buffer pointer borrowed for coding error checks.
 */
void Curl_multi_xfer_sockbuf_release(struct Curl_easy *data, char *buf);

/**
 * Get the transfer handle for the given id. Returns NULL if not found.
 */
struct Curl_easy *Curl_multi_get_handle(struct Curl_multi *multi,
                                        curl_off_t id);

#endif /* HEADER_CURL_MULTIIF_H */
