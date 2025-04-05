#ifndef HEADER_CURL_REQUEST_H
#define HEADER_CURL_REQUEST_H
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

/* This file is for lib internal stuff */

#include "curl_setup.h"

#include "bufq.h"

/* forward declarations */
struct UserDefined;
#ifndef CURL_DISABLE_DOH
struct doh_probes;
#endif

enum expect100 {
  EXP100_SEND_DATA,           /* enough waiting, just send the body now */
  EXP100_AWAITING_CONTINUE,   /* waiting for the 100 Continue header */
  EXP100_SENDING_REQUEST,     /* still sending the request but will wait for
                                 the 100 header once done with the request */
  EXP100_FAILED               /* used on 417 Expectation Failed */
};

enum upgrade101 {
  UPGR101_INIT,               /* default state */
  UPGR101_WS,                 /* upgrade to WebSockets requested */
  UPGR101_H2,                 /* upgrade to HTTP/2 requested */
  UPGR101_RECEIVED,           /* 101 response received */
  UPGR101_WORKING             /* talking upgraded protocol */
};


/*
 * Request specific data in the easy handle (Curl_easy). Previously,
 * these members were on the connectdata struct but since a conn struct may
 * now be shared between different Curl_easys, we store connection-specific
 * data here. This struct only keeps stuff that is interesting for *this*
 * request, as it will be cleared between multiple ones
 */
struct SingleRequest {
  curl_off_t size;        /* -1 if unknown at this point */
  curl_off_t maxdownload; /* in bytes, the maximum amount of data to fetch,
                             -1 means unlimited */
  curl_off_t bytecount;         /* total number of bytes read */
  curl_off_t writebytecount;    /* number of bytes written */

  struct curltime start;         /* transfer started at this time */
  unsigned int headerbytecount;  /* received server headers (not CONNECT
                                    headers) */
  unsigned int allheadercount;   /* all received headers (server + CONNECT) */
  unsigned int deductheadercount; /* this amount of bytes does not count when
                                     we check if anything has been transferred
                                     at the end of a connection. We use this
                                     counter to make only a 100 reply (without
                                     a following second response code) result
                                     in a CURLE_GOT_NOTHING error code */
  int headerline;               /* counts header lines to better track the
                                   first one */
  curl_off_t offset;            /* possible resume offset read from the
                                   Content-Range: header */
  int httpversion;              /* Version in response (09, 10, 11, etc.) */
  int httpcode;                 /* error code from the 'HTTP/1.? XXX' or
                                   'RTSP/1.? XXX' line */
  int keepon;
  enum upgrade101 upgr101;      /* 101 upgrade state */

  /* Client Writer stack, handles transfer- and content-encodings, protocol
   * checks, pausing by client callbacks. */
  struct Curl_cwriter *writer_stack;
  /* Client Reader stack, handles transfer- and content-encodings, protocol
   * checks, pausing by client callbacks. */
  struct Curl_creader *reader_stack;
  struct bufq sendbuf; /* data which needs to be send to the server */
  size_t sendbuf_hds_len; /* amount of header bytes in sendbuf */
  time_t timeofdoc;
  char *location;   /* This points to an allocated version of the Location:
                       header data */
  char *newurl;     /* Set to the new URL to use when a redirect or a retry is
                       wanted */

  /* Allocated protocol-specific data. Each protocol handler makes sure this
     points to data it needs. */
  union {
    struct FILEPROTO *file;
    struct FTP *ftp;
    struct IMAP *imap;
    struct ldapreqinfo *ldap;
    struct MQTT *mqtt;
    struct POP3 *pop3;
    struct RTSP *rtsp;
    struct smb_request *smb;
    struct SMTP *smtp;
    struct SSHPROTO *ssh;
    struct TELNET *telnet;
  } p;
#ifndef CURL_DISABLE_DOH
  struct doh_probes *doh; /* DoH specific data for this request */
#endif
#ifndef CURL_DISABLE_COOKIES
  unsigned char setcookies;
#endif
  BIT(header);        /* incoming data has HTTP header */
  BIT(done);          /* request is done, e.g. no more send/recv should
                       * happen. This can be TRUE before `upload_done` or
                       * `download_done` is TRUE. */
  BIT(content_range); /* set TRUE if Content-Range: was found */
  BIT(download_done); /* set to TRUE when download is complete */
  BIT(eos_written);   /* iff EOS has been written to client */
  BIT(eos_read);      /* iff EOS has been read from the client */
  BIT(eos_sent);      /* iff EOS has been sent to the server */
  BIT(rewind_read);   /* iff reader needs rewind at next start */
  BIT(upload_done);   /* set to TRUE when all request data has been sent */
  BIT(upload_aborted); /* set to TRUE when upload was aborted. Will also
                        * show `upload_done` as TRUE. */
  BIT(ignorebody);    /* we read a response-body but we ignore it! */
  BIT(http_bodyless); /* HTTP response status code is between 100 and 199,
                         204 or 304 */
  BIT(chunk);         /* if set, this is a chunked transfer-encoding */
  BIT(resp_trailer);  /* response carried 'Trailer:' header field */
  BIT(ignore_cl);     /* ignore content-length */
  BIT(upload_chunky); /* set TRUE if we are doing chunked transfer-encoding
                         on upload */
  BIT(getheader);    /* TRUE if header parsing is wanted */
  BIT(no_body);      /* the response has no body */
  BIT(authneg);      /* TRUE when the auth phase has started, which means
                        that we are creating a request with an auth header,
                        but it is not the final request in the auth
                        negotiation. */
  BIT(sendbuf_init); /* sendbuf is initialized */
  BIT(shutdown);     /* request end will shutdown connection */
  BIT(shutdown_err_ignore); /* errors in shutdown will not fail request */
#ifdef USE_HYPER
  BIT(bodywritten);
#endif
};

/**
 * Initialize the state of the request for first use.
 */
void Curl_req_init(struct SingleRequest *req);

/**
 * The request is about to start. Record time and do a soft reset.
 */
CURLcode Curl_req_start(struct SingleRequest *req,
                        struct Curl_easy *data);

/**
 * The request may continue with a follow up. Reset
 * members, but keep start time for overall duration calc.
 */
CURLcode Curl_req_soft_reset(struct SingleRequest *req,
                             struct Curl_easy *data);

/**
 * The request is done. If not aborted, make sure that buffers are
 * flushed to the client.
 * @param req        the request
 * @param data       the transfer
 * @param aborted    TRUE iff the request was aborted/errored
 */
CURLcode Curl_req_done(struct SingleRequest *req,
                       struct Curl_easy *data, bool aborted);

/**
 * Free the state of the request, not usable afterwards.
 */
void Curl_req_free(struct SingleRequest *req, struct Curl_easy *data);

/**
 * Hard reset the state of the request to virgin state base on
 * transfer settings.
 */
void Curl_req_hard_reset(struct SingleRequest *req, struct Curl_easy *data);

#ifndef USE_HYPER
/**
 * Send request headers. If not all could be sent
 * they will be buffered. Use `Curl_req_flush()` to make sure
 * bytes are really send.
 * @param data      the transfer making the request
 * @param buf       the complete header bytes, no body
 * @return CURLE_OK (on blocking with *pnwritten == 0) or error.
 */
CURLcode Curl_req_send(struct Curl_easy *data, struct dynbuf *buf);

#endif /* !USE_HYPER */

/**
 * TRUE iff the request has sent all request headers and data.
 */
bool Curl_req_done_sending(struct Curl_easy *data);

/*
 * Read more from client and flush all buffered request bytes.
 * @return CURLE_OK on success or the error on the sending.
 *         Never returns CURLE_AGAIN.
 */
CURLcode Curl_req_send_more(struct Curl_easy *data);

/**
 * TRUE iff the request wants to send, e.g. has buffered bytes.
 */
bool Curl_req_want_send(struct Curl_easy *data);

/**
 * TRUE iff the request has no buffered bytes yet to send.
 */
bool Curl_req_sendbuf_empty(struct Curl_easy *data);

/**
 * Stop sending any more request data to the server.
 * Will clear the send buffer and mark request sending as done.
 */
CURLcode Curl_req_abort_sending(struct Curl_easy *data);

/**
 * Stop sending and receiving any more request data.
 * Will abort sending if not done.
 */
CURLcode Curl_req_stop_send_recv(struct Curl_easy *data);

/**
 * Invoked when all request data has been uploaded.
 */
CURLcode Curl_req_set_upload_done(struct Curl_easy *data);

#endif /* HEADER_CURL_REQUEST_H */
