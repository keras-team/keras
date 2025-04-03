#ifndef HEADER_CURL_URLDATA_H
#define HEADER_CURL_URLDATA_H
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

#define PORT_FTP 21
#define PORT_FTPS 990
#define PORT_TELNET 23
#define PORT_HTTP 80
#define PORT_HTTPS 443
#define PORT_DICT 2628
#define PORT_LDAP 389
#define PORT_LDAPS 636
#define PORT_TFTP 69
#define PORT_SSH 22
#define PORT_IMAP 143
#define PORT_IMAPS 993
#define PORT_POP3 110
#define PORT_POP3S 995
#define PORT_SMB 445
#define PORT_SMBS 445
#define PORT_SMTP 25
#define PORT_SMTPS 465 /* sometimes called SSMTP */
#define PORT_RTSP 554
#define PORT_RTMP 1935
#define PORT_RTMPT PORT_HTTP
#define PORT_RTMPS PORT_HTTPS
#define PORT_GOPHER 70
#define PORT_MQTT 1883

struct curl_trc_featt;

#ifdef USE_ECH
/* CURLECH_ bits for the tls_ech option */
# define CURLECH_DISABLE    (1<<0)
# define CURLECH_GREASE     (1<<1)
# define CURLECH_ENABLE     (1<<2)
# define CURLECH_HARD       (1<<3)
# define CURLECH_CLA_CFG    (1<<4)
#endif

#ifndef CURL_DISABLE_WEBSOCKETS
/* CURLPROTO_GOPHERS (29) is the highest publicly used protocol bit number,
 * the rest are internal information. If we use higher bits we only do this on
 * platforms that have a >= 64-bit type and then we use such a type for the
 * protocol fields in the protocol handler.
 */
#define CURLPROTO_WS     (1<<30)
#define CURLPROTO_WSS    ((curl_prot_t)1<<31)
#else
#define CURLPROTO_WS 0
#define CURLPROTO_WSS 0
#endif

/* the default protocols accepting a redirect to */
#define CURLPROTO_REDIR (CURLPROTO_HTTP | CURLPROTO_HTTPS | CURLPROTO_FTP | \
                         CURLPROTO_FTPS)

/* This should be undefined once we need bit 32 or higher */
#define PROTO_TYPE_SMALL

#ifndef PROTO_TYPE_SMALL
typedef curl_off_t curl_prot_t;
#else
typedef unsigned int curl_prot_t;
#endif

/* This mask is for all the old protocols that are provided and defined in the
   public header and shall exclude protocols added since which are not exposed
   in the API */
#define CURLPROTO_MASK   (0x3ffffff)

#define DICT_MATCH "/MATCH:"
#define DICT_MATCH2 "/M:"
#define DICT_MATCH3 "/FIND:"
#define DICT_DEFINE "/DEFINE:"
#define DICT_DEFINE2 "/D:"
#define DICT_DEFINE3 "/LOOKUP:"

#define CURL_DEFAULT_USER "anonymous"
#define CURL_DEFAULT_PASSWORD "ftp@example.com"

#if !defined(_WIN32) && !defined(MSDOS) && !defined(__EMX__)
/* do FTP line-end CRLF => LF conversions on platforms that prefer LF-only. It
   also means: keep CRLF line endings on the CRLF platforms */
#define CURL_PREFER_LF_LINEENDS
#endif

/* Convenience defines for checking protocols or their SSL based version. Each
   protocol handler should only ever have a single CURLPROTO_ in its protocol
   field. */
#define PROTO_FAMILY_HTTP (CURLPROTO_HTTP|CURLPROTO_HTTPS|CURLPROTO_WS| \
                           CURLPROTO_WSS)
#define PROTO_FAMILY_FTP  (CURLPROTO_FTP|CURLPROTO_FTPS)
#define PROTO_FAMILY_POP3 (CURLPROTO_POP3|CURLPROTO_POP3S)
#define PROTO_FAMILY_SMB  (CURLPROTO_SMB|CURLPROTO_SMBS)
#define PROTO_FAMILY_SMTP (CURLPROTO_SMTP|CURLPROTO_SMTPS)
#define PROTO_FAMILY_SSH  (CURLPROTO_SCP|CURLPROTO_SFTP)

#if !defined(CURL_DISABLE_FTP) || defined(USE_SSH) ||   \
  !defined(CURL_DISABLE_POP3) || !defined(CURL_DISABLE_FILE)
/* these protocols support CURLOPT_DIRLISTONLY */
#define CURL_LIST_ONLY_PROTOCOL 1
#endif

#define DEFAULT_CONNCACHE_SIZE 5

/* length of longest IPv6 address string including the trailing null */
#define MAX_IPADR_LEN sizeof("ffff:ffff:ffff:ffff:ffff:ffff:255.255.255.255")

/* Default FTP/IMAP etc response timeout in milliseconds */
#define RESP_TIMEOUT (120*1000)

/* Max string input length is a precaution against abuse and to detect junk
   input easier and better. */
#define CURL_MAX_INPUT_LENGTH 8000000


#include "cookie.h"
#include "psl.h"
#include "formdata.h"

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETINET_IN6_H
#include <netinet/in6.h>
#endif

#include "timeval.h"

#include <curl/curl.h>

#include "http_chunks.h" /* for the structs and enum stuff */
#include "hostip.h"
#include "hash.h"
#include "splay.h"
#include "dynbuf.h"
#include "dynhds.h"
#include "request.h"
#include "netrc.h"

/* return the count of bytes sent, or -1 on error */
typedef ssize_t (Curl_send)(struct Curl_easy *data,   /* transfer */
                            int sockindex,            /* socketindex */
                            const void *buf,          /* data to write */
                            size_t len,               /* max amount to write */
                            bool eos,                 /* last chunk */
                            CURLcode *err);           /* error to return */

/* return the count of bytes read, or -1 on error */
typedef ssize_t (Curl_recv)(struct Curl_easy *data,   /* transfer */
                            int sockindex,            /* socketindex */
                            char *buf,                /* store data here */
                            size_t len,               /* max amount to read */
                            CURLcode *err);           /* error to return */

#ifdef USE_HYPER
typedef CURLcode (*Curl_datastream)(struct Curl_easy *data,
                                    struct connectdata *conn,
                                    int *didwhat,
                                    int select_res);
#endif

#include "mime.h"
#include "imap.h"
#include "pop3.h"
#include "smtp.h"
#include "ftp.h"
#include "file.h"
#include "vssh/ssh.h"
#include "http.h"
#include "rtsp.h"
#include "smb.h"
#include "mqtt.h"
#include "ftplistparser.h"
#include "multihandle.h"
#include "c-hyper.h"
#include "cf-socket.h"

#ifdef HAVE_GSSAPI
# ifdef HAVE_GSSGNU
#  include <gss.h>
# elif defined HAVE_GSSAPI_GSSAPI_H
#  include <gssapi/gssapi.h>
# else
#  include <gssapi.h>
# endif
# ifdef HAVE_GSSAPI_GSSAPI_GENERIC_H
#  include <gssapi/gssapi_generic.h>
# endif
#endif

#ifdef USE_LIBSSH2
#include <libssh2.h>
#include <libssh2_sftp.h>
#endif /* USE_LIBSSH2 */

#define READBUFFER_SIZE CURL_MAX_WRITE_SIZE
#define READBUFFER_MAX  CURL_MAX_READ_SIZE
#define READBUFFER_MIN  1024

/* The default upload buffer size, should not be smaller than
   CURL_MAX_WRITE_SIZE, as it needs to hold a full buffer as could be sent in
   a write callback.

   The size was 16KB for many years but was bumped to 64KB because it makes
   libcurl able to do significantly faster uploads in some circumstances. Even
   larger buffers can help further, but this is deemed a fair memory/speed
   compromise. */
#define UPLOADBUFFER_DEFAULT 65536
#define UPLOADBUFFER_MAX (2*1024*1024)
#define UPLOADBUFFER_MIN CURL_MAX_WRITE_SIZE

#define CURLEASY_MAGIC_NUMBER 0xc0dedbadU
#ifdef DEBUGBUILD
/* On a debug build, we want to fail hard on easy handles that
 * are not NULL, but no longer have the MAGIC touch. This gives
 * us early warning on things only discovered by valgrind otherwise. */
#define GOOD_EASY_HANDLE(x) \
  (((x) && ((x)->magic == CURLEASY_MAGIC_NUMBER))? TRUE: \
  (DEBUGASSERT(!(x)), FALSE))
#else
#define GOOD_EASY_HANDLE(x) \
  ((x) && ((x)->magic == CURLEASY_MAGIC_NUMBER))
#endif

#ifdef HAVE_GSSAPI
/* Types needed for krb5-ftp connections */
struct krb5buffer {
  struct dynbuf buf;
  size_t index;
  BIT(eof_flag);
};

enum protection_level {
  PROT_NONE, /* first in list */
  PROT_CLEAR,
  PROT_SAFE,
  PROT_CONFIDENTIAL,
  PROT_PRIVATE,
  PROT_CMD,
  PROT_LAST /* last in list */
};
#endif

/* SSL backend-specific data; declared differently by each SSL backend */
struct ssl_backend_data;

typedef enum {
  CURL_SSL_PEER_DNS,
  CURL_SSL_PEER_IPV4,
  CURL_SSL_PEER_IPV6
} ssl_peer_type;

struct ssl_peer {
  char *hostname;        /* hostname for verification */
  char *dispname;        /* display version of hostname */
  char *sni;             /* SNI version of hostname or NULL if not usable */
  ssl_peer_type type;    /* type of the peer information */
  int port;              /* port we are talking to */
  int transport;         /* one of TRNSPRT_* defines */
};

struct ssl_primary_config {
  char *CApath;          /* certificate dir (does not work on Windows) */
  char *CAfile;          /* certificate to verify peer against */
  char *issuercert;      /* optional issuer certificate filename */
  char *clientcert;
  char *cipher_list;     /* list of ciphers to use */
  char *cipher_list13;   /* list of TLS 1.3 cipher suites to use */
  char *pinned_key;
  char *CRLfile;         /* CRL to check certificate revocation */
  struct curl_blob *cert_blob;
  struct curl_blob *ca_info_blob;
  struct curl_blob *issuercert_blob;
#ifdef USE_TLS_SRP
  char *username; /* TLS username (for, e.g., SRP) */
  char *password; /* TLS password (for, e.g., SRP) */
#endif
  char *curves;          /* list of curves to use */
  unsigned char ssl_options;  /* the CURLOPT_SSL_OPTIONS bitmask */
  unsigned int version_max; /* max supported version the client wants to use */
  unsigned char version;    /* what version the client wants to use */
  BIT(verifypeer);       /* set TRUE if this is desired */
  BIT(verifyhost);       /* set TRUE if CN/SAN must match hostname */
  BIT(verifystatus);     /* set TRUE if certificate status must be checked */
  BIT(cache_session);    /* cache session or not */
};

struct ssl_config_data {
  struct ssl_primary_config primary;
  long certverifyresult; /* result from the certificate verification */
  curl_ssl_ctx_callback fsslctx; /* function to initialize ssl ctx */
  void *fsslctxp;        /* parameter for call back */
  char *cert_type; /* format for certificate (default: PEM)*/
  char *key; /* private key filename */
  struct curl_blob *key_blob;
  char *key_type; /* format for private key (default: PEM) */
  char *key_passwd; /* plain text private key password */
  BIT(certinfo);     /* gather lots of certificate info */
  BIT(falsestart);
  BIT(earlydata);    /* use tls1.3 early data */
  BIT(enable_beast); /* allow this flaw for interoperability's sake */
  BIT(no_revoke);    /* disable SSL certificate revocation checks */
  BIT(no_partialchain); /* do not accept partial certificate chains */
  BIT(revoke_best_effort); /* ignore SSL revocation offline/missing revocation
                              list errors */
  BIT(native_ca_store); /* use the native ca store of operating system */
  BIT(auto_client_cert);   /* automatically locate and use a client
                              certificate for authentication (Schannel) */
};

struct ssl_general_config {
  size_t max_ssl_sessions; /* SSL session id cache size */
  int ca_cache_timeout;  /* Certificate store cache timeout (seconds) */
};

typedef void Curl_ssl_sessionid_dtor(void *sessionid, size_t idsize);

/* information stored about one single SSL session */
struct Curl_ssl_session {
  char *name;       /* hostname for which this ID was used */
  char *conn_to_host; /* hostname for the connection (may be NULL) */
  const char *scheme; /* protocol scheme used */
  char *alpn;         /* APLN TLS negotiated protocol string */
  void *sessionid;  /* as returned from the SSL layer */
  size_t idsize;    /* if known, otherwise 0 */
  Curl_ssl_sessionid_dtor *sessionid_free; /* free `sessionid` callback */
  long age;         /* just a number, the higher the more recent */
  int remote_port;  /* remote port */
  int conn_to_port; /* remote port for the connection (may be -1) */
  int transport;    /* TCP or QUIC */
  struct ssl_primary_config ssl_config; /* setup for this session */
};

#ifdef USE_WINDOWS_SSPI
#include "curl_sspi.h"
#endif

#ifndef CURL_DISABLE_DIGEST_AUTH
/* Struct used for Digest challenge-response authentication */
struct digestdata {
#if defined(USE_WINDOWS_SSPI)
  BYTE *input_token;
  size_t input_token_len;
  CtxtHandle *http_context;
  /* copy of user/passwd used to make the identity for http_context.
     either may be NULL. */
  char *user;
  char *passwd;
#else
  char *nonce;
  char *cnonce;
  char *realm;
  char *opaque;
  char *qop;
  char *algorithm;
  int nc; /* nonce count */
  unsigned char algo;
  BIT(stale); /* set true for re-negotiation */
  BIT(userhash);
#endif
};
#endif

typedef enum {
  NTLMSTATE_NONE,
  NTLMSTATE_TYPE1,
  NTLMSTATE_TYPE2,
  NTLMSTATE_TYPE3,
  NTLMSTATE_LAST
} curlntlm;

typedef enum {
  GSS_AUTHNONE,
  GSS_AUTHRECV,
  GSS_AUTHSENT,
  GSS_AUTHDONE,
  GSS_AUTHSUCC
} curlnegotiate;

/* Struct used for GSSAPI (Kerberos V5) authentication */
#if defined(USE_KERBEROS5)
struct kerberos5data {
#if defined(USE_WINDOWS_SSPI)
  CredHandle *credentials;
  CtxtHandle *context;
  TCHAR *spn;
  SEC_WINNT_AUTH_IDENTITY identity;
  SEC_WINNT_AUTH_IDENTITY *p_identity;
  size_t token_max;
  BYTE *output_token;
#else
  gss_ctx_id_t context;
  gss_name_t spn;
#endif
};
#endif

/* Struct used for SCRAM-SHA-1 authentication */
#ifdef USE_GSASL
#include <gsasl.h>
struct gsasldata {
  Gsasl *ctx;
  Gsasl_session *client;
};
#endif

/* Struct used for NTLM challenge-response authentication */
#if defined(USE_NTLM)
struct ntlmdata {
#ifdef USE_WINDOWS_SSPI
/* The sslContext is used for the Schannel bindings. The
 * api is available on the Windows 7 SDK and later.
 */
#ifdef SECPKG_ATTR_ENDPOINT_BINDINGS
  CtxtHandle *sslContext;
#endif
  CredHandle *credentials;
  CtxtHandle *context;
  SEC_WINNT_AUTH_IDENTITY identity;
  SEC_WINNT_AUTH_IDENTITY *p_identity;
  size_t token_max;
  BYTE *output_token;
  BYTE *input_token;
  size_t input_token_len;
  TCHAR *spn;
#else
  unsigned int flags;
  unsigned char nonce[8];
  unsigned int target_info_len;
  void *target_info; /* TargetInfo received in the NTLM type-2 message */
#endif
};
#endif

/* Struct used for Negotiate (SPNEGO) authentication */
#ifdef USE_SPNEGO
struct negotiatedata {
#ifdef HAVE_GSSAPI
  OM_uint32 status;
  gss_ctx_id_t context;
  gss_name_t spn;
  gss_buffer_desc output_token;
  struct dynbuf channel_binding_data;
#else
#ifdef USE_WINDOWS_SSPI
#ifdef SECPKG_ATTR_ENDPOINT_BINDINGS
  CtxtHandle *sslContext;
#endif
  DWORD status;
  CredHandle *credentials;
  CtxtHandle *context;
  SEC_WINNT_AUTH_IDENTITY identity;
  SEC_WINNT_AUTH_IDENTITY *p_identity;
  TCHAR *spn;
  size_t token_max;
  BYTE *output_token;
  size_t output_token_length;
#endif
#endif
  BIT(noauthpersist);
  BIT(havenoauthpersist);
  BIT(havenegdata);
  BIT(havemultiplerequests);
};
#endif

#ifdef CURL_DISABLE_PROXY
#define CONN_IS_PROXIED(x) 0
#else
#define CONN_IS_PROXIED(x) x->bits.proxy
#endif

/*
 * Boolean values that concerns this connection.
 */
struct ConnectBits {
#ifndef CURL_DISABLE_PROXY
  BIT(httpproxy);  /* if set, this transfer is done through an HTTP proxy */
  BIT(socksproxy); /* if set, this transfer is done through a socks proxy */
  BIT(proxy_user_passwd); /* user+password for the proxy? */
  BIT(tunnel_proxy);  /* if CONNECT is used to "tunnel" through the proxy.
                         This is implicit when SSL-protocols are used through
                         proxies, but can also be enabled explicitly by
                         apps */
  BIT(proxy); /* if set, this transfer is done through a proxy - any type */
#endif
  /* always modify bits.close with the connclose() and connkeep() macros! */
  BIT(close); /* if set, we close the connection after this request */
  BIT(reuse); /* if set, this is a reused connection */
  BIT(altused); /* this is an alt-svc "redirect" */
  BIT(conn_to_host); /* if set, this connection has a "connect to host"
                        that overrides the host in the URL */
  BIT(conn_to_port); /* if set, this connection has a "connect to port"
                        that overrides the port in the URL (remote port) */
  BIT(ipv6_ip); /* we communicate with a remote site specified with pure IPv6
                   IP address */
  BIT(ipv6);    /* we communicate with a site using an IPv6 address */
  BIT(do_more); /* this is set TRUE if the ->curl_do_more() function is
                   supposed to be called, after ->curl_do() */
  BIT(protoconnstart);/* the protocol layer has STARTED its operation after
                         the TCP layer connect */
  BIT(retry);         /* this connection is about to get closed and then
                         re-attempted at another connection. */
#ifndef CURL_DISABLE_FTP
  BIT(ftp_use_epsv);  /* As set with CURLOPT_FTP_USE_EPSV, but if we find out
                         EPSV does not work we disable it for the forthcoming
                         requests */
  BIT(ftp_use_eprt);  /* As set with CURLOPT_FTP_USE_EPRT, but if we find out
                         EPRT does not work we disable it for the forthcoming
                         requests */
  BIT(ftp_use_data_ssl); /* Enabled SSL for the data connection */
  BIT(ftp_use_control_ssl); /* Enabled SSL for the control connection */
#endif
#ifndef CURL_DISABLE_NETRC
  BIT(netrc);         /* name+password provided by netrc */
#endif
  BIT(bound); /* set true if bind() has already been done on this socket/
                 connection */
  BIT(asks_multiplex); /* connection asks for multiplexing, but is not yet */
  BIT(multiplex); /* connection is multiplexed */
  BIT(tcp_fastopen); /* use TCP Fast Open */
  BIT(tls_enable_alpn); /* TLS ALPN extension? */
#ifndef CURL_DISABLE_DOH
  BIT(doh);
#endif
#ifdef USE_UNIX_SOCKETS
  BIT(abstract_unix_socket);
#endif
  BIT(tls_upgraded);
  BIT(sock_accepted); /* TRUE if the SECONDARYSOCKET was created with
                         accept() */
  BIT(parallel_connect); /* set TRUE when a parallel connect attempt has
                            started (happy eyeballs) */
  BIT(aborted); /* connection was aborted, e.g. in unclean state */
  BIT(shutdown_handler); /* connection shutdown: handler shut down */
  BIT(shutdown_filters); /* connection shutdown: filters shut down */
  BIT(in_cpool);     /* connection is kept in a connection pool */
};

struct hostname {
  char *rawalloc; /* allocated "raw" version of the name */
  char *encalloc; /* allocated IDN-encoded version of the name */
  char *name;     /* name to use internally, might be encoded, might be raw */
  const char *dispname; /* name to display, as 'name' might be encoded */
};

/*
 * Flags on the keepon member of the Curl_transfer_keeper
 */

#define KEEP_NONE  0
#define KEEP_RECV  (1<<0)     /* there is or may be data to read */
#define KEEP_SEND (1<<1)     /* there is or may be data to write */
#define KEEP_RECV_HOLD (1<<2) /* when set, no reading should be done but there
                                 might still be data to read */
#define KEEP_SEND_HOLD (1<<3) /* when set, no writing should be done but there
                                  might still be data to write */
#define KEEP_RECV_PAUSE (1<<4) /* reading is paused */
#define KEEP_SEND_PAUSE (1<<5) /* writing is paused */

/* KEEP_SEND_TIMED is set when the transfer should attempt sending
 * at timer (or other) events. A transfer waiting on a timer will
  * remove KEEP_SEND to suppress POLLOUTs of the connection.
  * Adding KEEP_SEND_TIMED will then attempt to send whenever the transfer
  * enters the "readwrite" loop, e.g. when a timer fires.
  * This is used in HTTP for 'Expect: 100-continue' waiting. */
#define KEEP_SEND_TIMED (1<<6)

#define KEEP_RECVBITS (KEEP_RECV | KEEP_RECV_HOLD | KEEP_RECV_PAUSE)
#define KEEP_SENDBITS (KEEP_SEND | KEEP_SEND_HOLD | KEEP_SEND_PAUSE)

/* transfer wants to send is not PAUSE or HOLD */
#define CURL_WANT_SEND(data) \
  (((data)->req.keepon & KEEP_SENDBITS) == KEEP_SEND)
/* transfer receive is not on PAUSE or HOLD */
#define CURL_WANT_RECV(data) \
  (((data)->req.keepon & KEEP_RECVBITS) == KEEP_RECV)

#if defined(CURLRES_ASYNCH) || !defined(CURL_DISABLE_DOH)
#define USE_CURL_ASYNC
struct Curl_async {
  char *hostname;
  struct Curl_dns_entry *dns;
  struct thread_data *tdata;
  void *resolver; /* resolver state, if it is used in the URL state -
                     ares_channel e.g. */
  int port;
  int status; /* if done is TRUE, this is the status from the callback */
  BIT(done);  /* set TRUE when the lookup is complete */
};

#endif

#define FIRSTSOCKET     0
#define SECONDARYSOCKET 1

/* Polling requested by an easy handle.
 * `action` is CURL_POLL_IN, CURL_POLL_OUT or CURL_POLL_INOUT.
 */
struct easy_pollset {
  curl_socket_t sockets[MAX_SOCKSPEREASYHANDLE];
  unsigned int num;
  unsigned char actions[MAX_SOCKSPEREASYHANDLE];
};

/*
 * Specific protocol handler.
 */

struct Curl_handler {
  const char *scheme;        /* URL scheme name in lowercase */

  /* Complement to setup_connection_internals(). This is done before the
     transfer "owns" the connection. */
  CURLcode (*setup_connection)(struct Curl_easy *data,
                               struct connectdata *conn);

  /* These two functions MUST be set to be protocol dependent */
  CURLcode (*do_it)(struct Curl_easy *data, bool *done);
  CURLcode (*done)(struct Curl_easy *, CURLcode, bool);

  /* If the curl_do() function is better made in two halves, this
   * curl_do_more() function will be called afterwards, if set. For example
   * for doing the FTP stuff after the PASV/PORT command.
   */
  CURLcode (*do_more)(struct Curl_easy *, int *);

  /* This function *MAY* be set to a protocol-dependent function that is run
   * after the connect() and everything is done, as a step in the connection.
   * The 'done' pointer points to a bool that should be set to TRUE if the
   * function completes before return. If it does not complete, the caller
   * should call the ->connecting() function until it is.
   */
  CURLcode (*connect_it)(struct Curl_easy *data, bool *done);

  /* See above. */
  CURLcode (*connecting)(struct Curl_easy *data, bool *done);
  CURLcode (*doing)(struct Curl_easy *data, bool *done);

  /* Called from the multi interface during the PROTOCONNECT phase, and it
     should then return a proper fd set */
  int (*proto_getsock)(struct Curl_easy *data,
                       struct connectdata *conn, curl_socket_t *socks);

  /* Called from the multi interface during the DOING phase, and it should
     then return a proper fd set */
  int (*doing_getsock)(struct Curl_easy *data,
                       struct connectdata *conn, curl_socket_t *socks);

  /* Called from the multi interface during the DO_MORE phase, and it should
     then return a proper fd set */
  int (*domore_getsock)(struct Curl_easy *data,
                        struct connectdata *conn, curl_socket_t *socks);

  /* Called from the multi interface during the DO_DONE, PERFORM and
     WAITPERFORM phases, and it should then return a proper fd set. Not setting
     this will make libcurl use the generic default one. */
  int (*perform_getsock)(struct Curl_easy *data,
                         struct connectdata *conn, curl_socket_t *socks);

  /* This function *MAY* be set to a protocol-dependent function that is run
   * by the curl_disconnect(), as a step in the disconnection. If the handler
   * is called because the connection has been considered dead,
   * dead_connection is set to TRUE. The connection is (again) associated with
   * the transfer here.
   */
  CURLcode (*disconnect)(struct Curl_easy *, struct connectdata *,
                         bool dead_connection);

  /* If used, this function gets called from transfer.c to
     allow the protocol to do extra handling in writing response to
     the client. */
  CURLcode (*write_resp)(struct Curl_easy *data, const char *buf, size_t blen,
                         bool is_eos);

  /* If used, this function gets called from transfer.c to
     allow the protocol to do extra handling in writing a single response
     header line to the client. */
  CURLcode (*write_resp_hd)(struct Curl_easy *data,
                            const char *hd, size_t hdlen, bool is_eos);

  /* This function can perform various checks on the connection. See
     CONNCHECK_* for more information about the checks that can be performed,
     and CONNRESULT_* for the results that can be returned. */
  unsigned int (*connection_check)(struct Curl_easy *data,
                                   struct connectdata *conn,
                                   unsigned int checks_to_perform);

  /* attach() attaches this transfer to this connection */
  void (*attach)(struct Curl_easy *data, struct connectdata *conn);

  int defport;            /* Default port. */
  curl_prot_t protocol;  /* See CURLPROTO_* - this needs to be the single
                            specific protocol bit */
  curl_prot_t family;    /* single bit for protocol family; basically the
                            non-TLS name of the protocol this is */
  unsigned int flags;     /* Extra particular characteristics, see PROTOPT_* */

};

#define PROTOPT_NONE 0             /* nothing extra */
#define PROTOPT_SSL (1<<0)         /* uses SSL */
#define PROTOPT_DUAL (1<<1)        /* this protocol uses two connections */
#define PROTOPT_CLOSEACTION (1<<2) /* need action before socket close */
/* some protocols will have to call the underlying functions without regard to
   what exact state the socket signals. IE even if the socket says "readable",
   the send function might need to be called while uploading, or vice versa.
*/
#define PROTOPT_DIRLOCK (1<<3)
#define PROTOPT_NONETWORK (1<<4)   /* protocol does not use the network! */
#define PROTOPT_NEEDSPWD (1<<5)    /* needs a password, and if none is set it
                                      gets a default */
#define PROTOPT_NOURLQUERY (1<<6)   /* protocol cannot handle
                                       URL query strings (?foo=bar) ! */
#define PROTOPT_CREDSPERREQUEST (1<<7) /* requires login credentials per
                                          request instead of per connection */
#define PROTOPT_ALPN (1<<8) /* set ALPN for this */
/* (1<<9) was PROTOPT_STREAM, now free */
#define PROTOPT_URLOPTIONS (1<<10) /* allow options part in the userinfo field
                                      of the URL */
#define PROTOPT_PROXY_AS_HTTP (1<<11) /* allow this non-HTTP scheme over a
                                         HTTP proxy as HTTP proxies may know
                                         this protocol and act as a gateway */
#define PROTOPT_WILDCARD (1<<12) /* protocol supports wildcard matching */
#define PROTOPT_USERPWDCTRL (1<<13) /* Allow "control bytes" (< 32 ASCII) in
                                       username and password */
#define PROTOPT_NOTCPPROXY (1<<14) /* this protocol cannot proxy over TCP */

#define CONNCHECK_NONE 0                 /* No checks */
#define CONNCHECK_ISDEAD (1<<0)          /* Check if the connection is dead. */
#define CONNCHECK_KEEPALIVE (1<<1)       /* Perform any keepalive function. */

#define CONNRESULT_NONE 0                /* No extra information. */
#define CONNRESULT_DEAD (1<<0)           /* The connection is dead. */

struct ip_quadruple {
  char remote_ip[MAX_IPADR_LEN];
  char local_ip[MAX_IPADR_LEN];
  int remote_port;
  int local_port;
};

struct proxy_info {
  struct hostname host;
  int port;
  unsigned char proxytype; /* curl_proxytype: what kind of proxy that is in
                              use */
  char *user;    /* proxy username string, allocated */
  char *passwd;  /* proxy password string, allocated */
};

struct ldapconninfo;

#define TRNSPRT_TCP 3
#define TRNSPRT_UDP 4
#define TRNSPRT_QUIC 5
#define TRNSPRT_UNIX 6

/*
 * The connectdata struct contains all fields and variables that should be
 * unique for an entire connection.
 */
struct connectdata {
  struct Curl_llist_node cpool_node; /* conncache lists */

  curl_closesocket_callback fclosesocket; /* function closing the socket(s) */
  void *closesocket_client;

  /* This is used by the connection pool logic. If this returns TRUE, this
     handle is still used by one or more easy handles and can only used by any
     other easy handle without careful consideration (== only for
     multiplexing) and it cannot be used by another multi handle! */
#define CONN_INUSE(c) Curl_llist_count(&(c)->easyq)

  /**** Fields set when inited and not modified again */
  curl_off_t connection_id; /* Contains a unique number to make it easier to
                               track the connections in the log output */
  char *destination; /* string carrying normalized hostname+port+scope */
  size_t destination_len; /* strlen(destination) + 1 */

  /* 'dns_entry' is the particular host we use. This points to an entry in the
     DNS cache and it will not get pruned while locked. It gets unlocked in
     multi_done(). This entry will be NULL if the connection is reused as then
     there is no name resolve done. */
  struct Curl_dns_entry *dns_entry;

  /* 'remote_addr' is the particular IP we connected to. it is owned, set
   * and NULLed by the connected socket filter (if there is one). */
  const struct Curl_sockaddr_ex *remote_addr;

  struct hostname host;
  char *hostname_resolve; /* hostname to resolve to address, allocated */
  char *secondaryhostname; /* secondary socket hostname (ftp) */
  struct hostname conn_to_host; /* the host to connect to. valid only if
                                   bits.conn_to_host is set */
#ifndef CURL_DISABLE_PROXY
  struct proxy_info socks_proxy;
  struct proxy_info http_proxy;
#endif
  /* 'primary' and 'secondary' get filled with IP quadruple
     (local/remote numerical ip address and port) whenever a connect is
     *attempted*.
     When more than one address is tried for a connection these will hold data
     for the last attempt. When the connection is actually established
     these are updated with data which comes directly from the socket. */
  struct ip_quadruple primary;
  struct ip_quadruple secondary;
  char *user;    /* username string, allocated */
  char *passwd;  /* password string, allocated */
  char *options; /* options string, allocated */
  char *sasl_authzid;     /* authorization identity string, allocated */
  char *oauth_bearer; /* OAUTH2 bearer, allocated */
  struct curltime now;     /* "current" time */
  struct curltime created; /* creation time */
  struct curltime lastused; /* when returned to the connection poolas idle */
  curl_socket_t sock[2]; /* two sockets, the second is used for the data
                            transfer when doing FTP */
  Curl_recv *recv[2];
  Curl_send *send[2];
  struct Curl_cfilter *cfilter[2]; /* connection filters */
  struct {
    struct curltime start[2]; /* when filter shutdown started */
    unsigned int timeout_ms; /* 0 means no timeout */
  } shutdown;
  /* Last pollset used in connection shutdown. Used to detect changes
   * for multi_socket API. */
  struct easy_pollset shutdown_poll;

  struct ssl_primary_config ssl_config;
#ifndef CURL_DISABLE_PROXY
  struct ssl_primary_config proxy_ssl_config;
#endif
  struct ConnectBits bits;    /* various state-flags for this connection */

  const struct Curl_handler *handler; /* Connection's protocol handler */
  const struct Curl_handler *given;   /* The protocol first given */

  /* Protocols can use a custom keepalive mechanism to keep connections alive.
     This allows those protocols to track the last time the keepalive mechanism
     was used on this connection. */
  struct curltime keepalive;

  /**** curl_get() phase fields */

  curl_socket_t sockfd;   /* socket to read from or CURL_SOCKET_BAD */
  curl_socket_t writesockfd; /* socket to write to, it may be the same we read
                                from. CURL_SOCKET_BAD disables */

#ifdef HAVE_GSSAPI
  BIT(sec_complete); /* if Kerberos is enabled for this connection */
  unsigned char command_prot; /* enum protection_level */
  unsigned char data_prot; /* enum protection_level */
  unsigned char request_data_prot; /* enum protection_level */
  size_t buffer_size;
  struct krb5buffer in_buffer;
  void *app_data;
  const struct Curl_sec_client_mech *mech;
  struct sockaddr_in local_addr;
#endif

#if defined(USE_KERBEROS5)    /* Consider moving some of the above GSS-API */
  struct kerberos5data krb5;  /* variables into the structure definition, */
#endif                        /* however, some of them are ftp specific. */

  struct Curl_llist easyq;    /* List of easy handles using this connection */

  /*************** Request - specific items ************/
#if defined(USE_WINDOWS_SSPI) && defined(SECPKG_ATTR_ENDPOINT_BINDINGS)
  CtxtHandle *sslContext;
#endif

#ifdef USE_GSASL
  struct gsasldata gsasl;
#endif

#if defined(USE_NTLM)
  curlntlm http_ntlm_state;
  curlntlm proxy_ntlm_state;

  struct ntlmdata ntlm;     /* NTLM differs from other authentication schemes
                               because it authenticates connections, not
                               single requests! */
  struct ntlmdata proxyntlm; /* NTLM data for proxy */
#endif

#ifdef USE_SPNEGO
  curlnegotiate http_negotiate_state;
  curlnegotiate proxy_negotiate_state;

  struct negotiatedata negotiate; /* state data for host Negotiate auth */
  struct negotiatedata proxyneg; /* state data for proxy Negotiate auth */
#endif

  union {
#ifndef CURL_DISABLE_FTP
    struct ftp_conn ftpc;
#endif
#ifdef USE_SSH
    struct ssh_conn sshc;
#endif
#ifndef CURL_DISABLE_TFTP
    struct tftp_state_data *tftpc;
#endif
#ifndef CURL_DISABLE_IMAP
    struct imap_conn imapc;
#endif
#ifndef CURL_DISABLE_POP3
    struct pop3_conn pop3c;
#endif
#ifndef CURL_DISABLE_SMTP
    struct smtp_conn smtpc;
#endif
#ifndef CURL_DISABLE_RTSP
    struct rtsp_conn rtspc;
#endif
#ifndef CURL_DISABLE_SMB
    struct smb_conn smbc;
#endif
#ifdef USE_LIBRTMP
    void *rtmp;
#endif
#ifdef USE_OPENLDAP
    struct ldapconninfo *ldapc;
#endif
#ifndef CURL_DISABLE_MQTT
    struct mqtt_conn mqtt;
#endif
#ifndef CURL_DISABLE_WEBSOCKETS
    struct websocket *ws;
#endif
    unsigned int unused:1; /* avoids empty union */
  } proto;

#ifdef USE_UNIX_SOCKETS
  char *unix_domain_socket;
#endif
#ifdef USE_HYPER
  /* if set, an alternative data transfer function */
  Curl_datastream datastream;
#endif
  /* When this connection is created, store the conditions for the local end
     bind. This is stored before the actual bind and before any connection is
     made and will serve the purpose of being used for comparison reasons so
     that subsequent bound-requested connections are not accidentally reusing
     wrong connections. */
  char *localdev;
  unsigned short localportrange;
  int waitfor;      /* current READ/WRITE bits to wait for */
#if defined(HAVE_GSSAPI) || defined(USE_WINDOWS_SSPI)
  int socks5_gssapi_enctype;
#endif
  /* The field below gets set in connect.c:connecthost() */
  int remote_port; /* the remote port, not the proxy port! */
  int conn_to_port; /* the remote port to connect to. valid only if
                       bits.conn_to_port is set */
#ifdef USE_IPV6
  unsigned int scope_id;  /* Scope id for IPv6 */
#endif
  unsigned short localport;
  unsigned short secondary_port; /* secondary socket remote port to connect to
                                    (ftp) */
  unsigned char alpn; /* APLN TLS negotiated protocol, a CURL_HTTP_VERSION*
                         value */
#ifndef CURL_DISABLE_PROXY
  unsigned char proxy_alpn; /* APLN of proxy tunnel, CURL_HTTP_VERSION* */
#endif
  unsigned char transport; /* one of the TRNSPRT_* defines */
  unsigned char ip_version; /* copied from the Curl_easy at creation time */
  unsigned char httpversion; /* the HTTP version*10 reported by the server */
  unsigned char connect_only;
  unsigned char gssapi_delegation; /* inherited from set.gssapi_delegation */
};

#ifndef CURL_DISABLE_PROXY
#define CURL_CONN_HOST_DISPNAME(c) \
          ((c)->bits.socksproxy ? (c)->socks_proxy.host.dispname : \
            (c)->bits.httpproxy ? (c)->http_proxy.host.dispname : \
              (c)->bits.conn_to_host ? (c)->conn_to_host.dispname : \
                (c)->host.dispname)
#else
#define CURL_CONN_HOST_DISPNAME(c) \
          (c)->bits.conn_to_host ? (c)->conn_to_host.dispname : \
            (c)->host.dispname
#endif

/* The end of connectdata. */

/*
 * Struct to keep statistical and informational data.
 * All variables in this struct must be initialized/reset in Curl_initinfo().
 */
struct PureInfo {
  int httpcode;  /* Recent HTTP, FTP, RTSP or SMTP response code */
  int httpproxycode; /* response code from proxy when received separate */
  int httpversion; /* the http version number X.Y = X*10+Y */
  time_t filetime; /* If requested, this is might get set. Set to -1 if the
                      time was unretrievable. */
  curl_off_t request_size; /* the amount of bytes sent in the request(s) */
  unsigned long proxyauthavail; /* what proxy auth types were announced */
  unsigned long httpauthavail;  /* what host auth types were announced */
  long numconnects; /* how many new connection did libcurl created */
  char *contenttype; /* the content type of the object */
  char *wouldredirect; /* URL this would have been redirected to if asked to */
  curl_off_t retry_after; /* info from Retry-After: header */
  unsigned int header_size;  /* size of read header(s) in bytes */

  /* PureInfo primary ip_quadruple is copied over from the connectdata
     struct in order to allow curl_easy_getinfo() to return this information
     even when the session handle is no longer associated with a connection,
     and also allow curl_easy_reset() to clear this information from the
     session handle without disturbing information which is still alive, and
     that might be reused, in the connection pool. */
  struct ip_quadruple primary;
  int conn_remote_port;  /* this is the "remote port", which is the port
                            number of the used URL, independent of proxy or
                            not */
  const char *conn_scheme;
  unsigned int conn_protocol;
  struct curl_certinfo certs; /* info about the certs. Asked for with
                                 CURLOPT_CERTINFO / CURLINFO_CERTINFO */
  CURLproxycode pxcode;
  BIT(timecond);  /* set to TRUE if the time condition did not match, which
                     thus made the document NOT get fetched */
  BIT(used_proxy); /* the transfer used a proxy */
};

struct pgrs_measure {
  struct curltime start; /* when measure started */
  curl_off_t start_size; /* the 'cur_size' the measure started at */
};

struct pgrs_dir {
  curl_off_t total_size; /* total expected bytes */
  curl_off_t cur_size; /* transferred bytes so far */
  curl_off_t speed; /* bytes per second transferred */
  struct pgrs_measure limit;
};

struct Progress {
  time_t lastshow; /* time() of the last displayed progress meter or NULL to
                      force redraw at next call */
  struct pgrs_dir ul;
  struct pgrs_dir dl;

  curl_off_t current_speed; /* uses the currently fastest transfer */
  curl_off_t earlydata_sent;

  int width; /* screen width at download start */
  int flags; /* see progress.h */

  timediff_t timespent;

  timediff_t t_postqueue;
  timediff_t t_nslookup;
  timediff_t t_connect;
  timediff_t t_appconnect;
  timediff_t t_pretransfer;
  timediff_t t_posttransfer;
  timediff_t t_starttransfer;
  timediff_t t_redirect;

  struct curltime start;
  struct curltime t_startsingle;
  struct curltime t_startop;
  struct curltime t_acceptdata;

#define CURR_TIME (5 + 1) /* 6 entries for 5 seconds */

  curl_off_t speeder[ CURR_TIME ];
  struct curltime speeder_time[ CURR_TIME ];
  int speeder_c;
  BIT(callback);  /* set when progress callback is used */
  BIT(is_t_startransfer_set);
};

typedef enum {
    RTSPREQ_NONE, /* first in list */
    RTSPREQ_OPTIONS,
    RTSPREQ_DESCRIBE,
    RTSPREQ_ANNOUNCE,
    RTSPREQ_SETUP,
    RTSPREQ_PLAY,
    RTSPREQ_PAUSE,
    RTSPREQ_TEARDOWN,
    RTSPREQ_GET_PARAMETER,
    RTSPREQ_SET_PARAMETER,
    RTSPREQ_RECORD,
    RTSPREQ_RECEIVE,
    RTSPREQ_LAST /* last in list */
} Curl_RtspReq;

struct auth {
  unsigned long want;  /* Bitmask set to the authentication methods wanted by
                          app (with CURLOPT_HTTPAUTH or CURLOPT_PROXYAUTH). */
  unsigned long picked;
  unsigned long avail; /* Bitmask for what the server reports to support for
                          this resource */
  BIT(done);  /* TRUE when the auth phase is done and ready to do the
                 actual request */
  BIT(multipass); /* TRUE if this is not yet authenticated but within the
                     auth multipass negotiation */
  BIT(iestyle); /* TRUE if digest should be done IE-style or FALSE if it
                   should be RFC compliant */
};

#ifdef USE_NGHTTP2
struct Curl_data_prio_node {
  struct Curl_data_prio_node *next;
  struct Curl_easy *data;
};
#endif

/**
 * Priority information for an easy handle in relation to others
 * on the same connection.
 * TODO: we need to adapt it to the new priority scheme as defined in RFC 9218
 */
struct Curl_data_priority {
#ifdef USE_NGHTTP2
  /* tree like dependencies only implemented in nghttp2 */
  struct Curl_easy *parent;
  struct Curl_data_prio_node *children;
#endif
  int weight;
#ifdef USE_NGHTTP2
  BIT(exclusive);
#endif
};

/* Timers */
typedef enum {
  EXPIRE_100_TIMEOUT,
  EXPIRE_ASYNC_NAME,
  EXPIRE_CONNECTTIMEOUT,
  EXPIRE_DNS_PER_NAME, /* family1 */
  EXPIRE_DNS_PER_NAME2, /* family2 */
  EXPIRE_HAPPY_EYEBALLS_DNS, /* See asyn-ares.c */
  EXPIRE_HAPPY_EYEBALLS,
  EXPIRE_MULTI_PENDING,
  EXPIRE_RUN_NOW,
  EXPIRE_SPEEDCHECK,
  EXPIRE_TIMEOUT,
  EXPIRE_TOOFAST,
  EXPIRE_QUIC,
  EXPIRE_FTP_ACCEPT,
  EXPIRE_ALPN_EYEBALLS,
  EXPIRE_LAST /* not an actual timer, used as a marker only */
} expire_id;


typedef enum {
  TRAILERS_NONE,
  TRAILERS_INITIALIZED,
  TRAILERS_SENDING,
  TRAILERS_DONE
} trailers_state;


/*
 * One instance for each timeout an easy handle can set.
 */
struct time_node {
  struct Curl_llist_node list;
  struct curltime time;
  expire_id eid;
};

/* individual pieces of the URL */
struct urlpieces {
  char *scheme;
  char *hostname;
  char *port;
  char *user;
  char *password;
  char *options;
  char *path;
  char *query;
};

#define CREDS_NONE   0
#define CREDS_URL    1 /* from URL */
#define CREDS_OPTION 2 /* set with a CURLOPT_ */
#define CREDS_NETRC  3 /* found in netrc */

struct UrlState {
  /* buffers to store authentication data in, as parsed from input options */
  struct curltime keeps_speed; /* for the progress meter really */

  curl_off_t lastconnect_id; /* The last connection, -1 if undefined */
  curl_off_t recent_conn_id; /* The most recent connection used, might no
                              * longer exist */
  struct dynbuf headerb; /* buffer to store headers in */
  struct curl_slist *hstslist; /* list of HSTS files set by
                                  curl_easy_setopt(HSTS) calls */
  curl_off_t current_speed;  /* the ProgressShow() function sets this,
                                bytes / second */

  /* hostname, port number and protocol of the first (not followed) request.
     if set, this should be the hostname that we will sent authorization to,
     no else. Used to make Location: following not keep sending user+password.
     This is strdup()ed data. */
  char *first_host;
  int first_remote_port;
  curl_prot_t first_remote_protocol;

  int retrycount; /* number of retries on a new connection */
  struct Curl_ssl_session *session; /* array of 'max_ssl_sessions' size */
  long sessionage;                  /* number of the most recent session */
  int os_errno;  /* filled in with errno whenever an error occurs */
  long followlocation; /* redirect counter */
  int requests; /* request counter: redirects + authentication retakes */
#ifdef HAVE_SIGNAL
  /* storage for the previous bag^H^H^HSIGPIPE signal handler :-) */
  void (*prev_signal)(int sig);
#endif
#ifndef CURL_DISABLE_DIGEST_AUTH
  struct digestdata digest;      /* state data for host Digest auth */
  struct digestdata proxydigest; /* state data for proxy Digest auth */
#endif
  struct auth authhost;  /* auth details for host */
  struct auth authproxy; /* auth details for proxy */
#ifdef USE_CURL_ASYNC
  struct Curl_async async;  /* asynchronous name resolver data */
#endif

#if defined(USE_OPENSSL)
  /* void instead of ENGINE to avoid bleeding OpenSSL into this header */
  void *engine;
#endif /* USE_OPENSSL */
  struct curltime expiretime; /* set this with Curl_expire() only */
  struct Curl_tree timenode; /* for the splay stuff */
  struct Curl_llist timeoutlist; /* list of pending timeouts */
  struct time_node expires[EXPIRE_LAST]; /* nodes for each expire type */

  /* a place to store the most recently set (S)FTP entrypath */
  char *most_recent_ftp_entrypath;
  char *range; /* range, if used. See README for detailed specification on
                  this syntax. */
  curl_off_t resume_from; /* continue [ftp] transfer from here */

#ifndef CURL_DISABLE_RTSP
  /* This RTSP state information survives requests and connections */
  long rtsp_next_client_CSeq; /* the session's next client CSeq */
  long rtsp_next_server_CSeq; /* the session's next server CSeq */
  long rtsp_CSeq_recv; /* most recent CSeq received */

  unsigned char rtp_channel_mask[32]; /* for the correctness checking of the
                                         interleaved data */
#endif

  curl_off_t infilesize; /* size of file to upload, -1 means unknown.
                            Copied from set.filesize at start of operation */
#if defined(USE_HTTP2) || defined(USE_HTTP3)
  struct Curl_data_priority priority; /* shallow copy of data->set */
#endif

  curl_read_callback fread_func; /* read callback/function */
  void *in;                      /* CURLOPT_READDATA */
  CURLU *uh; /* URL handle for the current parsed URL */
  struct urlpieces up;
  char *url;        /* work URL, copied from UserDefined */
  char *referer;    /* referer string */
  struct curl_slist *resolve; /* set to point to the set.resolve list when
                                 this should be dealt with in pretransfer */
#ifndef CURL_DISABLE_HTTP
  curl_mimepart *mimepost;
#ifndef CURL_DISABLE_FORM_API
  curl_mimepart *formp; /* storage for old API form-posting, allocated on
                           demand */
#endif
  size_t trailers_bytes_sent;
  struct dynbuf trailers_buf; /* a buffer containing the compiled trailing
                                 headers */
  struct Curl_llist httphdrs; /* received headers */
  struct curl_header headerout[2]; /* for external purposes */
  struct Curl_header_store *prevhead; /* the latest added header */
  trailers_state trailers_state; /* whether we are sending trailers
                                    and what stage are we at */
#endif
#ifndef CURL_DISABLE_COOKIES
  struct curl_slist *cookielist; /* list of cookie files set by
                                    curl_easy_setopt(COOKIEFILE) calls */
#endif
#ifdef USE_HYPER
  bool hconnect;  /* set if a CONNECT request */
  CURLcode hresult; /* used to pass return codes back from hyper callbacks */
#endif

#ifndef CURL_DISABLE_VERBOSE_STRINGS
  struct curl_trc_feat *feat; /* opt. trace feature transfer is part of */
#endif

#ifndef CURL_DISABLE_NETRC
  struct store_netrc netrc;
#endif

  /* Dynamically allocated strings, MUST be freed before this struct is
     killed. */
  struct dynamically_allocated_data {
    char *uagent;
    char *accept_encoding;
    char *userpwd;
    char *rangeline;
    char *ref;
    char *host;
#ifndef CURL_DISABLE_COOKIES
    char *cookiehost;
#endif
#ifndef CURL_DISABLE_RTSP
    char *rtsp_transport;
#endif
    char *te; /* TE: request header */

    /* transfer credentials */
    char *user;
    char *passwd;
#ifndef CURL_DISABLE_PROXY
    char *proxyuserpwd;
    char *proxyuser;
    char *proxypasswd;
#endif
  } aptr;
  unsigned char httpwant; /* when non-zero, a specific HTTP version requested
                             to be used in the library's request(s) */
  unsigned char httpversion; /* the lowest HTTP version*10 reported by any
                                server involved in this request */
  unsigned char httpreq; /* Curl_HttpReq; what kind of HTTP request (if any)
                            is this */
  unsigned char select_bits; /* != 0 -> bitmask of socket events for this
                                 transfer overriding anything the socket may
                                 report */
  unsigned int creds_from:2; /* where is the server credentials originating
                                from, see the CREDS_* defines above */

  /* when curl_easy_perform() is called, the multi handle is "owned" by
     the easy handle so curl_easy_cleanup() on such an easy handle will
     also close the multi handle! */
  BIT(multi_owned_by_easy);

  BIT(this_is_a_follow); /* this is a followed Location: request */
  BIT(refused_stream); /* this was refused, try again */
  BIT(errorbuf); /* Set to TRUE if the error buffer is already filled in.
                    This must be set to FALSE every time _easy_perform() is
                    called. */
  BIT(allow_port); /* Is set.use_port allowed to take effect or not. This
                      is always set TRUE when curl_easy_perform() is called. */
  BIT(authproblem); /* TRUE if there is some problem authenticating */
  /* set after initial USER failure, to prevent an authentication loop */
  BIT(wildcardmatch); /* enable wildcard matching */
  BIT(disableexpect);    /* TRUE if Expect: is disabled due to a previous
                            417 response */
  BIT(use_range);
  BIT(rangestringalloc); /* the range string is malloc()'ed */
  BIT(done); /* set to FALSE when Curl_init_do() is called and set to TRUE
                when multi_done() is called, to prevent multi_done() to get
                invoked twice when the multi interface is used. */
#ifndef CURL_DISABLE_COOKIES
  BIT(cookie_engine);
#endif
  BIT(prefer_ascii);   /* ASCII rather than binary */
#ifdef CURL_LIST_ONLY_PROTOCOL
  BIT(list_only);      /* list directory contents */
#endif
  BIT(url_alloc);   /* URL string is malloc()'ed */
  BIT(referer_alloc); /* referer string is malloc()ed */
  BIT(wildcard_resolve); /* Set to true if any resolve change is a wildcard */
  BIT(upload);         /* upload request */
  BIT(internal); /* internal: true if this easy handle was created for
                    internal use and the user does not have ownership of the
                    handle. */
};

/*
 * This 'UserDefined' struct must only contain data that is set once to go
 * for many (perhaps) independent connections. Values that are generated or
 * calculated internally for the "session handle" MUST be defined within the
 * 'struct UrlState' instead. The only exceptions MUST note the changes in
 * the 'DynamicStatic' struct.
 * Character pointer fields point to dynamic storage, unless otherwise stated.
 */

struct Curl_multi;    /* declared in multihandle.c */

enum dupstring {
  STRING_CERT,            /* client certificate filename */
  STRING_CERT_TYPE,       /* format for certificate (default: PEM)*/
  STRING_KEY,             /* private key filename */
  STRING_KEY_PASSWD,      /* plain text private key password */
  STRING_KEY_TYPE,        /* format for private key (default: PEM) */
  STRING_SSL_CAPATH,      /* CA directory name (does not work on Windows) */
  STRING_SSL_CAFILE,      /* certificate file to verify peer against */
  STRING_SSL_PINNEDPUBLICKEY, /* public key file to verify peer against */
  STRING_SSL_CIPHER_LIST, /* list of ciphers to use */
  STRING_SSL_CIPHER13_LIST, /* list of TLS 1.3 ciphers to use */
  STRING_SSL_CRLFILE,     /* crl file to check certificate */
  STRING_SSL_ISSUERCERT, /* issuer cert file to check certificate */
  STRING_SERVICE_NAME,    /* Service name */
#ifndef CURL_DISABLE_PROXY
  STRING_CERT_PROXY,      /* client certificate filename */
  STRING_CERT_TYPE_PROXY, /* format for certificate (default: PEM)*/
  STRING_KEY_PROXY,       /* private key filename */
  STRING_KEY_PASSWD_PROXY, /* plain text private key password */
  STRING_KEY_TYPE_PROXY,  /* format for private key (default: PEM) */
  STRING_SSL_CAPATH_PROXY, /* CA directory name (does not work on Windows) */
  STRING_SSL_CAFILE_PROXY, /* certificate file to verify peer against */
  STRING_SSL_PINNEDPUBLICKEY_PROXY, /* public key file to verify proxy */
  STRING_SSL_CIPHER_LIST_PROXY, /* list of ciphers to use */
  STRING_SSL_CIPHER13_LIST_PROXY, /* list of TLS 1.3 ciphers to use */
  STRING_SSL_CRLFILE_PROXY, /* crl file to check certificate */
  STRING_SSL_ISSUERCERT_PROXY, /* issuer cert file to check certificate */
  STRING_PROXY_SERVICE_NAME, /* Proxy service name */
#endif
#ifndef CURL_DISABLE_COOKIES
  STRING_COOKIE,          /* HTTP cookie string to send */
  STRING_COOKIEJAR,       /* dump all cookies to this file */
#endif
  STRING_CUSTOMREQUEST,   /* HTTP/FTP/RTSP request/method to use */
  STRING_DEFAULT_PROTOCOL, /* Protocol to use when the URL does not specify */
  STRING_DEVICE,          /* local network interface/address to use */
  STRING_INTERFACE,       /* local network interface to use */
  STRING_BINDHOST,        /* local address to use */
  STRING_ENCODING,        /* Accept-Encoding string */
#ifndef CURL_DISABLE_FTP
  STRING_FTP_ACCOUNT,     /* ftp account data */
  STRING_FTP_ALTERNATIVE_TO_USER, /* command to send if USER/PASS fails */
  STRING_FTPPORT,         /* port to send with the FTP PORT command */
#endif
#if defined(HAVE_GSSAPI)
  STRING_KRB_LEVEL,       /* krb security level */
#endif
#ifndef CURL_DISABLE_NETRC
  STRING_NETRC_FILE,      /* if not NULL, use this instead of trying to find
                             $HOME/.netrc */
#endif
#ifndef CURL_DISABLE_PROXY
  STRING_PROXY,           /* proxy to use */
  STRING_PRE_PROXY,       /* pre socks proxy to use */
#endif
  STRING_SET_RANGE,       /* range, if used */
  STRING_SET_REFERER,     /* custom string for the HTTP referer field */
  STRING_SET_URL,         /* what original URL to work on */
  STRING_USERAGENT,       /* User-Agent string */
  STRING_SSL_ENGINE,      /* name of ssl engine */
  STRING_USERNAME,        /* <username>, if used */
  STRING_PASSWORD,        /* <password>, if used */
  STRING_OPTIONS,         /* <options>, if used */
#ifndef CURL_DISABLE_PROXY
  STRING_PROXYUSERNAME,   /* Proxy <username>, if used */
  STRING_PROXYPASSWORD,   /* Proxy <password>, if used */
  STRING_NOPROXY,         /* List of hosts which should not use the proxy, if
                             used */
#endif
#ifndef CURL_DISABLE_RTSP
  STRING_RTSP_SESSION_ID, /* Session ID to use */
  STRING_RTSP_STREAM_URI, /* Stream URI for this request */
  STRING_RTSP_TRANSPORT,  /* Transport for this session */
#endif
#ifdef USE_SSH
  STRING_SSH_PRIVATE_KEY, /* path to the private key file for auth */
  STRING_SSH_PUBLIC_KEY,  /* path to the public key file for auth */
  STRING_SSH_HOST_PUBLIC_KEY_MD5, /* md5 of host public key in ASCII hex */
  STRING_SSH_HOST_PUBLIC_KEY_SHA256, /* sha256 of host public key in base64 */
  STRING_SSH_KNOWNHOSTS,  /* filename of knownhosts file */
#endif
#ifndef CURL_DISABLE_SMTP
  STRING_MAIL_FROM,
  STRING_MAIL_AUTH,
#endif
#ifdef USE_TLS_SRP
  STRING_TLSAUTH_USERNAME,  /* TLS auth <username> */
  STRING_TLSAUTH_PASSWORD,  /* TLS auth <password> */
#ifndef CURL_DISABLE_PROXY
  STRING_TLSAUTH_USERNAME_PROXY, /* TLS auth <username> */
  STRING_TLSAUTH_PASSWORD_PROXY, /* TLS auth <password> */
#endif
#endif
  STRING_BEARER,                /* <bearer>, if used */
#ifdef USE_UNIX_SOCKETS
  STRING_UNIX_SOCKET_PATH,      /* path to Unix socket, if used */
#endif
  STRING_TARGET,                /* CURLOPT_REQUEST_TARGET */
#ifndef CURL_DISABLE_DOH
  STRING_DOH,                   /* CURLOPT_DOH_URL */
#endif
#ifndef CURL_DISABLE_ALTSVC
  STRING_ALTSVC,                /* CURLOPT_ALTSVC */
#endif
#ifndef CURL_DISABLE_HSTS
  STRING_HSTS,                  /* CURLOPT_HSTS */
#endif
  STRING_SASL_AUTHZID,          /* CURLOPT_SASL_AUTHZID */
#ifdef USE_ARES
  STRING_DNS_SERVERS,
  STRING_DNS_INTERFACE,
  STRING_DNS_LOCAL_IP4,
  STRING_DNS_LOCAL_IP6,
#endif
  STRING_SSL_EC_CURVES,
#ifndef CURL_DISABLE_AWS
  STRING_AWS_SIGV4, /* Parameters for V4 signature */
#endif
#ifndef CURL_DISABLE_PROXY
  STRING_HAPROXY_CLIENT_IP,     /* CURLOPT_HAPROXY_CLIENT_IP */
#endif
  STRING_ECH_CONFIG,            /* CURLOPT_ECH_CONFIG */
  STRING_ECH_PUBLIC,            /* CURLOPT_ECH_PUBLIC */

  /* -- end of null-terminated strings -- */

  STRING_LASTZEROTERMINATED,

  /* -- below this are pointers to binary data that cannot be strdup'ed. --- */

  STRING_COPYPOSTFIELDS,  /* if POST, set the fields' values here */

  STRING_LAST /* not used, just an end-of-list marker */
};

enum dupblob {
  BLOB_CERT,
  BLOB_KEY,
  BLOB_SSL_ISSUERCERT,
  BLOB_CAINFO,
#ifndef CURL_DISABLE_PROXY
  BLOB_CERT_PROXY,
  BLOB_KEY_PROXY,
  BLOB_SSL_ISSUERCERT_PROXY,
  BLOB_CAINFO_PROXY,
#endif
  BLOB_LAST
};

/* callback that gets called when this easy handle is completed within a multi
   handle. Only used for internally created transfers, like for example
   DoH. */
typedef int (*multidone_func)(struct Curl_easy *easy, CURLcode result);

struct UserDefined {
  FILE *err;         /* the stderr user data goes here */
  void *debugdata;   /* the data that will be passed to fdebug */
  char *errorbuffer; /* (Static) store failure messages in here */
  void *out;         /* CURLOPT_WRITEDATA */
  void *in_set;      /* CURLOPT_READDATA */
  void *writeheader; /* write the header to this if non-NULL */
  unsigned short use_port; /* which port to use (when not using default) */
  unsigned long httpauth;  /* kind of HTTP authentication to use (bitmask) */
  unsigned long proxyauth; /* kind of proxy authentication to use (bitmask) */
  long maxredirs;    /* maximum no. of http(s) redirects to follow, set to -1
                        for infinity */

  void *postfields;  /* if POST, set the fields' values here */
  curl_seek_callback seek_func;      /* function that seeks the input */
  curl_off_t postfieldsize; /* if POST, this might have a size to use instead
                               of strlen(), and then the data *may* be binary
                               (contain zero bytes) */
#ifndef CURL_DISABLE_BINDLOCAL
  unsigned short localport; /* local port number to bind to */
  unsigned short localportrange; /* number of additional port numbers to test
                                    in case the 'localport' one cannot be
                                    bind()ed */
#endif
  curl_write_callback fwrite_func;   /* function that stores the output */
  curl_write_callback fwrite_header; /* function that stores headers */
  curl_write_callback fwrite_rtp;    /* function that stores interleaved RTP */
  curl_read_callback fread_func_set; /* function that reads the input */
  curl_progress_callback fprogress; /* OLD and deprecated progress callback  */
  curl_xferinfo_callback fxferinfo; /* progress callback */
  curl_debug_callback fdebug;      /* function that write informational data */
  curl_ioctl_callback ioctl_func;  /* function for I/O control */
  curl_sockopt_callback fsockopt;  /* function for setting socket options */
  void *sockopt_client; /* pointer to pass to the socket options callback */
  curl_opensocket_callback fopensocket; /* function for checking/translating
                                           the address and opening the
                                           socket */
  void *opensocket_client;
  curl_closesocket_callback fclosesocket; /* function for closing the
                                             socket */
  void *closesocket_client;
  curl_prereq_callback fprereq; /* pre-initial request callback */
  void *prereq_userp; /* pre-initial request user data */

  void *seek_client;    /* pointer to pass to the seek callback */
#ifndef CURL_DISABLE_HSTS
  curl_hstsread_callback hsts_read;
  void *hsts_read_userp;
  curl_hstswrite_callback hsts_write;
  void *hsts_write_userp;
#endif
  void *progress_client; /* pointer to pass to the progress callback */
  void *ioctl_client;   /* pointer to pass to the ioctl callback */
  unsigned int timeout;        /* ms, 0 means no timeout */
  unsigned int connecttimeout; /* ms, 0 means default timeout */
  unsigned int happy_eyeballs_timeout; /* ms, 0 is a valid value */
  unsigned int server_response_timeout; /* ms, 0 means no timeout */
  unsigned int shutdowntimeout; /* ms, 0 means default timeout */
  long maxage_conn;     /* in seconds, max idle time to allow a connection that
                           is to be reused */
  long maxlifetime_conn; /* in seconds, max time since creation to allow a
                            connection that is to be reused */
#ifndef CURL_DISABLE_TFTP
  long tftp_blksize;    /* in bytes, 0 means use default */
#endif
  curl_off_t filesize;  /* size of file to upload, -1 means unknown */
  long low_speed_limit; /* bytes/second */
  long low_speed_time;  /* number of seconds */
  curl_off_t max_send_speed; /* high speed limit in bytes/second for upload */
  curl_off_t max_recv_speed; /* high speed limit in bytes/second for
                                download */
  curl_off_t set_resume_from;  /* continue [ftp] transfer from here */
  struct curl_slist *headers; /* linked list of extra headers */
  struct curl_httppost *httppost;  /* linked list of old POST data */
#if !defined(CURL_DISABLE_MIME) || !defined(CURL_DISABLE_FORM_API)
  curl_mimepart mimepost;  /* MIME/POST data. */
#endif
#ifndef CURL_DISABLE_TELNET
  struct curl_slist *telnet_options; /* linked list of telnet options */
#endif
  struct curl_slist *resolve;     /* list of names to add/remove from
                                     DNS cache */
  struct curl_slist *connect_to; /* list of host:port mappings to override
                                    the hostname and port to connect to */
  time_t timevalue;       /* what time to compare with */
  unsigned char timecondition; /* kind of time comparison: curl_TimeCond */
  unsigned char method;   /* what kind of HTTP request: Curl_HttpReq */
  unsigned char httpwant; /* when non-zero, a specific HTTP version requested
                             to be used in the library's request(s) */
  struct ssl_config_data ssl;  /* user defined SSL stuff */
#ifndef CURL_DISABLE_PROXY
  struct ssl_config_data proxy_ssl;  /* user defined SSL stuff for proxy */
  struct curl_slist *proxyheaders; /* linked list of extra CONNECT headers */
  unsigned short proxyport; /* If non-zero, use this port number by
                               default. If the proxy string features a
                               ":[port]" that one will override this. */
  unsigned char proxytype; /* what kind of proxy: curl_proxytype */
  unsigned char socks5auth;/* kind of SOCKS5 authentication to use (bitmask) */
#endif
  struct ssl_general_config general_ssl; /* general user defined SSL stuff */
  int dns_cache_timeout; /* DNS cache timeout (seconds) */
  unsigned int buffer_size;      /* size of receive buffer to use */
  unsigned int upload_buffer_size; /* size of upload buffer to use,
                                      keep it >= CURL_MAX_WRITE_SIZE */
  void *private_data; /* application-private data */
#ifndef CURL_DISABLE_HTTP
  struct curl_slist *http200aliases; /* linked list of aliases for http200 */
#endif
  unsigned char ipver; /* the CURL_IPRESOLVE_* defines in the public header
                          file 0 - whatever, 1 - v2, 2 - v6 */
  curl_off_t max_filesize; /* Maximum file size to download */
#ifndef CURL_DISABLE_FTP
  unsigned char ftp_filemethod; /* how to get to a file: curl_ftpfile  */
  unsigned char ftpsslauth; /* what AUTH XXX to try: curl_ftpauth */
  unsigned char ftp_ccc;   /* FTP CCC options: curl_ftpccc */
  unsigned int accepttimeout;   /* in milliseconds, 0 means no timeout */
#endif
#if !defined(CURL_DISABLE_FTP) || defined(USE_SSH)
  struct curl_slist *quote;     /* after connection is established */
  struct curl_slist *postquote; /* after the transfer */
  struct curl_slist *prequote; /* before the transfer, after type */
  /* Despite the name, ftp_create_missing_dirs is for FTP(S) and SFTP
     1 - create directories that do not exist
     2 - the same but also allow MKD to fail once
  */
  unsigned char ftp_create_missing_dirs;
#endif
#ifdef USE_LIBSSH2
  curl_sshhostkeycallback ssh_hostkeyfunc; /* hostkey check callback */
  void *ssh_hostkeyfunc_userp;         /* custom pointer to callback */
#endif
#ifdef USE_SSH
  curl_sshkeycallback ssh_keyfunc; /* key matching callback */
  void *ssh_keyfunc_userp;         /* custom pointer to callback */
  int ssh_auth_types;    /* allowed SSH auth types */
  unsigned int new_directory_perms; /* when creating remote dirs */
#endif
#ifndef CURL_DISABLE_NETRC
  unsigned char use_netrc;        /* enum CURL_NETRC_OPTION values  */
#endif
  unsigned int new_file_perms;      /* when creating remote files */
  char *str[STRING_LAST]; /* array of strings, pointing to allocated memory */
  struct curl_blob *blobs[BLOB_LAST];
#ifdef USE_IPV6
  unsigned int scope_id;  /* Scope id for IPv6 */
#endif
  curl_prot_t allowed_protocols;
  curl_prot_t redir_protocols;
#ifndef CURL_DISABLE_RTSP
  void *rtp_out;     /* write RTP to this if non-NULL */
  /* Common RTSP header options */
  Curl_RtspReq rtspreq; /* RTSP request type */
#endif
#ifndef CURL_DISABLE_FTP
  curl_chunk_bgn_callback chunk_bgn; /* called before part of transfer
                                        starts */
  curl_chunk_end_callback chunk_end; /* called after part transferring
                                        stopped */
  curl_fnmatch_callback fnmatch; /* callback to decide which file corresponds
                                    to pattern (e.g. if WILDCARDMATCH is on) */
  void *fnmatch_data;
  void *wildcardptr;
#endif
 /* GSS-API credential delegation, see the documentation of
    CURLOPT_GSSAPI_DELEGATION */
  unsigned char gssapi_delegation;

  int tcp_keepidle;     /* seconds in idle before sending keepalive probe */
  int tcp_keepintvl;    /* seconds between TCP keepalive probes */
  int tcp_keepcnt;      /* maximum number of keepalive probes */

  long expect_100_timeout; /* in milliseconds */
#if defined(USE_HTTP2) || defined(USE_HTTP3)
  struct Curl_data_priority priority;
#endif
  curl_resolver_start_callback resolver_start; /* optional callback called
                                                  before resolver start */
  void *resolver_start_client; /* pointer to pass to resolver start callback */
  long upkeep_interval_ms;      /* Time between calls for connection upkeep. */
  multidone_func fmultidone;
#ifndef CURL_DISABLE_DOH
  curl_off_t dohfor_mid; /* this is a DoH request for that transfer */
#endif
  CURLU *uh; /* URL handle for the current parsed URL */
#ifndef CURL_DISABLE_HTTP
  void *trailer_data; /* pointer to pass to trailer data callback */
  curl_trailer_callback trailer_callback; /* trailing data callback */
#endif
  char keep_post;     /* keep POSTs as POSTs after a 30x request; each
                         bit represents a request, from 301 to 303 */
#ifndef CURL_DISABLE_SMTP
  struct curl_slist *mail_rcpt; /* linked list of mail recipients */
  BIT(mail_rcpt_allowfails); /* allow RCPT TO command to fail for some
                                recipients */
#endif
  unsigned int maxconnects; /* Max idle connections in the connection cache */
  unsigned char use_ssl;   /* if AUTH TLS is to be attempted etc, for FTP or
                              IMAP or POP3 or others! (type: curl_usessl)*/
  unsigned char connect_only; /* make connection/request, then let
                                 application use the socket */
#ifndef CURL_DISABLE_MIME
  BIT(mime_formescape);
#endif
  BIT(is_fread_set); /* has read callback been set to non-NULL? */
#ifndef CURL_DISABLE_TFTP
  BIT(tftp_no_options); /* do not send TFTP options requests */
#endif
  BIT(sep_headers);     /* handle host and proxy headers separately */
#ifndef CURL_DISABLE_COOKIES
  BIT(cookiesession);   /* new cookie session? */
#endif
  BIT(crlf);            /* convert crlf on ftp upload(?) */
#ifdef USE_SSH
  BIT(ssh_compression);            /* enable SSH compression */
#endif

/* Here follows boolean settings that define how to behave during
   this session. They are STATIC, set by libcurl users or at least initially
   and they do not change during operations. */
  BIT(quick_exit);       /* set 1L when it is okay to leak things (like
                            threads), as we are about to exit() anyway and
                            do not want lengthy cleanups to delay termination,
                            e.g. after a DNS timeout */
  BIT(get_filetime);     /* get the time and get of the remote file */
#ifndef CURL_DISABLE_PROXY
  BIT(tunnel_thru_httpproxy); /* use CONNECT through an HTTP proxy */
#endif
  BIT(prefer_ascii);     /* ASCII rather than binary */
  BIT(remote_append);    /* append, not overwrite, on upload */
#ifdef CURL_LIST_ONLY_PROTOCOL
  BIT(list_only);        /* list directory */
#endif
#ifndef CURL_DISABLE_FTP
  BIT(ftp_use_port);     /* use the FTP PORT command */
  BIT(ftp_use_epsv);     /* if EPSV is to be attempted or not */
  BIT(ftp_use_eprt);     /* if EPRT is to be attempted or not */
  BIT(ftp_use_pret);     /* if PRET is to be used before PASV or not */
  BIT(ftp_skip_ip);      /* skip the IP address the FTP server passes on to
                            us */
  BIT(wildcard_enabled); /* enable wildcard matching */
#endif
  BIT(hide_progress);    /* do not use the progress meter */
  BIT(http_fail_on_error);  /* fail on HTTP error codes >= 400 */
  BIT(http_keep_sending_on_error); /* for HTTP status codes >= 300 */
  BIT(http_follow_location); /* follow HTTP redirects */
  BIT(http_transfer_encoding); /* request compressed HTTP transfer-encoding */
  BIT(allow_auth_to_other_hosts);
  BIT(include_header); /* include received protocol headers in data output */
  BIT(http_set_referer); /* is a custom referer used */
  BIT(http_auto_referer); /* set "correct" referer when following
                             location: */
  BIT(opt_no_body);    /* as set with CURLOPT_NOBODY */
  BIT(verbose);        /* output verbosity */
#if defined(HAVE_GSSAPI)
  BIT(krb);            /* Kerberos connection requested */
#endif
  BIT(reuse_forbid);   /* forbidden to be reused, close after use */
  BIT(reuse_fresh);    /* do not reuse an existing connection  */
  BIT(no_signal);      /* do not use any signal/alarm handler */
  BIT(tcp_nodelay);    /* whether to enable TCP_NODELAY or not */
  BIT(ignorecl);       /* ignore content length */
  BIT(http_te_skip);   /* pass the raw body data to the user, even when
                          transfer-encoded (chunked, compressed) */
  BIT(http_ce_skip);   /* pass the raw body data to the user, even when
                          content-encoded (chunked, compressed) */
  BIT(proxy_transfer_mode); /* set transfer mode (;type=<a|i>) when doing
                               FTP via an HTTP proxy */
#if defined(HAVE_GSSAPI) || defined(USE_WINDOWS_SSPI)
  BIT(socks5_gssapi_nec); /* Flag to support NEC SOCKS5 server */
#endif
  BIT(sasl_ir);         /* Enable/disable SASL initial response */
  BIT(tcp_keepalive);  /* use TCP keepalives */
  BIT(tcp_fastopen);   /* use TCP Fast Open */
  BIT(ssl_enable_alpn);/* TLS ALPN extension? */
  BIT(path_as_is);     /* allow dotdots? */
  BIT(pipewait);       /* wait for multiplex status before starting a new
                          connection */
  BIT(suppress_connect_headers); /* suppress proxy CONNECT response headers
                                    from user callbacks */
  BIT(dns_shuffle_addresses); /* whether to shuffle addresses before use */
#ifndef CURL_DISABLE_PROXY
  BIT(haproxyprotocol); /* whether to send HAProxy PROXY protocol v1
                           header */
#endif
#ifdef USE_UNIX_SOCKETS
  BIT(abstract_unix_socket);
#endif
  BIT(disallow_username_in_url); /* disallow username in URL */
#ifndef CURL_DISABLE_DOH
  BIT(doh); /* DNS-over-HTTPS enabled */
  BIT(doh_verifypeer);     /* DoH certificate peer verification */
  BIT(doh_verifyhost);     /* DoH certificate hostname verification */
  BIT(doh_verifystatus);   /* DoH certificate status verification */
#endif
  BIT(http09_allowed); /* allow HTTP/0.9 responses */
#ifndef CURL_DISABLE_WEBSOCKETS
  BIT(ws_raw_mode);
#endif
#ifdef USE_ECH
  int tls_ech;      /* TLS ECH configuration  */
#endif
};

#ifndef CURL_DISABLE_MIME
#define IS_MIME_POST(a) ((a)->set.mimepost.kind != MIMEKIND_NONE)
#else
#define IS_MIME_POST(a) FALSE
#endif

struct Names {
  struct Curl_hash *hostcache;
  enum {
    HCACHE_NONE,    /* not pointing to anything */
    HCACHE_MULTI,   /* points to a shared one in the multi handle */
    HCACHE_SHARED   /* points to a shared one in a shared object */
  } hostcachetype;
};

/*
 * The 'connectdata' struct MUST have all the connection oriented stuff as we
 * may have several simultaneous connections and connection structs in memory.
 *
 * The 'struct UserDefined' must only contain data that is set once to go for
 * many (perhaps) independent connections. Values that are generated or
 * calculated internally for the "session handle" must be defined within the
 * 'struct UrlState' instead.
 */

struct Curl_easy {
  /* First a simple identifier to easier detect if a user mix up this easy
     handle with a multi handle. Set this to CURLEASY_MAGIC_NUMBER */
  unsigned int magic;
  /* once an easy handle is tied to a connection pool a non-negative number to
     distinguish this transfer from other using the same pool. For easier
     tracking in log output. This may wrap around after LONG_MAX to 0 again,
     so it has no uniqueness guarantee for large processings. Note: it has no
     uniqueness either IFF more than one connection pool is used by the
     libcurl application. */
  curl_off_t id;
  /* once an easy handle is added to a multi, either explicitly by the
   * libcurl application or implicitly during `curl_easy_perform()`,
   * a unique identifier inside this one multi instance. */
  curl_off_t mid;

  struct connectdata *conn;
  struct Curl_llist_node multi_queue; /* for multihandle list management */
  struct Curl_llist_node conn_queue; /* list per connectdata */

  CURLMstate mstate;  /* the handle's state */
  CURLcode result;   /* previous result */

  struct Curl_message msg; /* A single posted message. */

  /* Array with the plain socket numbers this handle takes care of, in no
     particular order. Note that all sockets are added to the sockhash, where
     the state etc are also kept. This array is mostly used to detect when a
     socket is to be removed from the hash. See singlesocket(). */
  struct easy_pollset last_poll;

  struct Names dns;
  struct Curl_multi *multi;    /* if non-NULL, points to the multi handle
                                  struct to which this "belongs" when used by
                                  the multi interface */
  struct Curl_multi *multi_easy; /* if non-NULL, points to the multi handle
                                    struct to which this "belongs" when used
                                    by the easy interface */
  struct Curl_share *share;    /* Share, handles global variable mutexing */
#ifdef USE_LIBPSL
  struct PslCache *psl;        /* The associated PSL cache. */
#endif
  struct SingleRequest req;    /* Request-specific data */
  struct UserDefined set;      /* values set by the libcurl user */
#ifndef CURL_DISABLE_COOKIES
  struct CookieInfo *cookies;  /* the cookies, read from files and servers.
                                  NOTE that the 'cookie' field in the
                                  UserDefined struct defines if the "engine"
                                  is to be used or not. */
#endif
#ifndef CURL_DISABLE_HSTS
  struct hsts *hsts;
#endif
#ifndef CURL_DISABLE_ALTSVC
  struct altsvcinfo *asi;      /* the alt-svc cache */
#endif
  struct Progress progress;    /* for all the progress meter data */
  struct UrlState state;       /* struct for fields used for state info and
                                  other dynamic purposes */
#ifndef CURL_DISABLE_FTP
  struct WildcardData *wildcard; /* wildcard download state info */
#endif
  struct PureInfo info;        /* stats, reports and info data */
  struct curl_tlssessioninfo tsi; /* Information about the TLS session, only
                                     valid after a client has asked for it */
#ifdef USE_HYPER
  struct hyptransfer hyp;
#endif
};

#define LIBCURL_NAME "libcurl"

#endif /* HEADER_CURL_URLDATA_H */
