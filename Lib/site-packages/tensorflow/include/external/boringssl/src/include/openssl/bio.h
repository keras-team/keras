/* Copyright (C) 1995-1998 Eric Young (eay@cryptsoft.com)
 * All rights reserved.
 *
 * This package is an SSL implementation written
 * by Eric Young (eay@cryptsoft.com).
 * The implementation was written so as to conform with Netscapes SSL.
 *
 * This library is free for commercial and non-commercial use as long as
 * the following conditions are aheared to.  The following conditions
 * apply to all code found in this distribution, be it the RC4, RSA,
 * lhash, DES, etc., code; not just the SSL code.  The SSL documentation
 * included with this distribution is covered by the same copyright terms
 * except that the holder is Tim Hudson (tjh@cryptsoft.com).
 *
 * Copyright remains Eric Young's, and as such any Copyright notices in
 * the code are not to be removed.
 * If this package is used in a product, Eric Young should be given attribution
 * as the author of the parts of the library used.
 * This can be in the form of a textual message at program startup or
 * in documentation (online or textual) provided with the package.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    "This product includes cryptographic software written by
 *     Eric Young (eay@cryptsoft.com)"
 *    The word 'cryptographic' can be left out if the rouines from the library
 *    being used are not cryptographic related :-).
 * 4. If you include any Windows specific code (or a derivative thereof) from
 *    the apps directory (application code) you must include an acknowledgement:
 *    "This product includes software written by Tim Hudson (tjh@cryptsoft.com)"
 *
 * THIS SOFTWARE IS PROVIDED BY ERIC YOUNG ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * The licence and distribution terms for any publically available version or
 * derivative of this code cannot be changed.  i.e. this code cannot simply be
 * copied and put under another distribution licence
 * [including the GNU Public Licence.] */

#ifndef OPENSSL_HEADER_BIO_H
#define OPENSSL_HEADER_BIO_H

#include <openssl/base.h>

#include <stdio.h>  // For FILE

#include <openssl/buffer.h>
#include <openssl/err.h>  // for ERR_print_errors_fp
#include <openssl/ex_data.h>
#include <openssl/stack.h>
#include <openssl/thread.h>

#if defined(__cplusplus)
extern "C" {
#endif


// BIO abstracts over a file-descriptor like interface.


// Allocation and freeing.

DEFINE_STACK_OF(BIO)

// BIO_new creates a new BIO with the given method and a reference count of one.
// It returns the fresh |BIO|, or NULL on error.
OPENSSL_EXPORT BIO *BIO_new(const BIO_METHOD *method);

// BIO_free decrements the reference count of |bio|. If the reference count
// drops to zero, it calls the destroy callback, if present, on the method and
// frees |bio| itself. It then repeats that for the next BIO in the chain, if
// any.
//
// It returns one on success or zero otherwise.
OPENSSL_EXPORT int BIO_free(BIO *bio);

// BIO_vfree performs the same actions as |BIO_free|, but has a void return
// value. This is provided for API-compat.
//
// TODO(fork): remove.
OPENSSL_EXPORT void BIO_vfree(BIO *bio);

// BIO_up_ref increments the reference count of |bio| and returns one.
OPENSSL_EXPORT int BIO_up_ref(BIO *bio);


// Basic I/O.

// BIO_read attempts to read |len| bytes into |data|. It returns the number of
// bytes read, zero on EOF, or a negative number on error.
OPENSSL_EXPORT int BIO_read(BIO *bio, void *data, int len);

// BIO_gets "reads a line" from |bio| and puts at most |size| bytes into |buf|.
// It returns the number of bytes read or a negative number on error. The
// phrase "reads a line" is in quotes in the previous sentence because the
// exact operation depends on the BIO's method. For example, a digest BIO will
// return the digest in response to a |BIO_gets| call.
//
// TODO(fork): audit the set of BIOs that we end up needing. If all actually
// return a line for this call, remove the warning above.
OPENSSL_EXPORT int BIO_gets(BIO *bio, char *buf, int size);

// BIO_write writes |len| bytes from |data| to |bio|. It returns the number of
// bytes written or a negative number on error.
OPENSSL_EXPORT int BIO_write(BIO *bio, const void *data, int len);

// BIO_write_all writes |len| bytes from |data| to |bio|, looping as necessary.
// It returns one if all bytes were successfully written and zero on error.
OPENSSL_EXPORT int BIO_write_all(BIO *bio, const void *data, size_t len);

// BIO_puts writes a NUL terminated string from |buf| to |bio|. It returns the
// number of bytes written or a negative number on error.
OPENSSL_EXPORT int BIO_puts(BIO *bio, const char *buf);

// BIO_flush flushes any buffered output. It returns one on success and zero
// otherwise.
OPENSSL_EXPORT int BIO_flush(BIO *bio);


// Low-level control functions.
//
// These are generic functions for sending control requests to a BIO. In
// general one should use the wrapper functions like |BIO_get_close|.

// BIO_ctrl sends the control request |cmd| to |bio|. The |cmd| argument should
// be one of the |BIO_C_*| values.
OPENSSL_EXPORT long BIO_ctrl(BIO *bio, int cmd, long larg, void *parg);

// BIO_ptr_ctrl acts like |BIO_ctrl| but passes the address of a |void*|
// pointer as |parg| and returns the value that is written to it, or NULL if
// the control request returns <= 0.
OPENSSL_EXPORT char *BIO_ptr_ctrl(BIO *bp, int cmd, long larg);

// BIO_int_ctrl acts like |BIO_ctrl| but passes the address of a copy of |iarg|
// as |parg|.
OPENSSL_EXPORT long BIO_int_ctrl(BIO *bp, int cmd, long larg, int iarg);

// BIO_reset resets |bio| to its initial state, the precise meaning of which
// depends on the concrete type of |bio|. It returns one on success and zero
// otherwise.
OPENSSL_EXPORT int BIO_reset(BIO *bio);

// BIO_eof returns non-zero when |bio| has reached end-of-file. The precise
// meaning of which depends on the concrete type of |bio|. Note that in the
// case of BIO_pair this always returns non-zero.
OPENSSL_EXPORT int BIO_eof(BIO *bio);

// BIO_set_flags ORs |flags| with |bio->flags|.
OPENSSL_EXPORT void BIO_set_flags(BIO *bio, int flags);

// BIO_test_flags returns |bio->flags| AND |flags|.
OPENSSL_EXPORT int BIO_test_flags(const BIO *bio, int flags);

// BIO_should_read returns non-zero if |bio| encountered a temporary error
// while reading (i.e. EAGAIN), indicating that the caller should retry the
// read.
OPENSSL_EXPORT int BIO_should_read(const BIO *bio);

// BIO_should_write returns non-zero if |bio| encountered a temporary error
// while writing (i.e. EAGAIN), indicating that the caller should retry the
// write.
OPENSSL_EXPORT int BIO_should_write(const BIO *bio);

// BIO_should_retry returns non-zero if the reason that caused a failed I/O
// operation is temporary and thus the operation should be retried. Otherwise,
// it was a permanent error and it returns zero.
OPENSSL_EXPORT int BIO_should_retry(const BIO *bio);

// BIO_should_io_special returns non-zero if |bio| encountered a temporary
// error while performing a special I/O operation, indicating that the caller
// should retry. The operation that caused the error is returned by
// |BIO_get_retry_reason|.
OPENSSL_EXPORT int BIO_should_io_special(const BIO *bio);

// BIO_RR_CONNECT indicates that a connect would have blocked
#define BIO_RR_CONNECT 0x02

// BIO_RR_ACCEPT indicates that an accept would have blocked
#define BIO_RR_ACCEPT 0x03

// BIO_get_retry_reason returns the special I/O operation that needs to be
// retried. The return value is one of the |BIO_RR_*| values.
OPENSSL_EXPORT int BIO_get_retry_reason(const BIO *bio);

// BIO_set_retry_reason sets the special I/O operation that needs to be retried
// to |reason|, which should be one of the |BIO_RR_*| values.
OPENSSL_EXPORT void BIO_set_retry_reason(BIO *bio, int reason);

// BIO_clear_flags ANDs |bio->flags| with the bitwise-complement of |flags|.
OPENSSL_EXPORT void BIO_clear_flags(BIO *bio, int flags);

// BIO_set_retry_read sets the |BIO_FLAGS_READ| and |BIO_FLAGS_SHOULD_RETRY|
// flags on |bio|.
OPENSSL_EXPORT void BIO_set_retry_read(BIO *bio);

// BIO_set_retry_write sets the |BIO_FLAGS_WRITE| and |BIO_FLAGS_SHOULD_RETRY|
// flags on |bio|.
OPENSSL_EXPORT void BIO_set_retry_write(BIO *bio);

// BIO_get_retry_flags gets the |BIO_FLAGS_READ|, |BIO_FLAGS_WRITE|,
// |BIO_FLAGS_IO_SPECIAL| and |BIO_FLAGS_SHOULD_RETRY| flags from |bio|.
OPENSSL_EXPORT int BIO_get_retry_flags(BIO *bio);

// BIO_clear_retry_flags clears the |BIO_FLAGS_READ|, |BIO_FLAGS_WRITE|,
// |BIO_FLAGS_IO_SPECIAL| and |BIO_FLAGS_SHOULD_RETRY| flags from |bio|.
OPENSSL_EXPORT void BIO_clear_retry_flags(BIO *bio);

// BIO_method_type returns the type of |bio|, which is one of the |BIO_TYPE_*|
// values.
OPENSSL_EXPORT int BIO_method_type(const BIO *bio);

// These are passed to the BIO callback
#define BIO_CB_FREE 0x01
#define BIO_CB_READ 0x02
#define BIO_CB_WRITE 0x03
#define BIO_CB_PUTS 0x04
#define BIO_CB_GETS 0x05
#define BIO_CB_CTRL 0x06

// The callback is called before and after the underling operation,
// The BIO_CB_RETURN flag indicates if it is after the call
#define BIO_CB_RETURN 0x80

// bio_info_cb is the type of a callback function that can be called for most
// BIO operations. The |event| argument is one of |BIO_CB_*| and can be ORed
// with |BIO_CB_RETURN| if the callback is being made after the operation in
// question. In that case, |return_value| will contain the return value from
// the operation.
typedef long (*bio_info_cb)(BIO *bio, int event, const char *parg, int cmd,
                            long larg, long return_value);

// BIO_callback_ctrl allows the callback function to be manipulated. The |cmd|
// arg will generally be |BIO_CTRL_SET_CALLBACK| but arbitrary command values
// can be interpreted by the |BIO|.
OPENSSL_EXPORT long BIO_callback_ctrl(BIO *bio, int cmd, bio_info_cb fp);

// BIO_pending returns the number of bytes pending to be read.
OPENSSL_EXPORT size_t BIO_pending(const BIO *bio);

// BIO_ctrl_pending calls |BIO_pending| and exists only for compatibility with
// OpenSSL.
OPENSSL_EXPORT size_t BIO_ctrl_pending(const BIO *bio);

// BIO_wpending returns the number of bytes pending to be written.
OPENSSL_EXPORT size_t BIO_wpending(const BIO *bio);

// BIO_set_close sets the close flag for |bio|. The meaning of which depends on
// the type of |bio| but, for example, a memory BIO interprets the close flag
// as meaning that it owns its buffer. It returns one on success and zero
// otherwise.
OPENSSL_EXPORT int BIO_set_close(BIO *bio, int close_flag);

// BIO_number_read returns the number of bytes that have been read from
// |bio|.
OPENSSL_EXPORT size_t BIO_number_read(const BIO *bio);

// BIO_number_written returns the number of bytes that have been written to
// |bio|.
OPENSSL_EXPORT size_t BIO_number_written(const BIO *bio);


// Managing chains of BIOs.
//
// BIOs can be put into chains where the output of one is used as the input of
// the next etc. The most common case is a buffering BIO, which accepts and
// buffers writes until flushed into the next BIO in the chain.

// BIO_push adds |appended_bio| to the end of the chain with |bio| at the head.
// It returns |bio|. Note that |appended_bio| may be the head of a chain itself
// and thus this function can be used to join two chains.
//
// BIO_push takes ownership of the caller's reference to |appended_bio|.
OPENSSL_EXPORT BIO *BIO_push(BIO *bio, BIO *appended_bio);

// BIO_pop removes |bio| from the head of a chain and returns the next BIO in
// the chain, or NULL if there is no next BIO.
//
// The caller takes ownership of the chain's reference to |bio|.
OPENSSL_EXPORT BIO *BIO_pop(BIO *bio);

// BIO_next returns the next BIO in the chain after |bio|, or NULL if there is
// no such BIO.
OPENSSL_EXPORT BIO *BIO_next(BIO *bio);

// BIO_free_all calls |BIO_free|.
//
// TODO(fork): update callers and remove.
OPENSSL_EXPORT void BIO_free_all(BIO *bio);

// BIO_find_type walks a chain of BIOs and returns the first that matches
// |type|, which is one of the |BIO_TYPE_*| values.
OPENSSL_EXPORT BIO *BIO_find_type(BIO *bio, int type);

// BIO_copy_next_retry sets the retry flags and |retry_reason| of |bio| from
// the next BIO in the chain.
OPENSSL_EXPORT void BIO_copy_next_retry(BIO *bio);


// Printf functions.

// BIO_printf behaves like |printf| but outputs to |bio| rather than a |FILE|.
// It returns the number of bytes written or a negative number on error.
OPENSSL_EXPORT int BIO_printf(BIO *bio, const char *format, ...)
    OPENSSL_PRINTF_FORMAT_FUNC(2, 3);


// Utility functions.

// BIO_indent prints min(|indent|, |max_indent|) spaces. It returns one on
// success and zero otherwise.
OPENSSL_EXPORT int BIO_indent(BIO *bio, unsigned indent, unsigned max_indent);

// BIO_hexdump writes a hex dump of |data| to |bio|. Each line will be indented
// by |indent| spaces. It returns one on success and zero otherwise.
OPENSSL_EXPORT int BIO_hexdump(BIO *bio, const uint8_t *data, size_t len,
                               unsigned indent);

// ERR_print_errors prints the current contents of the error stack to |bio|
// using human readable strings where possible.
OPENSSL_EXPORT void ERR_print_errors(BIO *bio);

// BIO_read_asn1 reads a single ASN.1 object from |bio|. If successful it sets
// |*out| to be an allocated buffer (that should be freed with |OPENSSL_free|),
// |*out_size| to the length, in bytes, of that buffer and returns one.
// Otherwise it returns zero.
//
// If the length of the object is greater than |max_len| or 2^32 then the
// function will fail. Long-form tags are not supported. If the length of the
// object is indefinite the full contents of |bio| are read, unless it would be
// greater than |max_len|, in which case the function fails.
//
// If the function fails then some unknown amount of data may have been read
// from |bio|.
OPENSSL_EXPORT int BIO_read_asn1(BIO *bio, uint8_t **out, size_t *out_len,
                                 size_t max_len);


// Memory BIOs.
//
// Memory BIOs can be used as a read-only source (with |BIO_new_mem_buf|) or a
// writable sink (with |BIO_new|, |BIO_s_mem| and |BIO_mem_contents|). Data
// written to a writable, memory BIO can be recalled by reading from it.
//
// Calling |BIO_reset| on a read-only BIO resets it to the original contents.
// On a writable BIO, it clears any data.
//
// If the close flag is set to |BIO_NOCLOSE| (not the default) then the
// underlying |BUF_MEM| will not be freed when the |BIO| is freed.
//
// Memory BIOs support |BIO_gets| and |BIO_puts|.
//
// |BIO_ctrl_pending| returns the number of bytes currently stored.

// BIO_NOCLOSE and |BIO_CLOSE| can be used as symbolic arguments when a "close
// flag" is passed to a BIO function.
#define BIO_NOCLOSE 0
#define BIO_CLOSE 1

// BIO_s_mem returns a |BIO_METHOD| that uses a in-memory buffer.
OPENSSL_EXPORT const BIO_METHOD *BIO_s_mem(void);

// BIO_new_mem_buf creates read-only BIO that reads from |len| bytes at |buf|.
// It returns the BIO or NULL on error. This function does not copy or take
// ownership of |buf|. The caller must ensure the memory pointed to by |buf|
// outlives the |BIO|.
//
// If |len| is negative, then |buf| is treated as a NUL-terminated string, but
// don't depend on this in new code.
OPENSSL_EXPORT BIO *BIO_new_mem_buf(const void *buf, ossl_ssize_t len);

// BIO_mem_contents sets |*out_contents| to point to the current contents of
// |bio| and |*out_len| to contain the length of that data. It returns one on
// success and zero otherwise.
OPENSSL_EXPORT int BIO_mem_contents(const BIO *bio,
                                    const uint8_t **out_contents,
                                    size_t *out_len);

// BIO_get_mem_data sets |*contents| to point to the current contents of |bio|
// and returns the length of the data.
//
// WARNING: don't use this, use |BIO_mem_contents|. A return value of zero from
// this function can mean either that it failed or that the memory buffer is
// empty.
OPENSSL_EXPORT long BIO_get_mem_data(BIO *bio, char **contents);

// BIO_get_mem_ptr sets |*out| to a BUF_MEM containing the current contents of
// |bio|. It returns one on success or zero on error.
OPENSSL_EXPORT int BIO_get_mem_ptr(BIO *bio, BUF_MEM **out);

// BIO_set_mem_buf sets |b| as the contents of |bio|. If |take_ownership| is
// non-zero, then |b| will be freed when |bio| is closed. Returns one on
// success or zero otherwise.
OPENSSL_EXPORT int BIO_set_mem_buf(BIO *bio, BUF_MEM *b, int take_ownership);

// BIO_set_mem_eof_return sets the value that will be returned from reading
// |bio| when empty. If |eof_value| is zero then an empty memory BIO will
// return EOF (that is it will return zero and |BIO_should_retry| will be
// false). If |eof_value| is non zero then it will return |eof_value| when it
// is empty and it will set the read retry flag (that is |BIO_read_retry| is
// true). To avoid ambiguity with a normal positive return value, |eof_value|
// should be set to a negative value, typically -1.
//
// For a read-only BIO, the default is zero (EOF). For a writable BIO, the
// default is -1 so that additional data can be written once exhausted.
OPENSSL_EXPORT int BIO_set_mem_eof_return(BIO *bio, int eof_value);


// File descriptor BIOs.
//
// File descriptor BIOs are wrappers around the system's |read| and |write|
// functions. If the close flag is set then then |close| is called on the
// underlying file descriptor when the BIO is freed.
//
// |BIO_reset| attempts to seek the file pointer to the start of file using
// |lseek|.

// BIO_s_fd returns a |BIO_METHOD| for file descriptor fds.
OPENSSL_EXPORT const BIO_METHOD *BIO_s_fd(void);

// BIO_new_fd creates a new file descriptor BIO wrapping |fd|. If |close_flag|
// is non-zero, then |fd| will be closed when the BIO is.
OPENSSL_EXPORT BIO *BIO_new_fd(int fd, int close_flag);

// BIO_set_fd sets the file descriptor of |bio| to |fd|. If |close_flag| is
// non-zero then |fd| will be closed when |bio| is. It returns one on success
// or zero on error.
//
// This function may also be used with socket BIOs (see |BIO_s_socket| and
// |BIO_new_socket|).
OPENSSL_EXPORT int BIO_set_fd(BIO *bio, int fd, int close_flag);

// BIO_get_fd returns the file descriptor currently in use by |bio| or -1 if
// |bio| does not wrap a file descriptor. If there is a file descriptor and
// |out_fd| is not NULL, it also sets |*out_fd| to the file descriptor.
//
// This function may also be used with socket BIOs (see |BIO_s_socket| and
// |BIO_new_socket|).
OPENSSL_EXPORT int BIO_get_fd(BIO *bio, int *out_fd);


// File BIOs.
//
// File BIOs are wrappers around a C |FILE| object.
//
// |BIO_flush| on a file BIO calls |fflush| on the wrapped stream.
//
// |BIO_reset| attempts to seek the file pointer to the start of file using
// |fseek|.
//
// Setting the close flag causes |fclose| to be called on the stream when the
// BIO is freed.

// BIO_s_file returns a BIO_METHOD that wraps a |FILE|.
OPENSSL_EXPORT const BIO_METHOD *BIO_s_file(void);

// BIO_new_file creates a file BIO by opening |filename| with the given mode.
// See the |fopen| manual page for details of the mode argument.
OPENSSL_EXPORT BIO *BIO_new_file(const char *filename, const char *mode);

// BIO_new_fp creates a new file BIO that wraps the given |FILE|. If
// |close_flag| is |BIO_CLOSE|, then |fclose| will be called on |stream| when
// the BIO is closed.
OPENSSL_EXPORT BIO *BIO_new_fp(FILE *stream, int close_flag);

// BIO_get_fp sets |*out_file| to the current |FILE| for |bio|. It returns one
// on success and zero otherwise.
OPENSSL_EXPORT int BIO_get_fp(BIO *bio, FILE **out_file);

// BIO_set_fp sets the |FILE| for |bio|. If |close_flag| is |BIO_CLOSE| then
// |fclose| will be called on |file| when |bio| is closed. It returns one on
// success and zero otherwise.
OPENSSL_EXPORT int BIO_set_fp(BIO *bio, FILE *file, int close_flag);

// BIO_read_filename opens |filename| for reading and sets the result as the
// |FILE| for |bio|. It returns one on success and zero otherwise. The |FILE|
// will be closed when |bio| is freed.
OPENSSL_EXPORT int BIO_read_filename(BIO *bio, const char *filename);

// BIO_write_filename opens |filename| for writing and sets the result as the
// |FILE| for |bio|. It returns one on success and zero otherwise. The |FILE|
// will be closed when |bio| is freed.
OPENSSL_EXPORT int BIO_write_filename(BIO *bio, const char *filename);

// BIO_append_filename opens |filename| for appending and sets the result as
// the |FILE| for |bio|. It returns one on success and zero otherwise. The
// |FILE| will be closed when |bio| is freed.
OPENSSL_EXPORT int BIO_append_filename(BIO *bio, const char *filename);

// BIO_rw_filename opens |filename| for reading and writing and sets the result
// as the |FILE| for |bio|. It returns one on success and zero otherwise. The
// |FILE| will be closed when |bio| is freed.
OPENSSL_EXPORT int BIO_rw_filename(BIO *bio, const char *filename);

// BIO_tell returns the file offset of |bio|, or a negative number on error or
// if |bio| does not support the operation.
//
// TODO(https://crbug.com/boringssl/465): On platforms where |long| is 32-bit,
// this function cannot report 64-bit offsets.
OPENSSL_EXPORT long BIO_tell(BIO *bio);

// BIO_seek sets the file offset of |bio| to |offset|. It returns a non-negative
// number on success and a negative number on error. If |bio| is a file
// descriptor |BIO|, it returns the resulting file offset on success. If |bio|
// is a file |BIO|, it returns zero on success.
//
// WARNING: This function's return value conventions differs from most functions
// in this library.
//
// TODO(https://crbug.com/boringssl/465): On platforms where |long| is 32-bit,
// this function cannot handle 64-bit offsets.
OPENSSL_EXPORT long BIO_seek(BIO *bio, long offset);


// Socket BIOs.
//
// Socket BIOs behave like file descriptor BIOs but, on Windows systems, wrap
// the system's |recv| and |send| functions instead of |read| and |write|. On
// Windows, file descriptors are provided by C runtime and are not
// interchangeable with sockets.
//
// Socket BIOs may be used with |BIO_set_fd| and |BIO_get_fd|.
//
// TODO(davidben): Add separate APIs and fix the internals to use |SOCKET|s
// around rather than rely on int casts.

OPENSSL_EXPORT const BIO_METHOD *BIO_s_socket(void);

// BIO_new_socket allocates and initialises a fresh BIO which will read and
// write to the socket |fd|. If |close_flag| is |BIO_CLOSE| then closing the
// BIO will close |fd|. It returns the fresh |BIO| or NULL on error.
OPENSSL_EXPORT BIO *BIO_new_socket(int fd, int close_flag);


// Connect BIOs.
//
// A connection BIO creates a network connection and transfers data over the
// resulting socket.

OPENSSL_EXPORT const BIO_METHOD *BIO_s_connect(void);

// BIO_new_connect returns a BIO that connects to the given hostname and port.
// The |host_and_optional_port| argument should be of the form
// "www.example.com" or "www.example.com:443". If the port is omitted, it must
// be provided with |BIO_set_conn_port|.
//
// It returns the new BIO on success, or NULL on error.
OPENSSL_EXPORT BIO *BIO_new_connect(const char *host_and_optional_port);

// BIO_set_conn_hostname sets |host_and_optional_port| as the hostname and
// optional port that |bio| will connect to. If the port is omitted, it must be
// provided with |BIO_set_conn_port|.
//
// It returns one on success and zero otherwise.
OPENSSL_EXPORT int BIO_set_conn_hostname(BIO *bio,
                                         const char *host_and_optional_port);

// BIO_set_conn_port sets |port_str| as the port or service name that |bio|
// will connect to. It returns one on success and zero otherwise.
OPENSSL_EXPORT int BIO_set_conn_port(BIO *bio, const char *port_str);

// BIO_set_conn_int_port sets |*port| as the port that |bio| will connect to.
// It returns one on success and zero otherwise.
OPENSSL_EXPORT int BIO_set_conn_int_port(BIO *bio, const int *port);

// BIO_set_nbio sets whether |bio| will use non-blocking I/O operations. It
// returns one on success and zero otherwise.
OPENSSL_EXPORT int BIO_set_nbio(BIO *bio, int on);

// BIO_do_connect connects |bio| if it has not been connected yet. It returns
// one on success and <= 0 otherwise.
OPENSSL_EXPORT int BIO_do_connect(BIO *bio);


// Datagram BIOs.
//
// TODO(fork): not implemented.

#define BIO_CTRL_DGRAM_QUERY_MTU 40  // as kernel for current MTU

#define BIO_CTRL_DGRAM_SET_MTU 42 /* set cached value for  MTU. want to use
                                     this if asking the kernel fails */

#define BIO_CTRL_DGRAM_MTU_EXCEEDED 43 /* check whether the MTU was exceed in
                                          the previous write operation. */

// BIO_CTRL_DGRAM_SET_NEXT_TIMEOUT is unsupported as it is unused by consumers
// and depends on |timeval|, which is not 2038-clean on all platforms.

#define BIO_CTRL_DGRAM_GET_PEER           46

#define BIO_CTRL_DGRAM_GET_FALLBACK_MTU   47


// BIO Pairs.
//
// BIO pairs provide a "loopback" like system: a pair of BIOs where data
// written to one can be read from the other and vice versa.

// BIO_new_bio_pair sets |*out1| and |*out2| to two freshly created BIOs where
// data written to one can be read from the other and vice versa. The
// |writebuf1| argument gives the size of the buffer used in |*out1| and
// |writebuf2| for |*out2|. It returns one on success and zero on error.
OPENSSL_EXPORT int BIO_new_bio_pair(BIO **out1, size_t writebuf1, BIO **out2,
                                    size_t writebuf2);

// BIO_ctrl_get_read_request returns the number of bytes that the other side of
// |bio| tried (unsuccessfully) to read.
OPENSSL_EXPORT size_t BIO_ctrl_get_read_request(BIO *bio);

// BIO_ctrl_get_write_guarantee returns the number of bytes that |bio| (which
// must have been returned by |BIO_new_bio_pair|) will accept on the next
// |BIO_write| call.
OPENSSL_EXPORT size_t BIO_ctrl_get_write_guarantee(BIO *bio);

// BIO_shutdown_wr marks |bio| as closed, from the point of view of the other
// side of the pair. Future |BIO_write| calls on |bio| will fail. It returns
// one on success and zero otherwise.
OPENSSL_EXPORT int BIO_shutdown_wr(BIO *bio);


// Custom BIOs.
//
// Consumers can create custom |BIO|s by filling in a |BIO_METHOD| and using
// low-level control functions to set state.

// BIO_get_new_index returns a new "type" value for a custom |BIO|.
OPENSSL_EXPORT int BIO_get_new_index(void);

// BIO_meth_new returns a newly-allocated |BIO_METHOD| or NULL on allocation
// error. The |type| specifies the type that will be returned by
// |BIO_method_type|. If this is unnecessary, this value may be zero. The |name|
// parameter is vestigial and may be NULL.
//
// Use the |BIO_meth_set_*| functions below to initialize the |BIO_METHOD|. The
// function implementations may use |BIO_set_data| and |BIO_get_data| to add
// method-specific state to associated |BIO|s. Additionally, |BIO_set_init| must
// be called after an associated |BIO| is fully initialized. State set via
// |BIO_set_data| may be released by configuring a destructor with
// |BIO_meth_set_destroy|.
OPENSSL_EXPORT BIO_METHOD *BIO_meth_new(int type, const char *name);

// BIO_meth_free releases memory associated with |method|.
OPENSSL_EXPORT void BIO_meth_free(BIO_METHOD *method);

// BIO_meth_set_create sets a function to be called on |BIO_new| for |method|
// and returns one. The function should return one on success and zero on
// error.
OPENSSL_EXPORT int BIO_meth_set_create(BIO_METHOD *method,
                                       int (*create)(BIO *));

// BIO_meth_set_destroy sets a function to release data associated with a |BIO|
// and returns one. The function's return value is ignored.
OPENSSL_EXPORT int BIO_meth_set_destroy(BIO_METHOD *method,
                                        int (*destroy)(BIO *));

// BIO_meth_set_write sets the implementation of |BIO_write| for |method| and
// returns one. |BIO_METHOD|s which implement |BIO_write| should also implement
// |BIO_CTRL_FLUSH|. (See |BIO_meth_set_ctrl|.)
OPENSSL_EXPORT int BIO_meth_set_write(BIO_METHOD *method,
                                      int (*write)(BIO *, const char *, int));

// BIO_meth_set_read sets the implementation of |BIO_read| for |method| and
// returns one.
OPENSSL_EXPORT int BIO_meth_set_read(BIO_METHOD *method,
                                     int (*read)(BIO *, char *, int));

// BIO_meth_set_gets sets the implementation of |BIO_gets| for |method| and
// returns one.
OPENSSL_EXPORT int BIO_meth_set_gets(BIO_METHOD *method,
                                     int (*gets)(BIO *, char *, int));

// BIO_meth_set_ctrl sets the implementation of |BIO_ctrl| for |method| and
// returns one.
OPENSSL_EXPORT int BIO_meth_set_ctrl(BIO_METHOD *method,
                                     long (*ctrl)(BIO *, int, long, void *));

// BIO_set_data sets custom data on |bio|. It may be retried with
// |BIO_get_data|.
OPENSSL_EXPORT void BIO_set_data(BIO *bio, void *ptr);

// BIO_get_data returns custom data on |bio| set by |BIO_get_data|.
OPENSSL_EXPORT void *BIO_get_data(BIO *bio);

// BIO_set_init sets whether |bio| has been fully initialized. Until fully
// initialized, |BIO_read| and |BIO_write| will fail.
OPENSSL_EXPORT void BIO_set_init(BIO *bio, int init);

// BIO_get_init returns whether |bio| has been fully initialized.
OPENSSL_EXPORT int BIO_get_init(BIO *bio);

// These are values of the |cmd| argument to |BIO_ctrl|.

// BIO_CTRL_RESET implements |BIO_reset|. The arguments are unused.
#define BIO_CTRL_RESET 1

// BIO_CTRL_EOF implements |BIO_eof|. The arguments are unused.
#define BIO_CTRL_EOF 2

// BIO_CTRL_INFO is a legacy command that returns information specific to the
// type of |BIO|. It is not safe to call generically and should not be
// implemented in new |BIO| types.
#define BIO_CTRL_INFO 3

// BIO_CTRL_GET_CLOSE returns the close flag set by |BIO_CTRL_SET_CLOSE|. The
// arguments are unused.
#define BIO_CTRL_GET_CLOSE 8

// BIO_CTRL_SET_CLOSE implements |BIO_set_close|. The |larg| argument is the
// close flag.
#define BIO_CTRL_SET_CLOSE 9

// BIO_CTRL_PENDING implements |BIO_pending|. The arguments are unused.
#define BIO_CTRL_PENDING 10

// BIO_CTRL_FLUSH implements |BIO_flush|. The arguments are unused.
#define BIO_CTRL_FLUSH 11

// BIO_CTRL_WPENDING implements |BIO_wpending|. The arguments are unused.
#define BIO_CTRL_WPENDING 13

// BIO_CTRL_SET_CALLBACK sets an informational callback of type
// int cb(BIO *bio, int state, int ret)
#define BIO_CTRL_SET_CALLBACK 14

// BIO_CTRL_GET_CALLBACK returns the callback set by |BIO_CTRL_SET_CALLBACK|.
#define BIO_CTRL_GET_CALLBACK 15

// The following are never used, but are defined to aid porting existing code.
#define BIO_CTRL_SET 4
#define BIO_CTRL_GET 5
#define BIO_CTRL_PUSH 6
#define BIO_CTRL_POP 7
#define BIO_CTRL_DUP 12
#define BIO_CTRL_SET_FILENAME 30


// Deprecated functions.

// BIO_f_base64 returns a filter |BIO| that base64-encodes data written into
// it, and decodes data read from it. |BIO_gets| is not supported. Call
// |BIO_flush| when done writing, to signal that no more data are to be
// encoded. The flag |BIO_FLAGS_BASE64_NO_NL| may be set to encode all the data
// on one line.
//
// Use |EVP_EncodeBlock| and |EVP_DecodeBase64| instead.
OPENSSL_EXPORT const BIO_METHOD *BIO_f_base64(void);

OPENSSL_EXPORT void BIO_set_retry_special(BIO *bio);

// BIO_set_write_buffer_size returns zero.
OPENSSL_EXPORT int BIO_set_write_buffer_size(BIO *bio, int buffer_size);

// BIO_set_shutdown sets a method-specific "shutdown" bit on |bio|.
OPENSSL_EXPORT void BIO_set_shutdown(BIO *bio, int shutdown);

// BIO_get_shutdown returns the method-specific "shutdown" bit.
OPENSSL_EXPORT int BIO_get_shutdown(BIO *bio);

// BIO_meth_set_puts returns one. |BIO_puts| is implemented with |BIO_write| in
// BoringSSL.
OPENSSL_EXPORT int BIO_meth_set_puts(BIO_METHOD *method,
                                     int (*puts)(BIO *, const char *));


// Private functions

#define BIO_FLAGS_READ 0x01
#define BIO_FLAGS_WRITE 0x02
#define BIO_FLAGS_IO_SPECIAL 0x04
#define BIO_FLAGS_RWS (BIO_FLAGS_READ | BIO_FLAGS_WRITE | BIO_FLAGS_IO_SPECIAL)
#define BIO_FLAGS_SHOULD_RETRY 0x08
#define BIO_FLAGS_BASE64_NO_NL 0x100
// BIO_FLAGS_MEM_RDONLY is used with memory BIOs. It means we shouldn't free up
// or change the data in any way.
#define BIO_FLAGS_MEM_RDONLY 0x200

// These are the 'types' of BIOs
#define BIO_TYPE_NONE 0
#define BIO_TYPE_MEM (1 | 0x0400)
#define BIO_TYPE_FILE (2 | 0x0400)
#define BIO_TYPE_FD (4 | 0x0400 | 0x0100)
#define BIO_TYPE_SOCKET (5 | 0x0400 | 0x0100)
#define BIO_TYPE_NULL (6 | 0x0400)
#define BIO_TYPE_SSL (7 | 0x0200)
#define BIO_TYPE_MD (8 | 0x0200)                 // passive filter
#define BIO_TYPE_BUFFER (9 | 0x0200)             // filter
#define BIO_TYPE_CIPHER (10 | 0x0200)            // filter
#define BIO_TYPE_BASE64 (11 | 0x0200)            // filter
#define BIO_TYPE_CONNECT (12 | 0x0400 | 0x0100)  // socket - connect
#define BIO_TYPE_ACCEPT (13 | 0x0400 | 0x0100)   // socket for accept
#define BIO_TYPE_PROXY_CLIENT (14 | 0x0200)      // client proxy BIO
#define BIO_TYPE_PROXY_SERVER (15 | 0x0200)      // server proxy BIO
#define BIO_TYPE_NBIO_TEST (16 | 0x0200)         // server proxy BIO
#define BIO_TYPE_NULL_FILTER (17 | 0x0200)
#define BIO_TYPE_BER (18 | 0x0200)         // BER -> bin filter
#define BIO_TYPE_BIO (19 | 0x0400)         // (half a) BIO pair
#define BIO_TYPE_LINEBUFFER (20 | 0x0200)  // filter
#define BIO_TYPE_DGRAM (21 | 0x0400 | 0x0100)
#define BIO_TYPE_ASN1 (22 | 0x0200)  // filter
#define BIO_TYPE_COMP (23 | 0x0200)  // filter

// BIO_TYPE_DESCRIPTOR denotes that the |BIO| responds to the |BIO_C_SET_FD|
// (|BIO_set_fd|) and |BIO_C_GET_FD| (|BIO_get_fd|) control hooks.
#define BIO_TYPE_DESCRIPTOR 0x0100  // socket, fd, connect or accept
#define BIO_TYPE_FILTER 0x0200
#define BIO_TYPE_SOURCE_SINK 0x0400

// BIO_TYPE_START is the first user-allocated |BIO| type. No pre-defined type,
// flag bits aside, may exceed this value.
#define BIO_TYPE_START 128

struct bio_method_st {
  int type;
  const char *name;
  int (*bwrite)(BIO *, const char *, int);
  int (*bread)(BIO *, char *, int);
  // TODO(fork): remove bputs.
  int (*bputs)(BIO *, const char *);
  int (*bgets)(BIO *, char *, int);
  long (*ctrl)(BIO *, int, long, void *);
  int (*create)(BIO *);
  int (*destroy)(BIO *);
  long (*callback_ctrl)(BIO *, int, bio_info_cb);
};

struct bio_st {
  const BIO_METHOD *method;

  // init is non-zero if this |BIO| has been initialised.
  int init;
  // shutdown is often used by specific |BIO_METHOD|s to determine whether
  // they own some underlying resource. This flag can often by controlled by
  // |BIO_set_close|. For example, whether an fd BIO closes the underlying fd
  // when it, itself, is closed.
  int shutdown;
  int flags;
  int retry_reason;
  // num is a BIO-specific value. For example, in fd BIOs it's used to store a
  // file descriptor.
  int num;
  CRYPTO_refcount_t references;
  void *ptr;
  // next_bio points to the next |BIO| in a chain. This |BIO| owns a reference
  // to |next_bio|.
  BIO *next_bio;  // used by filter BIOs
  size_t num_read, num_write;
};

#define BIO_C_SET_CONNECT 100
#define BIO_C_DO_STATE_MACHINE 101
#define BIO_C_SET_NBIO 102
#define BIO_C_SET_PROXY_PARAM 103
#define BIO_C_SET_FD 104
#define BIO_C_GET_FD 105
#define BIO_C_SET_FILE_PTR 106
#define BIO_C_GET_FILE_PTR 107
#define BIO_C_SET_FILENAME 108
#define BIO_C_SET_SSL 109
#define BIO_C_GET_SSL 110
#define BIO_C_SET_MD 111
#define BIO_C_GET_MD 112
#define BIO_C_GET_CIPHER_STATUS 113
#define BIO_C_SET_BUF_MEM 114
#define BIO_C_GET_BUF_MEM_PTR 115
#define BIO_C_GET_BUFF_NUM_LINES 116
#define BIO_C_SET_BUFF_SIZE 117
#define BIO_C_SET_ACCEPT 118
#define BIO_C_SSL_MODE 119
#define BIO_C_GET_MD_CTX 120
#define BIO_C_GET_PROXY_PARAM 121
#define BIO_C_SET_BUFF_READ_DATA 122  // data to read first
#define BIO_C_GET_ACCEPT 124
#define BIO_C_SET_SSL_RENEGOTIATE_BYTES 125
#define BIO_C_GET_SSL_NUM_RENEGOTIATES 126
#define BIO_C_SET_SSL_RENEGOTIATE_TIMEOUT 127
#define BIO_C_FILE_SEEK 128
#define BIO_C_GET_CIPHER_CTX 129
#define BIO_C_SET_BUF_MEM_EOF_RETURN 130  // return end of input value
#define BIO_C_SET_BIND_MODE 131
#define BIO_C_GET_BIND_MODE 132
#define BIO_C_FILE_TELL 133
#define BIO_C_GET_SOCKS 134
#define BIO_C_SET_SOCKS 135

#define BIO_C_SET_WRITE_BUF_SIZE 136  // for BIO_s_bio
#define BIO_C_GET_WRITE_BUF_SIZE 137
#define BIO_C_GET_WRITE_GUARANTEE 140
#define BIO_C_GET_READ_REQUEST 141
#define BIO_C_SHUTDOWN_WR 142
#define BIO_C_NREAD0 143
#define BIO_C_NREAD 144
#define BIO_C_NWRITE0 145
#define BIO_C_NWRITE 146
#define BIO_C_RESET_READ_REQUEST 147
#define BIO_C_SET_MD_CTX 148

#define BIO_C_SET_PREFIX 149
#define BIO_C_GET_PREFIX 150
#define BIO_C_SET_SUFFIX 151
#define BIO_C_GET_SUFFIX 152

#define BIO_C_SET_EX_ARG 153
#define BIO_C_GET_EX_ARG 154


#if defined(__cplusplus)
}  // extern C

extern "C++" {

BSSL_NAMESPACE_BEGIN

BORINGSSL_MAKE_DELETER(BIO, BIO_free)
BORINGSSL_MAKE_UP_REF(BIO, BIO_up_ref)
BORINGSSL_MAKE_DELETER(BIO_METHOD, BIO_meth_free)

BSSL_NAMESPACE_END

}  // extern C++

#endif

#define BIO_R_BAD_FOPEN_MODE 100
#define BIO_R_BROKEN_PIPE 101
#define BIO_R_CONNECT_ERROR 102
#define BIO_R_ERROR_SETTING_NBIO 103
#define BIO_R_INVALID_ARGUMENT 104
#define BIO_R_IN_USE 105
#define BIO_R_KEEPALIVE 106
#define BIO_R_NBIO_CONNECT_ERROR 107
#define BIO_R_NO_HOSTNAME_SPECIFIED 108
#define BIO_R_NO_PORT_SPECIFIED 109
#define BIO_R_NO_SUCH_FILE 110
#define BIO_R_NULL_PARAMETER 111
#define BIO_R_SYS_LIB 112
#define BIO_R_UNABLE_TO_CREATE_SOCKET 113
#define BIO_R_UNINITIALIZED 114
#define BIO_R_UNSUPPORTED_METHOD 115
#define BIO_R_WRITE_TO_READ_ONLY_BIO 116

#endif  // OPENSSL_HEADER_BIO_H
