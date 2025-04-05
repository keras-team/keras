/* Copyright (c) 2015, Google Inc.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
 * OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#ifndef OPENSSL_HEADER_CRYPTO_CONF_INTERNAL_H
#define OPENSSL_HEADER_CRYPTO_CONF_INTERNAL_H

#include <openssl/base.h>

#if defined(__cplusplus)
extern "C" {
#endif


// CONF_VALUE_new returns a freshly allocated and zeroed |CONF_VALUE|.
CONF_VALUE *CONF_VALUE_new(void);

// CONF_parse_list takes a list separated by 'sep' and calls |list_cb| giving
// the start and length of each member, optionally stripping leading and
// trailing whitespace. This can be used to parse comma separated lists for
// example. If |list_cb| returns <= 0, then the iteration is halted and that
// value is returned immediately. Otherwise it returns one. Note that |list_cb|
// may be called on an empty member.
OPENSSL_EXPORT int CONF_parse_list(
    const char *list, char sep, int remove_whitespace,
    int (*list_cb)(const char *elem, size_t len, void *usr), void *arg);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_CONF_INTERNAL_H
