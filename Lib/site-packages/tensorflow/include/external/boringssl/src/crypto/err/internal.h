/* Copyright (c) 2017, Google Inc.
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

#ifndef OPENSSL_HEADER_CRYPTO_ERR_INTERNAL_H
#define OPENSSL_HEADER_CRYPTO_ERR_INTERNAL_H

#include <openssl/err.h>

#if defined(__cplusplus)
extern "C" {
#endif


// Private error queue functions.

// ERR_SAVE_STATE contains a saved representation of the error queue. It is
// slightly more compact than |ERR_STATE| as the error queue will typically not
// contain |ERR_NUM_ERRORS| entries.
typedef struct err_save_state_st ERR_SAVE_STATE;

// ERR_SAVE_STATE_free releases all memory associated with |state|.
OPENSSL_EXPORT void ERR_SAVE_STATE_free(ERR_SAVE_STATE *state);

// ERR_save_state returns a newly-allocated |ERR_SAVE_STATE| structure
// containing the current state of the error queue or NULL on allocation
// error. It should be released with |ERR_SAVE_STATE_free|.
OPENSSL_EXPORT ERR_SAVE_STATE *ERR_save_state(void);

// ERR_restore_state clears the error queue and replaces it with |state|.
OPENSSL_EXPORT void ERR_restore_state(const ERR_SAVE_STATE *state);


#if defined(__cplusplus)
}  // extern C

extern "C++" {

BSSL_NAMESPACE_BEGIN

BORINGSSL_MAKE_DELETER(ERR_SAVE_STATE, ERR_SAVE_STATE_free)

BSSL_NAMESPACE_END

}  // extern C++
#endif

#endif  // OPENSSL_HEADER_CRYPTO_ERR_INTERNAL_H
