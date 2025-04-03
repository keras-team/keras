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

#include <openssl/crypto.h>

#include "../../internal.h"
#include "../delocate.h"


int FIPS_mode(void) {
#if defined(BORINGSSL_FIPS) && !defined(OPENSSL_ASAN)
  return 1;
#else
  return 0;
#endif
}

int FIPS_mode_set(int on) { return on == FIPS_mode(); }

const char *FIPS_module_name(void) { return "BoringCrypto"; }

uint32_t FIPS_version(void) {
  return 0;
}

int FIPS_query_algorithm_status(const char *algorithm) {
#if defined(BORINGSSL_FIPS)
  static const char kApprovedAlgorithms[][13] = {
    "AES-CBC",
    "AES-CCM",
    "AES-CTR",
    "AES-ECB",
    "AES-GCM",
    "AES-KW",
    "AES-KWP",
    "ctrDRBG",
    "ECC-SSC",
    "ECDSA-sign",
    "ECDSA-verify",
    "FFC-SSC",
    "HMAC",
    "RSA-sign",
    "RSA-verify",
    "SHA-1",
    "SHA2-224",
    "SHA2-256",
    "SHA2-384",
    "SHA2-512",
    "SHA2-512/256",
  };
  for (size_t i = 0; i < OPENSSL_ARRAY_SIZE(kApprovedAlgorithms); i++) {
    if (strcmp(algorithm, kApprovedAlgorithms[i]) == 0) {
      return 1;
    }
  }
#endif  // BORINGSSL_FIPS

  return 0;
}

#if defined(BORINGSSL_FIPS_COUNTERS)

size_t FIPS_read_counter(enum fips_counter_t counter) {
  if (counter < 0 || counter > fips_counter_max) {
    abort();
  }

  const size_t *array =
      CRYPTO_get_thread_local(OPENSSL_THREAD_LOCAL_FIPS_COUNTERS);
  if (!array) {
    return 0;
  }

  return array[counter];
}

void boringssl_fips_inc_counter(enum fips_counter_t counter) {
  if (counter < 0 || counter > fips_counter_max) {
    abort();
  }

  size_t *array =
      CRYPTO_get_thread_local(OPENSSL_THREAD_LOCAL_FIPS_COUNTERS);
  if (!array) {
    const size_t num_bytes = sizeof(size_t) * (fips_counter_max + 1);
    array = OPENSSL_malloc(num_bytes);
    if (!array) {
      return;
    }

    OPENSSL_memset(array, 0, num_bytes);
    if (!CRYPTO_set_thread_local(OPENSSL_THREAD_LOCAL_FIPS_COUNTERS, array,
                                 OPENSSL_free)) {
      // |OPENSSL_free| has already been called by |CRYPTO_set_thread_local|.
      return;
    }
  }

  array[counter]++;
}

#else

size_t FIPS_read_counter(enum fips_counter_t counter) { return 0; }

// boringssl_fips_inc_counter is a no-op, inline function in internal.h in this
// case. That should let the compiler optimise away the callsites.

#endif
