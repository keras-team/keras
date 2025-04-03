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

#ifndef OPENSSL_HEADER_CRYPTO_RAND_INTERNAL_H
#define OPENSSL_HEADER_CRYPTO_RAND_INTERNAL_H

#include <openssl/aes.h>
#include <openssl/ctrdrbg.h>

#include "../../internal.h"
#include "../modes/internal.h"

#if defined(__cplusplus)
extern "C" {
#endif


#if !defined(OPENSSL_WINDOWS) && !defined(OPENSSL_FUCHSIA) && \
    !defined(BORINGSSL_UNSAFE_DETERMINISTIC_MODE) && !defined(OPENSSL_TRUSTY)
#define OPENSSL_URANDOM
#endif

// RAND_bytes_with_additional_data samples from the RNG after mixing 32 bytes
// from |user_additional_data| in.
void RAND_bytes_with_additional_data(uint8_t *out, size_t out_len,
                                     const uint8_t user_additional_data[32]);

#if defined(BORINGSSL_FIPS)

// We overread from /dev/urandom or RDRAND by a factor of 10 and XOR to whiten.
#define BORINGSSL_FIPS_OVERREAD 10

// CRYPTO_get_seed_entropy writes |out_entropy_len| bytes of entropy, suitable
// for seeding a DRBG, to |out_entropy|. It sets |*out_used_cpu| to one if the
// entropy came directly from the CPU and zero if it came from the OS. It
// actively obtains entropy from the CPU/OS and so should not be called from
// within the FIPS module.
void CRYPTO_get_seed_entropy(uint8_t *out_entropy, size_t out_entropy_len,
                             int *out_used_cpu);

// RAND_load_entropy supplies |entropy_len| bytes of entropy to the module. The
// |want_additional_input| parameter is true iff the entropy was obtained from
// a source other than the system, e.g. directly from the CPU.
void RAND_load_entropy(const uint8_t *entropy, size_t entropy_len,
                       int want_additional_input);

// RAND_need_entropy is implemented outside of the FIPS module and is called
// when the module has stopped because it has run out of entropy.
void RAND_need_entropy(size_t bytes_needed);

#endif  // BORINGSSL_FIPS

// CRYPTO_sysrand fills |len| bytes at |buf| with entropy from the operating
// system.
void CRYPTO_sysrand(uint8_t *buf, size_t len);

// CRYPTO_sysrand_for_seed fills |len| bytes at |buf| with entropy from the
// operating system. It may draw from the |GRND_RANDOM| pool on Android,
// depending on the vendor's configuration.
void CRYPTO_sysrand_for_seed(uint8_t *buf, size_t len);

#if defined(OPENSSL_URANDOM)
// CRYPTO_init_sysrand initializes long-lived resources needed to draw entropy
// from the operating system.
void CRYPTO_init_sysrand(void);

// CRYPTO_sysrand_if_available fills |len| bytes at |buf| with entropy from the
// operating system, or early /dev/urandom data, and returns 1, _if_ the entropy
// pool is initialized or if getrandom() is not available and not in FIPS mode.
// Otherwise it will not block and will instead fill |buf| with all zeros and
// return 0.
int CRYPTO_sysrand_if_available(uint8_t *buf, size_t len);
#else
OPENSSL_INLINE void CRYPTO_init_sysrand(void) {}

OPENSSL_INLINE int CRYPTO_sysrand_if_available(uint8_t *buf, size_t len) {
  CRYPTO_sysrand(buf, len);
  return 1;
}
#endif

// rand_fork_unsafe_buffering_enabled returns whether fork-unsafe buffering has
// been enabled via |RAND_enable_fork_unsafe_buffering|.
int rand_fork_unsafe_buffering_enabled(void);

// CTR_DRBG_STATE contains the state of a CTR_DRBG based on AES-256. See SP
// 800-90Ar1.
struct ctr_drbg_state_st {
  AES_KEY ks;
  block128_f block;
  ctr128_f ctr;
  uint8_t counter[16];
  uint64_t reseed_counter;
};

// CTR_DRBG_init initialises |*drbg| given |CTR_DRBG_ENTROPY_LEN| bytes of
// entropy in |entropy| and, optionally, a personalization string up to
// |CTR_DRBG_ENTROPY_LEN| bytes in length. It returns one on success and zero
// on error.
OPENSSL_EXPORT int CTR_DRBG_init(CTR_DRBG_STATE *drbg,
                                 const uint8_t entropy[CTR_DRBG_ENTROPY_LEN],
                                 const uint8_t *personalization,
                                 size_t personalization_len);

#if defined(OPENSSL_X86_64) && !defined(OPENSSL_NO_ASM)

OPENSSL_INLINE int have_rdrand(void) {
  return CRYPTO_is_RDRAND_capable();
}

// have_fast_rdrand returns true if RDRAND is supported and it's reasonably
// fast. Concretely the latter is defined by whether the chip is Intel (fast) or
// not (assumed slow).
OPENSSL_INLINE int have_fast_rdrand(void) {
  return CRYPTO_is_RDRAND_capable() && CRYPTO_is_intel_cpu();
}

// CRYPTO_rdrand writes eight bytes of random data from the hardware RNG to
// |out|. It returns one on success or zero on hardware failure.
int CRYPTO_rdrand(uint8_t out[8]);

// CRYPTO_rdrand_multiple8_buf fills |len| bytes at |buf| with random data from
// the hardware RNG. The |len| argument must be a multiple of eight. It returns
// one on success and zero on hardware failure.
int CRYPTO_rdrand_multiple8_buf(uint8_t *buf, size_t len);

#else  // OPENSSL_X86_64 && !OPENSSL_NO_ASM

OPENSSL_INLINE int have_rdrand(void) {
  return 0;
}

OPENSSL_INLINE int have_fast_rdrand(void) {
  return 0;
}

#endif  // OPENSSL_X86_64 && !OPENSSL_NO_ASM


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_RAND_INTERNAL_H
