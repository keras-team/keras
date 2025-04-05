/* Copyright (c) 2020, Google Inc.
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

#ifndef OPENSSL_HEADER_CRYPTO_RAND_GETRANDOM_FILLIN_H
#define OPENSSL_HEADER_CRYPTO_RAND_GETRANDOM_FILLIN_H

#include <openssl/base.h>


#if defined(OPENSSL_LINUX)

#include <sys/syscall.h>

#if defined(OPENSSL_X86_64)
#define EXPECTED_NR_getrandom 318
#elif defined(OPENSSL_X86)
#define EXPECTED_NR_getrandom 355
#elif defined(OPENSSL_AARCH64)
#define EXPECTED_NR_getrandom 278
#elif defined(OPENSSL_ARM)
#define EXPECTED_NR_getrandom 384
#elif defined(OPENSSL_RISCV64)
#define EXPECTED_NR_getrandom 278
#endif

#if defined(EXPECTED_NR_getrandom)
#define USE_NR_getrandom

#if defined(__NR_getrandom)

#if __NR_getrandom != EXPECTED_NR_getrandom
#error "system call number for getrandom is not the expected value"
#endif

#else  // __NR_getrandom

#define __NR_getrandom EXPECTED_NR_getrandom

#endif  // __NR_getrandom

#endif  // EXPECTED_NR_getrandom

#if !defined(GRND_NONBLOCK)
#define GRND_NONBLOCK 1
#endif
#if !defined(GRND_RANDOM)
#define GRND_RANDOM 2
#endif

#endif  // OPENSSL_LINUX


#endif  // OPENSSL_HEADER_CRYPTO_RAND_GETRANDOM_FILLIN_H
