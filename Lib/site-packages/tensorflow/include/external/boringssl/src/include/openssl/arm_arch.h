/* ====================================================================
 * Copyright (c) 1998-2011 The OpenSSL Project.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. All advertising materials mentioning features or use of this
 *    software must display the following acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit. (http://www.openssl.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    openssl-core@openssl.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.openssl.org/)"
 *
 * THIS SOFTWARE IS PROVIDED BY THE OpenSSL PROJECT ``AS IS'' AND ANY
 * EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE OpenSSL PROJECT OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * ====================================================================
 *
 * This product includes cryptographic software written by Eric Young
 * (eay@cryptsoft.com).  This product includes software written by Tim
 * Hudson (tjh@cryptsoft.com). */

#ifndef OPENSSL_HEADER_ARM_ARCH_H
#define OPENSSL_HEADER_ARM_ARCH_H

// arm_arch.h contains symbols used by ARM assembly, and the C code that calls
// it. It is included as a public header to simplify the build, but is not
// intended for external use.

#if defined(__ARMEL__) || defined(_M_ARM) || defined(__AARCH64EL__) || \
    defined(_M_ARM64)

// ARMV7_NEON is true when a NEON unit is present in the current CPU.
#define ARMV7_NEON (1 << 0)

// ARMV8_AES indicates support for hardware AES instructions.
#define ARMV8_AES (1 << 2)

// ARMV8_SHA1 indicates support for hardware SHA-1 instructions.
#define ARMV8_SHA1 (1 << 3)

// ARMV8_SHA256 indicates support for hardware SHA-256 instructions.
#define ARMV8_SHA256 (1 << 4)

// ARMV8_PMULL indicates support for carryless multiplication.
#define ARMV8_PMULL (1 << 5)

// ARMV8_SHA512 indicates support for hardware SHA-512 instructions.
#define ARMV8_SHA512 (1 << 6)

#if defined(__ASSEMBLER__)

// We require the ARM assembler provide |__ARM_ARCH| from Arm C Language
// Extensions (ACLE). This is supported in GCC 4.8+ and Clang 3.2+. MSVC does
// not implement ACLE, but we require Clang's assembler on Windows.
#if !defined(__ARM_ARCH)
#error "ARM assembler must define __ARM_ARCH"
#endif

// __ARM_ARCH__ is used by OpenSSL assembly to determine the minimum target ARM
// version.
//
// TODO(davidben): Switch the assembly to use |__ARM_ARCH| directly.
#define __ARM_ARCH__ __ARM_ARCH

// Even when building for 32-bit ARM, support for aarch64 crypto instructions
// will be included.
#define __ARM_MAX_ARCH__ 8

// Support macros for
//   - Armv8.3-A Pointer Authentication and
//   - Armv8.5-A Branch Target Identification
// features which require emitting a .note.gnu.property section with the
// appropriate architecture-dependent feature bits set.
//
// |AARCH64_SIGN_LINK_REGISTER| and |AARCH64_VALIDATE_LINK_REGISTER| expand to
// PACIxSP and AUTIxSP, respectively. |AARCH64_SIGN_LINK_REGISTER| should be
// used immediately before saving the LR register (x30) to the stack.
// |AARCH64_VALIDATE_LINK_REGISTER| should be used immediately after restoring
// it. Note |AARCH64_SIGN_LINK_REGISTER|'s modifications to LR must be undone
// with |AARCH64_VALIDATE_LINK_REGISTER| before RET. The SP register must also
// have the same value at the two points. For example:
//
//   .global f
//   f:
//     AARCH64_SIGN_LINK_REGISTER
//     stp x29, x30, [sp, #-96]!
//     mov x29, sp
//     ...
//     ldp x29, x30, [sp], #96
//     AARCH64_VALIDATE_LINK_REGISTER
//     ret
//
// |AARCH64_VALID_CALL_TARGET| expands to BTI 'c'. Either it, or
// |AARCH64_SIGN_LINK_REGISTER|, must be used at every point that may be an
// indirect call target. In particular, all symbols exported from a file must
// begin with one of these macros. For example, a leaf function that does not
// save LR can instead use |AARCH64_VALID_CALL_TARGET|:
//
//   .globl return_zero
//   return_zero:
//     AARCH64_VALID_CALL_TARGET
//     mov x0, #0
//     ret
//
// A non-leaf function which does not immediately save LR may need both macros
// because |AARCH64_SIGN_LINK_REGISTER| appears late. For example, the function
// may jump to an alternate implementation before setting up the stack:
//
//   .globl with_early_jump
//   with_early_jump:
//     AARCH64_VALID_CALL_TARGET
//     cmp x0, #128
//     b.lt .Lwith_early_jump_128
//     AARCH64_SIGN_LINK_REGISTER
//     stp x29, x30, [sp, #-96]!
//     mov x29, sp
//     ...
//     ldp x29, x30, [sp], #96
//     AARCH64_VALIDATE_LINK_REGISTER
//     ret
//
//  .Lwith_early_jump_128:
//     ...
//     ret
//
// These annotations are only required with indirect calls. Private symbols that
// are only the target of direct calls do not require annotations. Also note
// that |AARCH64_VALID_CALL_TARGET| is only valid for indirect calls (BLR), not
// indirect jumps (BR). Indirect jumps in assembly are currently not supported
// and would require a macro for BTI 'j'.
//
// Although not necessary, it is safe to use these macros in 32-bit ARM
// assembly. This may be used to simplify dual 32-bit and 64-bit files.
//
// References:
// - "ELF for the Arm® 64-bit Architecture"
//   https://github.com/ARM-software/abi-aa/blob/master/aaelf64/aaelf64.rst
// - "Providing protection for complex software"
//   https://developer.arm.com/architectures/learn-the-architecture/providing-protection-for-complex-software

#if defined(__ARM_FEATURE_BTI_DEFAULT) && __ARM_FEATURE_BTI_DEFAULT == 1
#define GNU_PROPERTY_AARCH64_BTI (1 << 0)   // Has Branch Target Identification
#define AARCH64_VALID_CALL_TARGET hint #34  // BTI 'c'
#else
#define GNU_PROPERTY_AARCH64_BTI 0  // No Branch Target Identification
#define AARCH64_VALID_CALL_TARGET
#endif

#if defined(__ARM_FEATURE_PAC_DEFAULT) && \
    (__ARM_FEATURE_PAC_DEFAULT & 1) == 1  // Signed with A-key
#define GNU_PROPERTY_AARCH64_POINTER_AUTH \
  (1 << 1)                                       // Has Pointer Authentication
#define AARCH64_SIGN_LINK_REGISTER hint #25      // PACIASP
#define AARCH64_VALIDATE_LINK_REGISTER hint #29  // AUTIASP
#elif defined(__ARM_FEATURE_PAC_DEFAULT) && \
    (__ARM_FEATURE_PAC_DEFAULT & 2) == 2  // Signed with B-key
#define GNU_PROPERTY_AARCH64_POINTER_AUTH \
  (1 << 1)                                       // Has Pointer Authentication
#define AARCH64_SIGN_LINK_REGISTER hint #27      // PACIBSP
#define AARCH64_VALIDATE_LINK_REGISTER hint #31  // AUTIBSP
#else
#define GNU_PROPERTY_AARCH64_POINTER_AUTH 0  // No Pointer Authentication
#if GNU_PROPERTY_AARCH64_BTI != 0
#define AARCH64_SIGN_LINK_REGISTER AARCH64_VALID_CALL_TARGET
#else
#define AARCH64_SIGN_LINK_REGISTER
#endif
#define AARCH64_VALIDATE_LINK_REGISTER
#endif

#if GNU_PROPERTY_AARCH64_POINTER_AUTH != 0 || GNU_PROPERTY_AARCH64_BTI != 0
.pushsection .note.gnu.property, "a";
.balign 8;
.long 4;
.long 0x10;
.long 0x5;
.asciz "GNU";
.long 0xc0000000; /* GNU_PROPERTY_AARCH64_FEATURE_1_AND */
.long 4;
.long (GNU_PROPERTY_AARCH64_POINTER_AUTH | GNU_PROPERTY_AARCH64_BTI);
.long 0;
.popsection;
#endif

#endif  // __ASSEMBLER__

#endif  // __ARMEL__ || _M_ARM || __AARCH64EL__ || _M_ARM64

#endif  // OPENSSL_HEADER_ARM_ARCH_H
