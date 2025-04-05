#ifndef HEADER_CURL_SSPI_H
#define HEADER_CURL_SSPI_H
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

#include "curl_setup.h"

#ifdef USE_WINDOWS_SSPI

#include <curl/curl.h>

/*
 * When including the following three headers, it is mandatory to define either
 * SECURITY_WIN32 or SECURITY_KERNEL, indicating who is compiling the code.
 */

#undef SECURITY_WIN32
#undef SECURITY_KERNEL
#define SECURITY_WIN32 1
#include <security.h>
#include <sspi.h>
#include <rpc.h>

CURLcode Curl_sspi_global_init(void);
void Curl_sspi_global_cleanup(void);

/* This is used to populate the domain in a SSPI identity structure */
CURLcode Curl_override_sspi_http_realm(const char *chlg,
                                       SEC_WINNT_AUTH_IDENTITY *identity);

/* This is used to generate an SSPI identity structure */
CURLcode Curl_create_sspi_identity(const char *userp, const char *passwdp,
                                   SEC_WINNT_AUTH_IDENTITY *identity);

/* This is used to free an SSPI identity structure */
void Curl_sspi_free_identity(SEC_WINNT_AUTH_IDENTITY *identity);

/* Forward-declaration of global variables defined in curl_sspi.c */
extern HMODULE Curl_hSecDll;
extern PSecurityFunctionTable Curl_pSecFn;

/* Provide some definitions missing in old headers */
#define SP_NAME_DIGEST              "WDigest"
#define SP_NAME_NTLM                "NTLM"
#define SP_NAME_NEGOTIATE           "Negotiate"
#define SP_NAME_KERBEROS            "Kerberos"

#ifndef ISC_REQ_USE_HTTP_STYLE
#define ISC_REQ_USE_HTTP_STYLE                0x01000000
#endif

#ifndef SEC_E_INVALID_PARAMETER
# define SEC_E_INVALID_PARAMETER              ((HRESULT)0x8009035DL)
#endif
#ifndef SEC_E_DELEGATION_POLICY
# define SEC_E_DELEGATION_POLICY              ((HRESULT)0x8009035EL)
#endif
#ifndef SEC_E_POLICY_NLTM_ONLY
# define SEC_E_POLICY_NLTM_ONLY               ((HRESULT)0x8009035FL)
#endif

#ifndef SEC_I_SIGNATURE_NEEDED
# define SEC_I_SIGNATURE_NEEDED               ((HRESULT)0x0009035CL)
#endif

#ifndef CRYPT_E_REVOKED
# define CRYPT_E_REVOKED                      ((HRESULT)0x80092010L)
#endif

#ifndef CRYPT_E_NO_REVOCATION_DLL
# define CRYPT_E_NO_REVOCATION_DLL            ((HRESULT)0x80092011L)
#endif

#ifndef CRYPT_E_NO_REVOCATION_CHECK
# define CRYPT_E_NO_REVOCATION_CHECK          ((HRESULT)0x80092012L)
#endif

#ifndef CRYPT_E_REVOCATION_OFFLINE
# define CRYPT_E_REVOCATION_OFFLINE           ((HRESULT)0x80092013L)
#endif

#ifndef CRYPT_E_NOT_IN_REVOCATION_DATABASE
# define CRYPT_E_NOT_IN_REVOCATION_DATABASE   ((HRESULT)0x80092014L)
#endif

#ifdef UNICODE
#  define SECFLAG_WINNT_AUTH_IDENTITY \
     (unsigned long)SEC_WINNT_AUTH_IDENTITY_UNICODE
#else
#  define SECFLAG_WINNT_AUTH_IDENTITY \
     (unsigned long)SEC_WINNT_AUTH_IDENTITY_ANSI
#endif

/*
 * Definitions required from ntsecapi.h are directly provided below this point
 * to avoid including ntsecapi.h due to a conflict with OpenSSL's safestack.h
 */
#define KERB_WRAP_NO_ENCRYPT 0x80000001

#endif /* USE_WINDOWS_SSPI */

#endif /* HEADER_CURL_SSPI_H */
