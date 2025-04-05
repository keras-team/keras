#ifndef HEADER_CURL_CONFIG_WIN32_H
#define HEADER_CURL_CONFIG_WIN32_H
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

/* ================================================================ */
/*               Hand crafted config file for Windows               */
/* ================================================================ */

/* ---------------------------------------------------------------- */
/*                          HEADER FILES                            */
/* ---------------------------------------------------------------- */

/* Define if you have the <arpa/inet.h> header file. */
/* #define HAVE_ARPA_INET_H 1 */

/* Define if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define if you have the <io.h> header file. */
#define HAVE_IO_H 1

/* Define if you have the <locale.h> header file. */
#define HAVE_LOCALE_H 1

/* Define if you need <malloc.h> header even with <stdlib.h> header file. */
#define NEED_MALLOC_H 1

/* Define if you have the <netdb.h> header file. */
/* #define HAVE_NETDB_H 1 */

/* Define if you have the <netinet/in.h> header file. */
/* #define HAVE_NETINET_IN_H 1 */

/* Define to 1 if you have the <stdbool.h> header file. */
#if (defined(_MSC_VER) && (_MSC_VER >= 1800)) || defined(__MINGW32__)
#define HAVE_STDBOOL_H 1
#endif

/* Define if you have the <sys/param.h> header file. */
#if defined(__MINGW32__)
#define HAVE_SYS_PARAM_H 1
#endif

/* Define if you have the <sys/select.h> header file. */
/* #define HAVE_SYS_SELECT_H 1 */

/* Define if you have the <sys/socket.h> header file. */
/* #define HAVE_SYS_SOCKET_H 1 */

/* Define if you have the <sys/sockio.h> header file. */
/* #define HAVE_SYS_SOCKIO_H 1 */

/* Define if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define if you have the <sys/time.h> header file. */
#if defined(__MINGW32__)
#define HAVE_SYS_TIME_H 1
#endif

/* Define if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define if you have the <sys/utime.h> header file. */
#define HAVE_SYS_UTIME_H 1

/* Define if you have the <termio.h> header file. */
/* #define HAVE_TERMIO_H 1 */

/* Define if you have the <termios.h> header file. */
/* #define HAVE_TERMIOS_H 1 */

/* Define if you have the <unistd.h> header file. */
#if defined(__MINGW32__)
#define HAVE_UNISTD_H 1
#endif

/* Define to 1 if you have the <libgen.h> header file. */
#if defined(__MINGW32__)
#define HAVE_LIBGEN_H 1
#endif

/* ---------------------------------------------------------------- */
/*                        OTHER HEADER INFO                         */
/* ---------------------------------------------------------------- */

/* Define if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to 1 if bool is an available type. */
#if (defined(_MSC_VER) && (_MSC_VER >= 1800)) || defined(__MINGW32__)
#define HAVE_BOOL_T 1
#endif

/* ---------------------------------------------------------------- */
/*                             FUNCTIONS                            */
/* ---------------------------------------------------------------- */

/* Define if you have the closesocket function. */
#define HAVE_CLOSESOCKET 1

/* Define if you have the ftruncate function. */
#if defined(__MINGW32__)
#define HAVE_FTRUNCATE 1
#endif

/* Define to 1 if you have the `getpeername' function. */
#define HAVE_GETPEERNAME 1

/* Define to 1 if you have the getsockname function. */
#define HAVE_GETSOCKNAME 1

/* Define if you have the gethostname function. */
#define HAVE_GETHOSTNAME 1

/* Define if you have the gettimeofday function. */
#if defined(__MINGW32__)
#define HAVE_GETTIMEOFDAY 1
#endif

/* Define if you have the ioctlsocket function. */
#define HAVE_IOCTLSOCKET 1

/* Define if you have a working ioctlsocket FIONBIO function. */
#define HAVE_IOCTLSOCKET_FIONBIO 1

/* Define if you have the select function. */
#define HAVE_SELECT 1

/* Define if you have the setlocale function. */
#define HAVE_SETLOCALE 1

/* Define if you have the setmode function. */
#define HAVE_SETMODE 1

/* Define if you have the _setmode function. */
#define HAVE__SETMODE 1

/* Define if you have the socket function. */
#define HAVE_SOCKET 1

/* Define if you have the strcasecmp function. */
#if defined(__MINGW32__)
#define HAVE_STRCASECMP 1
#endif

/* Define if you have the strdup function. */
#define HAVE_STRDUP 1

/* Define if you have the stricmp function. */
#define HAVE_STRICMP 1

/* Define if you have the strtoll function. */
#if (defined(_MSC_VER) && (_MSC_VER >= 1800)) || defined(__MINGW32__)
#define HAVE_STRTOLL 1
#endif

/* Define if you have the utime function. */
#define HAVE_UTIME 1

/* Define if you have the recv function. */
#define HAVE_RECV 1

/* Define to the type of arg 1 for recv. */
#define RECV_TYPE_ARG1 SOCKET

/* Define to the type of arg 2 for recv. */
#define RECV_TYPE_ARG2 char *

/* Define to the type of arg 3 for recv. */
#define RECV_TYPE_ARG3 int

/* Define to the type of arg 4 for recv. */
#define RECV_TYPE_ARG4 int

/* Define to the function return type for recv. */
#define RECV_TYPE_RETV int

/* Define if you have the send function. */
#define HAVE_SEND 1

/* Define to the type of arg 1 for send. */
#define SEND_TYPE_ARG1 SOCKET

/* Define to the type qualifier of arg 2 for send. */
#define SEND_QUAL_ARG2 const

/* Define to the type of arg 2 for send. */
#define SEND_TYPE_ARG2 char *

/* Define to the type of arg 3 for send. */
#define SEND_TYPE_ARG3 int

/* Define to the type of arg 4 for send. */
#define SEND_TYPE_ARG4 int

/* Define to the function return type for send. */
#define SEND_TYPE_RETV int

/* Define to 1 if you have the snprintf function. */
#if (defined(_MSC_VER) && (_MSC_VER >= 1900)) || defined(__MINGW32__)
#define HAVE_SNPRINTF 1
#endif

#if defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x600  /* Vista */
/* Define to 1 if you have a IPv6 capable working inet_ntop function. */
#define HAVE_INET_NTOP 1
/* Define to 1 if you have a IPv6 capable working inet_pton function. */
#define HAVE_INET_PTON 1
#endif

/* Define to 1 if you have the `basename' function. */
#if defined(__MINGW32__)
#define HAVE_BASENAME 1
#endif

/* Define to 1 if you have the strtok_r function. */
#if defined(__MINGW32__)
#define HAVE_STRTOK_R 1
#endif

/* Define to 1 if you have the signal function. */
#define HAVE_SIGNAL 1

/* ---------------------------------------------------------------- */
/*                       TYPEDEF REPLACEMENTS                       */
/* ---------------------------------------------------------------- */

/* Define if in_addr_t is not an available 'typedefed' type. */
#define in_addr_t unsigned long

/* Define if ssize_t is not an available 'typedefed' type. */
#ifndef _SSIZE_T_DEFINED
#  if defined(__MINGW32__)
#  elif defined(_WIN64)
#    define _SSIZE_T_DEFINED
#    define ssize_t __int64
#  else
#    define _SSIZE_T_DEFINED
#    define ssize_t int
#  endif
#endif

/* ---------------------------------------------------------------- */
/*                            TYPE SIZES                            */
/* ---------------------------------------------------------------- */

/* Define to the size of `int', as computed by sizeof. */
#define SIZEOF_INT 4

/* Define to the size of `long long', as computed by sizeof. */
/* #define SIZEOF_LONG_LONG 8 */

/* Define to the size of `long', as computed by sizeof. */
#define SIZEOF_LONG 4

/* Define to the size of `size_t', as computed by sizeof. */
#if defined(_WIN64)
#  define SIZEOF_SIZE_T 8
#else
#  define SIZEOF_SIZE_T 4
#endif

/* Define to the size of `curl_off_t', as computed by sizeof. */
#define SIZEOF_CURL_OFF_T 8

/* ---------------------------------------------------------------- */
/*                        COMPILER SPECIFIC                         */
/* ---------------------------------------------------------------- */

/* Define to nothing if compiler does not support 'const' qualifier. */
/* #define const */

/* Define to nothing if compiler does not support 'volatile' qualifier. */
/* #define volatile */

/* Windows should not have HAVE_GMTIME_R defined */
/* #undef HAVE_GMTIME_R */

/* Define if the compiler supports the 'long long' data type. */
#if (defined(_MSC_VER) && (_MSC_VER >= 1310)) || defined(__MINGW32__)
#define HAVE_LONGLONG 1
#endif

/* Define to avoid VS2005 complaining about portable C functions. */
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
#define _CRT_SECURE_NO_DEPRECATE 1
#define _CRT_NONSTDC_NO_DEPRECATE 1
#endif

/* mingw-w64 and visual studio >= 2005 (MSVCR80)
   all default to 64-bit time_t unless _USE_32BIT_TIME_T is defined */
#if (defined(_MSC_VER) && (_MSC_VER >= 1400)) || defined(__MINGW32__)
#  ifndef _USE_32BIT_TIME_T
#    define SIZEOF_TIME_T 8
#  else
#    define SIZEOF_TIME_T 4
#  endif
#endif

/* Define some minimum and default build targets for Visual Studio */
#if defined(_MSC_VER)
   /* Officially, Microsoft's Windows SDK versions 6.X does not support Windows
      2000 as a supported build target. VS2008 default installations provides
      an embedded Windows SDK v6.0A along with the claim that Windows 2000 is a
      valid build target for VS2008. Popular belief is that binaries built with
      VS2008 using Windows SDK versions v6.X and Windows 2000 as a build target
      are functional. */
#  define VS2008_MIN_TARGET 0x0500

   /* The minimum build target for VS2012 is Vista unless Update 1 is installed
      and the v110_xp toolset is chosen. */
#  if defined(_USING_V110_SDK71_)
#    define VS2012_MIN_TARGET 0x0501
#  else
#    define VS2012_MIN_TARGET 0x0600
#  endif

   /* VS2008 default build target is Windows Vista. We override default target
      to be Windows XP. */
#  define VS2008_DEF_TARGET 0x0501

   /* VS2012 default build target is Windows Vista unless Update 1 is installed
      and the v110_xp toolset is chosen. */
#  if defined(_USING_V110_SDK71_)
#    define VS2012_DEF_TARGET 0x0501
#  else
#    define VS2012_DEF_TARGET 0x0600
#  endif
#endif

/* VS2008 default target settings and minimum build target check. */
#if defined(_MSC_VER) && (_MSC_VER >= 1500) && (_MSC_VER <= 1600)
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT VS2008_DEF_TARGET
#  endif
#  ifndef WINVER
#    define WINVER VS2008_DEF_TARGET
#  endif
#  if (_WIN32_WINNT < VS2008_MIN_TARGET) || (WINVER < VS2008_MIN_TARGET)
#    error VS2008 does not support Windows build targets prior to Windows 2000
#  endif
#endif

/* VS2012 default target settings and minimum build target check. */
#if defined(_MSC_VER) && (_MSC_VER >= 1700)
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT VS2012_DEF_TARGET
#  endif
#  ifndef WINVER
#    define WINVER VS2012_DEF_TARGET
#  endif
#  if (_WIN32_WINNT < VS2012_MIN_TARGET) || (WINVER < VS2012_MIN_TARGET)
#    if defined(_USING_V110_SDK71_)
#      error VS2012 does not support Windows build targets prior to Windows XP
#    else
#      error VS2012 does not support Windows build targets prior to Windows \
Vista
#    endif
#  endif
#endif

/* Windows XP is required for freeaddrinfo, getaddrinfo */
#define HAVE_FREEADDRINFO           1
#define HAVE_GETADDRINFO            1
#define HAVE_GETADDRINFO_THREADSAFE 1

/* ---------------------------------------------------------------- */
/*                          STRUCT RELATED                          */
/* ---------------------------------------------------------------- */

/* Define if you have struct sockaddr_storage. */
#define HAVE_STRUCT_SOCKADDR_STORAGE 1

/* Define if you have struct timeval. */
#define HAVE_STRUCT_TIMEVAL 1

/* Define if struct sockaddr_in6 has the sin6_scope_id member. */
#define HAVE_SOCKADDR_IN6_SIN6_SCOPE_ID 1

/* ---------------------------------------------------------------- */
/*                        LARGE FILE SUPPORT                        */
/* ---------------------------------------------------------------- */

#if defined(_MSC_VER) && !defined(_WIN32_WCE)
#  if (_MSC_VER >= 900) && (_INTEGRAL_MAX_BITS >= 64)
#    define USE_WIN32_LARGE_FILES
#  else
#    define USE_WIN32_SMALL_FILES
#  endif
#endif

#if defined(__MINGW32__) && !defined(USE_WIN32_LARGE_FILES)
#  define USE_WIN32_LARGE_FILES
#endif

#if !defined(USE_WIN32_LARGE_FILES) && !defined(USE_WIN32_SMALL_FILES)
#  define USE_WIN32_SMALL_FILES
#endif

/* Number of bits in a file offset, on hosts where this is settable. */
#if defined(USE_WIN32_LARGE_FILES) && defined(__MINGW32__)
#  ifndef _FILE_OFFSET_BITS
#  define _FILE_OFFSET_BITS 64
#  endif
#endif

#ifdef USE_WIN32_LARGE_FILES
#define HAVE__FSEEKI64
#endif

/* Define to the size of `off_t', as computed by sizeof. */
#if defined(__MINGW32__) && \
  defined(_FILE_OFFSET_BITS) && (_FILE_OFFSET_BITS == 64)
#  define SIZEOF_OFF_T 8
#else
#  define SIZEOF_OFF_T 4
#endif

/* ---------------------------------------------------------------- */
/*                       DNS RESOLVER SPECIALTY                     */
/* ---------------------------------------------------------------- */

/*
 * Undefine both USE_ARES and USE_THREADS_WIN32 for synchronous DNS.
 */

/* Define to enable c-ares asynchronous DNS lookups. */
/* #define USE_ARES 1 */

/* Default define to enable threaded asynchronous DNS lookups. */
#if !defined(USE_SYNC_DNS) && !defined(USE_ARES) && \
    !defined(USE_THREADS_WIN32)
#  define USE_THREADS_WIN32 1
#endif

#if defined(USE_ARES) && defined(USE_THREADS_WIN32)
#  error "Only one DNS lookup specialty may be defined at most"
#endif

/* ---------------------------------------------------------------- */
/*                           LDAP SUPPORT                           */
/* ---------------------------------------------------------------- */

#if defined(CURL_HAS_NOVELL_LDAPSDK)
#undef USE_WIN32_LDAP
#define HAVE_LDAP_SSL_H 1
#define HAVE_LDAP_URL_PARSE 1
#elif defined(CURL_HAS_OPENLDAP_LDAPSDK)
#undef USE_WIN32_LDAP
#define HAVE_LDAP_URL_PARSE 1
#else
#undef HAVE_LDAP_URL_PARSE
#define HAVE_LDAP_SSL 1
#define USE_WIN32_LDAP 1
#endif

/* Define to use the Windows crypto library. */
#if !defined(CURL_WINDOWS_UWP)
#define USE_WIN32_CRYPTO
#endif

/* Define to use Unix sockets. */
#define USE_UNIX_SOCKETS

/* ---------------------------------------------------------------- */
/*                       ADDITIONAL DEFINITIONS                     */
/* ---------------------------------------------------------------- */

/* Define cpu-machine-OS */
#ifndef CURL_OS
#if defined(_M_IX86) || defined(__i386__) /* x86 (MSVC or gcc) */
#define CURL_OS "i386-pc-win32"
#elif defined(_M_X64) || defined(__x86_64__) /* x86_64 (MSVC >=2005 or gcc) */
#define CURL_OS "x86_64-pc-win32"
#elif defined(_M_IA64) || defined(__ia64__) /* Itanium */
#define CURL_OS "ia64-pc-win32"
#elif defined(_M_ARM_NT) || defined(__arm__) /* ARMv7-Thumb2 (Windows RT) */
#define CURL_OS "thumbv7a-pc-win32"
#elif defined(_M_ARM64) || defined(__aarch64__) /* ARM64 (Windows 10) */
#define CURL_OS "aarch64-pc-win32"
#else
#define CURL_OS "unknown-pc-win32"
#endif
#endif

/* Name of package */
#define PACKAGE "curl"

/* If you want to build curl with the built-in manual */
#define USE_MANUAL 1

#endif /* HEADER_CURL_CONFIG_WIN32_H */
