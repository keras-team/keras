#define HAVE_STDDEF_H 1
#define HAVE_STDINT_H 1

#ifdef __has_builtin
#  if !defined(HAVE_BUILTIN_EXPECT) && __has_builtin(__builtin_expect)
#    define HAVE_BUILTIN_EXPECT 1
#  endif
#  if !defined(HAVE_BUILTIN_CTZ) && __has_builtin(__builtin_ctzll)
#    define HAVE_BUILTIN_CTZ 1
#  endif
#elif defined(__GNUC__) && (__GNUC__ > 3 || __GNUC__ == 3 && __GNUC_MINOR__ >= 4)
#  ifndef HAVE_BUILTIN_EXPECT
#    define HAVE_BUILTIN_EXPECT 1
#  endif
#  ifndef HAVE_BUILTIN_CTZ
#    define HAVE_BUILTIN_CTZ 1
#  endif
#endif

#ifdef __has_include
#  if !defined(HAVE_BYTESWAP_H) && __has_include(<byteswap.h>)
#    define HAVE_BYTESWAP_H 1
#  endif
#  if !defined(HAVE_UNISTD_H) && __has_include(<unistd.h>)
#    define HAVE_UNISTD_H 1
#  endif
#  if !defined(HAVE_SYS_ENDIAN_H) && __has_include(<sys/endian.h>)
#    define HAVE_SYS_ENDIAN_H 1
#  endif
#  if !defined(HAVE_SYS_MMAN_H) && __has_include(<sys/mman.h>)
#    define HAVE_SYS_MMAN_H 1
#  endif
#  if !defined(HAVE_SYS_UIO_H) && __has_include(<sys/uio.h>)
#    define HAVE_SYS_UIO_H 1
#  endif
#endif

#ifndef SNAPPY_IS_BIG_ENDIAN
#  ifdef __s390x__
#    define SNAPPY_IS_BIG_ENDIAN 1
#  elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#    define SNAPPY_IS_BIG_ENDIAN 1
#  endif
#endif
