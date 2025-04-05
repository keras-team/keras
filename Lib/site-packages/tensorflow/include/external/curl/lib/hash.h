#ifndef HEADER_CURL_HASH_H
#define HEADER_CURL_HASH_H
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

#include <stddef.h>

#include "llist.h"

/* Hash function prototype */
typedef size_t (*hash_function) (void *key,
                                 size_t key_length,
                                 size_t slots_num);

/*
   Comparator function prototype. Compares two keys.
*/
typedef size_t (*comp_function) (void *key1,
                                 size_t key1_len,
                                 void *key2,
                                 size_t key2_len);

typedef void (*Curl_hash_dtor)(void *);

struct Curl_hash {
  struct Curl_llist *table;

  /* Hash function to be used for this hash table */
  hash_function hash_func;

  /* Comparator function to compare keys */
  comp_function comp_func;
  Curl_hash_dtor   dtor;
  size_t slots;
  size_t size;
#ifdef DEBUGBUILD
  int init;
#endif
};

typedef void (*Curl_hash_elem_dtor)(void *key, size_t key_len, void *p);

struct Curl_hash_element {
  struct Curl_llist_node list;
  void   *ptr;
  Curl_hash_elem_dtor dtor;
  size_t key_len;
#ifdef DEBUGBUILD
  int init;
#endif
  char   key[1]; /* allocated memory following the struct */
};

struct Curl_hash_iterator {
  struct Curl_hash *hash;
  size_t slot_index;
  struct Curl_llist_node *current_element;
#ifdef DEBUGBUILD
  int init;
#endif
};

void Curl_hash_init(struct Curl_hash *h,
                    size_t slots,
                    hash_function hfunc,
                    comp_function comparator,
                    Curl_hash_dtor dtor);

void *Curl_hash_add(struct Curl_hash *h, void *key, size_t key_len, void *p);
void *Curl_hash_add2(struct Curl_hash *h, void *key, size_t key_len, void *p,
                     Curl_hash_elem_dtor dtor);
int Curl_hash_delete(struct Curl_hash *h, void *key, size_t key_len);
void *Curl_hash_pick(struct Curl_hash *, void *key, size_t key_len);

void Curl_hash_destroy(struct Curl_hash *h);
size_t Curl_hash_count(struct Curl_hash *h);
void Curl_hash_clean(struct Curl_hash *h);
void Curl_hash_clean_with_criterium(struct Curl_hash *h, void *user,
                                    int (*comp)(void *, void *));
size_t Curl_hash_str(void *key, size_t key_length, size_t slots_num);
size_t Curl_str_key_compare(void *k1, size_t key1_len, void *k2,
                            size_t key2_len);
void Curl_hash_start_iterate(struct Curl_hash *hash,
                             struct Curl_hash_iterator *iter);
struct Curl_hash_element *
Curl_hash_next_element(struct Curl_hash_iterator *iter);

void Curl_hash_print(struct Curl_hash *h,
                     void (*func)(void *));

/* Hash for `curl_off_t` as key */
void Curl_hash_offt_init(struct Curl_hash *h, size_t slots,
                         Curl_hash_dtor dtor);

void *Curl_hash_offt_set(struct Curl_hash *h, curl_off_t id, void *elem);
int Curl_hash_offt_remove(struct Curl_hash *h, curl_off_t id);
void *Curl_hash_offt_get(struct Curl_hash *h, curl_off_t id);


#endif /* HEADER_CURL_HASH_H */
