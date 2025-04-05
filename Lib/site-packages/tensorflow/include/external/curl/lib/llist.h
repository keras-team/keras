#ifndef HEADER_CURL_LLIST_H
#define HEADER_CURL_LLIST_H
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

typedef void (*Curl_llist_dtor)(void *user, void *elem);

/* none of these struct members should be referenced directly, use the
   dedicated functions */

struct Curl_llist {
  struct Curl_llist_node *_head;
  struct Curl_llist_node *_tail;
  Curl_llist_dtor _dtor;
  size_t _size;
#ifdef DEBUGBUILD
  int _init;      /* detect API usage mistakes */
#endif
};

struct Curl_llist_node {
  struct Curl_llist *_list; /* the list where this belongs */
  void *_ptr;
  struct Curl_llist_node *_prev;
  struct Curl_llist_node *_next;
#ifdef DEBUGBUILD
  int _init;      /* detect API usage mistakes */
#endif
};

void Curl_llist_init(struct Curl_llist *, Curl_llist_dtor);
void Curl_llist_insert_next(struct Curl_llist *, struct Curl_llist_node *,
                            const void *, struct Curl_llist_node *node);
void Curl_llist_append(struct Curl_llist *,
                       const void *, struct Curl_llist_node *node);
void Curl_node_uremove(struct Curl_llist_node *, void *);
void Curl_node_remove(struct Curl_llist_node *);
void Curl_llist_destroy(struct Curl_llist *, void *);

/* Curl_llist_head() returns the first 'struct Curl_llist_node *', which
   might be NULL */
struct Curl_llist_node *Curl_llist_head(struct Curl_llist *list);

/* Curl_llist_tail() returns the last 'struct Curl_llist_node *', which
   might be NULL */
struct Curl_llist_node *Curl_llist_tail(struct Curl_llist *list);

/* Curl_llist_count() returns a size_t the number of nodes in the list */
size_t Curl_llist_count(struct Curl_llist *list);

/* Curl_node_elem() returns the custom data from a Curl_llist_node */
void *Curl_node_elem(struct Curl_llist_node *n);

/* Curl_node_next() returns the next element in a list from a given
   Curl_llist_node */
struct Curl_llist_node *Curl_node_next(struct Curl_llist_node *n);

/* Curl_node_prev() returns the previous element in a list from a given
   Curl_llist_node */
struct Curl_llist_node *Curl_node_prev(struct Curl_llist_node *n);

/* Curl_node_llist() return the list the node is in or NULL. */
struct Curl_llist *Curl_node_llist(struct Curl_llist_node *n);

#endif /* HEADER_CURL_LLIST_H */
