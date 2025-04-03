/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef GRPC_CORE_LIB_AVL_AVL_H
#define GRPC_CORE_LIB_AVL_AVL_H

#include <grpc/support/port_platform.h>

#include <grpc/support/sync.h>

/** internal node of an AVL tree */
typedef struct grpc_avl_node {
  gpr_refcount refs;
  void* key;
  void* value;
  struct grpc_avl_node* left;
  struct grpc_avl_node* right;
  long height;
} grpc_avl_node;

/** vtable for the AVL tree
 * The optional user_data is propagated from the top level grpc_avl_XXX API.
 * From the same API call, multiple vtable functions may be called multiple
 * times.
 */
typedef struct grpc_avl_vtable {
  /** destroy a key */
  void (*destroy_key)(void* key, void* user_data);
  /** copy a key, returning new value */
  void* (*copy_key)(void* key, void* user_data);
  /** compare key1, key2; return <0 if key1 < key2,
      >0 if key1 > key2, 0 if key1 == key2 */
  long (*compare_keys)(void* key1, void* key2, void* user_data);
  /** destroy a value */
  void (*destroy_value)(void* value, void* user_data);
  /** copy a value */
  void* (*copy_value)(void* value, void* user_data);
} grpc_avl_vtable;

/** "pointer" to an AVL tree - this is a reference
    counted object - use grpc_avl_ref to add a reference,
    grpc_avl_unref when done with a reference */
typedef struct grpc_avl {
  const grpc_avl_vtable* vtable;
  grpc_avl_node* root;
} grpc_avl;

/** Create an immutable AVL tree. */
grpc_avl grpc_avl_create(const grpc_avl_vtable* vtable);
/** Add a reference to an existing tree - returns
    the tree as a convenience. The optional user_data will be passed to vtable
    functions. */
grpc_avl grpc_avl_ref(grpc_avl avl, void* user_data);
/** Remove a reference to a tree - destroying it if there
    are no references left. The optional user_data will be passed to vtable
    functions. */
void grpc_avl_unref(grpc_avl avl, void* user_data);
/** Return a new tree with (key, value) added to avl.
    implicitly unrefs avl to allow easy chaining.
    if key exists in avl, the new tree's key entry updated
    (i.e. a duplicate is not created). The optional user_data will be passed to
    vtable functions. */
grpc_avl grpc_avl_add(grpc_avl avl, void* key, void* value, void* user_data);
/** Return a new tree with key deleted
    implicitly unrefs avl to allow easy chaining. The optional user_data will be
    passed to vtable functions. */
grpc_avl grpc_avl_remove(grpc_avl avl, void* key, void* user_data);
/** Lookup key, and return the associated value.
    Does not mutate avl.
    Returns NULL if key is not found. The optional user_data will be passed to
    vtable functions.*/
void* grpc_avl_get(grpc_avl avl, void* key, void* user_data);
/** Return 1 if avl contains key, 0 otherwise; if it has the key, sets *value to
    its value. The optional user_data will be passed to vtable functions. */
int grpc_avl_maybe_get(grpc_avl avl, void* key, void** value, void* user_data);
/** Return 1 if avl is empty, 0 otherwise */
int grpc_avl_is_empty(grpc_avl avl);

#endif /* GRPC_CORE_LIB_AVL_AVL_H */
