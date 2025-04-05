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

#ifndef GRPC_CORE_LIB_SECURITY_CREDENTIALS_JWT_JSON_TOKEN_H
#define GRPC_CORE_LIB_SECURITY_CREDENTIALS_JWT_JSON_TOKEN_H

#include <grpc/support/port_platform.h>

#include "src/core/tsi/grpc_shadow_boringssl.h"

#include <grpc/slice.h>
#include <openssl/rsa.h>

#include "src/core/lib/json/json.h"

/* --- Constants. --- */

#define GRPC_JWT_OAUTH2_AUDIENCE "https://oauth2.googleapis.com/token"

/* --- auth_json_key parsing. --- */

typedef struct {
  const char* type;
  char* private_key_id;
  char* client_id;
  char* client_email;
  RSA* private_key;
} grpc_auth_json_key;

/* Returns 1 if the object is valid, 0 otherwise. */
int grpc_auth_json_key_is_valid(const grpc_auth_json_key* json_key);

/* Creates a json_key object from string. Returns an invalid object if a parsing
   error has been encountered. */
grpc_auth_json_key grpc_auth_json_key_create_from_string(
    const char* json_string);

/* Creates a json_key object from parsed json. Returns an invalid object if a
   parsing error has been encountered. */
grpc_auth_json_key grpc_auth_json_key_create_from_json(const grpc_json* json);

/* Destructs the object. */
void grpc_auth_json_key_destruct(grpc_auth_json_key* json_key);

/* --- json token encoding and signing. --- */

/* Caller is responsible for calling gpr_free on the returned value. May return
   NULL on invalid input. The scope parameter may be NULL. */
char* grpc_jwt_encode_and_sign(const grpc_auth_json_key* json_key,
                               const char* audience,
                               gpr_timespec token_lifetime, const char* scope);

/* Override encode_and_sign function for testing. */
typedef char* (*grpc_jwt_encode_and_sign_override)(
    const grpc_auth_json_key* json_key, const char* audience,
    gpr_timespec token_lifetime, const char* scope);

/* Set a custom encode_and_sign override for testing. */
void grpc_jwt_encode_and_sign_set_override(
    grpc_jwt_encode_and_sign_override func);

#endif /* GRPC_CORE_LIB_SECURITY_CREDENTIALS_JWT_JSON_TOKEN_H */
