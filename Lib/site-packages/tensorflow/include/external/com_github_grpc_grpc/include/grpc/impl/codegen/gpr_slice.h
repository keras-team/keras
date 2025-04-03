/*
 *
 * Copyright 2016 gRPC authors.
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
#ifndef GRPC_IMPL_CODEGEN_GPR_SLICE_H
#define GRPC_IMPL_CODEGEN_GPR_SLICE_H

/** WARNING: Please do not use this header. This was added as a temporary
 * measure to not break some of the external projects that depend on
 * gpr_slice_* functions. We are actively working on moving all the
 * gpr_slice_* references to grpc_slice_* and this file will be removed
 */

/* TODO (sreek) - Allowed by default but will be very soon turned off */
#define GRPC_ALLOW_GPR_SLICE_FUNCTIONS 1

#ifdef GRPC_ALLOW_GPR_SLICE_FUNCTIONS

#define gpr_slice_refcount grpc_slice_refcount
#define gpr_slice grpc_slice
#define gpr_slice_buffer grpc_slice_buffer

#define gpr_slice_ref grpc_slice_ref
#define gpr_slice_unref grpc_slice_unref
#define gpr_slice_new grpc_slice_new
#define gpr_slice_new_with_user_data grpc_slice_new_with_user_data
#define gpr_slice_new_with_len grpc_slice_new_with_len
#define gpr_slice_malloc grpc_slice_malloc
#define gpr_slice_from_copied_string grpc_slice_from_copied_string
#define gpr_slice_from_copied_buffer grpc_slice_from_copied_buffer
#define gpr_slice_from_static_string grpc_slice_from_static_string
#define gpr_slice_sub grpc_slice_sub
#define gpr_slice_sub_no_ref grpc_slice_sub_no_ref
#define gpr_slice_split_tail grpc_slice_split_tail
#define gpr_slice_split_head grpc_slice_split_head
#define gpr_slice_cmp grpc_slice_cmp
#define gpr_slice_str_cmp grpc_slice_str_cmp

#define gpr_slice_buffer grpc_slice_buffer
#define gpr_slice_buffer_init grpc_slice_buffer_init
#define gpr_slice_buffer_destroy grpc_slice_buffer_destroy
#define gpr_slice_buffer_add grpc_slice_buffer_add
#define gpr_slice_buffer_add_indexed grpc_slice_buffer_add_indexed
#define gpr_slice_buffer_addn grpc_slice_buffer_addn
#define gpr_slice_buffer_tiny_add grpc_slice_buffer_tiny_add
#define gpr_slice_buffer_pop grpc_slice_buffer_pop
#define gpr_slice_buffer_reset_and_unref grpc_slice_buffer_reset_and_unref
#define gpr_slice_buffer_swap grpc_slice_buffer_swap
#define gpr_slice_buffer_move_into grpc_slice_buffer_move_into
#define gpr_slice_buffer_trim_end grpc_slice_buffer_trim_end
#define gpr_slice_buffer_move_first grpc_slice_buffer_move_first
#define gpr_slice_buffer_take_first grpc_slice_buffer_take_first

#endif /* GRPC_ALLOW_GPR_SLICE_FUNCTIONS */

#endif /* GRPC_IMPL_CODEGEN_GPR_SLICE_H */
