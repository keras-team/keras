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

#ifndef GRPCPP_CHANNEL_H
#define GRPCPP_CHANNEL_H

#include <grpcpp/channel_impl.h>

namespace grpc {

typedef ::grpc_impl::Channel Channel;

namespace experimental {
/// Resets the channel's connection backoff.
/// TODO(roth): Once we see whether this proves useful, either create a gRFC
/// and change this to be a method of the Channel class, or remove it.
void ChannelResetConnectionBackoff(Channel* channel);
}  // namespace experimental

}  // namespace grpc

#endif  // GRPCPP_CHANNEL_H
