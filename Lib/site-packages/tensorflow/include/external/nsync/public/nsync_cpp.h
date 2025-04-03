/* Copyright 2016 Google Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. */

#ifndef NSYNC_PUBLIC_NSYNC_CPP_H_
#define NSYNC_PUBLIC_NSYNC_CPP_H_

/* This header file permits compilation via a C++ compiler using the macros
   NSYNC_CPP_START_, NSYNC_CPP_END_, and NSYNC_CPP_USING_.

   NSYNC_CPP_START_ and NSYNC_CPP_END_ surround C code in the public library.
   They put all public symbols into the "nsync" name space.

   NSYNC_CPP_USING_ is used before C code (used for testing) that might use
   public exports from this package.  It makes symbols in the "nsync" 
   name space available without the "nsync::" prefix.

   NSYNC_C_START_ and NSYNC_C_END_ surround C code in the C++ modules.
 */

#if defined(__cplusplus)
#define NSYNC_CPP_START_ namespace nsync {
#define NSYNC_CPP_END_ }
#define NSYNC_CPP_USING_ using namespace nsync;
#define NSYNC_C_START_ extern "C" {
#define NSYNC_C_END_ }
#else
#define NSYNC_CPP_START_
#define NSYNC_CPP_END_
#define NSYNC_CPP_USING_
#define NSYNC_C_START_
#define NSYNC_C_END_
#endif

#endif /*NSYNC_PUBLIC_NSYNC_CPP_H_*/
