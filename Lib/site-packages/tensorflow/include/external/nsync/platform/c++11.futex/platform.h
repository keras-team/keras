/* Copyright 2018 Google Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. */

#ifndef NSYNC_PLATFORM_CPP11_FUTEX_PLATFORM_H_
#define NSYNC_PLATFORM_CPP11_FUTEX_PLATFORM_H_

/* This header is for use of when a mostly-C++11 build is desired,
   but on a Linux system where fuxtex is available, and outperforms
   the C++11 library functions. */

#include "../c++11/platform.h"
#include <unistd.h>
#include <linux/futex.h>
#include <sys/syscall.h>

#endif /*NSYNC_PLATFORM_CPP11_FUTEX_PLATFORM_H_*/
