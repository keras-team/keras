/** \file ducc0/infra/error_handling.h
 *
 * \copyright Copyright (C) 2019-2021 Max-Planck-Society
 * \author Martin Reinecke
 */

/* SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0-or-later */

/*
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

#ifndef DUCC0_ERROR_HANDLING_H
#define DUCC0_ERROR_HANDLING_H

#include <sstream>
#include <stdexcept>
#include "ducc0/infra/useful_macros.h"

namespace ducc0 {

namespace detail_error_handling {

#if defined (__GNUC__)
#define DUCC0_ERROR_HANDLING_LOC_ ::ducc0::detail_error_handling::CodeLocation(__FILE__, __LINE__, __PRETTY_FUNCTION__)
#else
#define DUCC0_ERROR_HANDLING_LOC_ ::ducc0::detail_error_handling::CodeLocation(__FILE__, __LINE__)
#endif

// to be replaced with std::source_location once generally available
class CodeLocation
  {
  private:
    const char *file, *func;
    int line;

  public:
    CodeLocation(const char *file_, int line_, const char *func_=nullptr)
      : file(file_), func(func_), line(line_) {}

    inline ::std::ostream &print(::std::ostream &os) const
      {
      os << "\n" << file <<  ": " <<  line;
      if (func) os << " (" << func << ")";
      os << ":\n";
      return os;
      }
  };

inline ::std::ostream &operator<<(::std::ostream &os, const CodeLocation &loc)
  { return loc.print(os); }

template<typename ...Args>
void streamDump__(::std::ostream &os, Args&&... args)
  { (os << ... << args); }
template<typename ...Args>
[[noreturn]] DUCC0_NOINLINE void fail__(Args&&... args)
  {
  ::std::ostringstream msg; \
  ::ducc0::detail_error_handling::streamDump__(msg, std::forward<Args>(args)...); \
    throw ::std::runtime_error(msg.str()); \
  }

/// Throws a std::runtime_error containing the code location and the
/// passed arguments.
#define MR_fail(...) \
  do { \
    ::ducc0::detail_error_handling::fail__(DUCC0_ERROR_HANDLING_LOC_, "\n", ##__VA_ARGS__, "\n"); \
    } while(0)

/// If \a cond is false, throws a std::runtime_error containing the code
/// location and the passed arguments.
#define MR_assert(cond,...) \
  do { \
    if (cond); \
    else { MR_fail("Assertion failure\n", ##__VA_ARGS__); } \
    } while(0)

}}

#endif
