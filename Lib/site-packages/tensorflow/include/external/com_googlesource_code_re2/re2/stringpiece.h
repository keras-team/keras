// Copyright 2022 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef RE2_STRINGPIECE_H_
#define RE2_STRINGPIECE_H_

#include "absl/strings/string_view.h"

namespace re2 {

// RE2 has two versions: "sans Abseil" in the main branch; and "avec Abseil" in
// the abseil branch. This has led to a diamond dependency problem for projects
// like Envoy: as per https://github.com/google/re2/issues/388, GoogleTest took
// a dependency on RE2 avec Abseil, but other things depend on RE2 sans Abseil.
// To resolve this conflict until both versions can migrate to std::string_view
// (C++17), those other things must be able to #include "re2/stringpiece.h" and
// use re2::StringPiece. (This is a hack, obviously, but it beats telling every
// project in this situation that they have to perform source transformations.)
using StringPiece = absl::string_view;

}  // namespace re2

#endif  // RE2_STRINGPIECE_H_
