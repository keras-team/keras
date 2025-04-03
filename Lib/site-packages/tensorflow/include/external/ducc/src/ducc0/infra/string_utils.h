/*
 *  This file is part of the MR utility library.
 *
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

/** \file ducc0/infra/string_utils.h
 *
 *  \copyright Copyright (C) 2019-2021 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef DUCC0_STRING_UTILS_H
#define DUCC0_STRING_UTILS_H

// FIXME: most of this will be superseded by C++20 std::format

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace ducc0 {

namespace detail_string_utils {

/*! \defgroup stringutilsgroup String handling helper functions */
/*! \{ */

/// Returns the string \a orig without leading and trailing whitespace.
std::string trim (const std::string &orig);

/// Returns a string containing the text representation of \a x.
/*! Care is taken that no information is lost in the conversion. */
template<typename T> std::string dataToString(const T &x);
template<> std::string dataToString (const bool &x);
template<> std::string dataToString (const std::string &x);
template<> std::string dataToString (const float &x);
template<> std::string dataToString (const double &x);
template<> std::string dataToString (const long double &x);

/// Returns a string containing the text representation of \a x, padded
/// with leading zeroes to \a width characters.
std::string intToString(std::int64_t x, std::size_t width);

/// Reads a value of a given datatype from a string.
template<typename T> T stringToData (const std::string &x);
template<> std::string stringToData (const std::string &x);
template<> bool stringToData (const std::string &x);

/// Case-insensitive string comparison
/*! Returns \a true, if \a a and \a b differ only in capitalisation,
    else \a false. */
bool equal_nocase (const std::string &a, const std::string &b);

/// Returns lowercase version of \a input.
std::string tolower(const std::string &input);

/// Tries to split \a inp into a white-space separated list of values of
/// type \a T, and appends them to \a list.
template<typename T> inline std::vector<T> split (const std::string &inp);

/// Breaks the string \a inp into tokens separated by \a delim, and returns them
/// as a vector<string>.
std::vector<std::string> tokenize (const std::string &inp, char delim);

/// Breaks the contents of file \a filename into tokens separated by white
/// space, and returns them as a vector<string>.
std::vector<std::string> parse_words_from_file (const std::string &filename);

/*! \} */

}

using detail_string_utils::trim;
//using detail_string_utils::intToString;
using detail_string_utils::dataToString;
using detail_string_utils::stringToData;
using detail_string_utils::equal_nocase;
//using detail_string_utils::tolower;
//using detail_string_utils::split;
//using detail_string_utils::tokenize;
//using detail_string_utils::parse_words_from_file;

}

#endif
