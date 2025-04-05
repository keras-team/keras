//===- IndentedOstream.h - raw ostream wrapper to indent --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// raw_ostream subclass that keeps track of indentation for textual output
// where indentation helps readability.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_INDENTEDOSTREAM_H_
#define MLIR_SUPPORT_INDENTEDOSTREAM_H_

#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

/// raw_ostream subclass that simplifies indention a sequence of code.
class raw_indented_ostream : public raw_ostream {
public:
  explicit raw_indented_ostream(llvm::raw_ostream &os) : os(os) {
    SetUnbuffered();
  }

  /// Simple RAII struct to use to indentation around entering/exiting region.
  struct DelimitedScope {
    explicit DelimitedScope(raw_indented_ostream &os, StringRef open = "",
                            StringRef close = "", bool indent = true)
        : os(os), open(open), close(close), indent(indent) {
      os << open;
      if (indent)
        os.indent();
    }
    ~DelimitedScope() {
      if (indent)
        os.unindent();
      os << close;
    }

    raw_indented_ostream &os;

  private:
    StringRef open, close;
    bool indent;
  };

  /// Returns the underlying (unindented) raw_ostream.
  raw_ostream &getOStream() const { return os; }

  /// Returns DelimitedScope.
  DelimitedScope scope(StringRef open = "", StringRef close = "",
                       bool indent = true) {
    return DelimitedScope(*this, open, close, indent);
  }

  /// Prints a string re-indented to the current indent. Re-indents by removing
  /// the leading whitespace from the first non-empty line from every line of
  /// the string, skipping over empty lines at the start. Prefixes each line
  /// with extraPrefix after the indentation.
  raw_indented_ostream &printReindented(StringRef str,
                                        StringRef extraPrefix = "");

  /// Increases the indent and returning this raw_indented_ostream.
  raw_indented_ostream &indent() {
    currentIndent += indentSize;
    return *this;
  }

  /// Decreases the indent and returning this raw_indented_ostream.
  raw_indented_ostream &unindent() {
    currentIndent = std::max(0, currentIndent - indentSize);
    return *this;
  }

  /// Emits whitespace and sets the indentation for the stream.
  raw_indented_ostream &indent(int with) {
    os.indent(with);
    atStartOfLine = false;
    currentIndent = with;
    return *this;
  }

private:
  void write_impl(const char *ptr, size_t size) final;

  /// Return the current position within the stream, not counting the bytes
  /// currently in the buffer.
  uint64_t current_pos() const final { return os.tell(); }

  /// Constant indent added/removed.
  static constexpr int indentSize = 2;

  /// Tracker for current indentation.
  int currentIndent = 0;

  /// The leading whitespace of the string being printed, if reindent is used.
  int leadingWs = 0;

  /// The extra prefix to be printed, if reindent is used.
  StringRef currentExtraPrefix;

  /// Tracks whether at start of line and so indent is required or not.
  bool atStartOfLine = true;

  /// The underlying raw_ostream.
  raw_ostream &os;
};

inline raw_indented_ostream &
mlir::raw_indented_ostream::printReindented(StringRef str,
                                            StringRef extraPrefix) {
  StringRef output = str;
  // Skip empty lines.
  while (!output.empty()) {
    auto split = output.split('\n');
    // Trim Windows \r characters from \r\n line endings.
    auto firstTrimmed = split.first.rtrim('\r');
    size_t indent = firstTrimmed.find_first_not_of(" \t");
    if (indent != StringRef::npos) {
      // Set an initial value.
      leadingWs = indent;
      break;
    }
    output = split.second;
  }
  // Determine the maximum indent.
  StringRef remaining = output;
  while (!remaining.empty()) {
    auto split = remaining.split('\n');
    auto firstTrimmed = split.first.rtrim('\r');
    size_t indent = firstTrimmed.find_first_not_of(" \t");
    if (indent != StringRef::npos)
      leadingWs = std::min(leadingWs, static_cast<int>(indent));
    remaining = split.second;
  }
  // Print, skipping the empty lines.
  std::swap(currentExtraPrefix, extraPrefix);
  *this << output;
  std::swap(currentExtraPrefix, extraPrefix);
  leadingWs = 0;
  return *this;
}

inline void mlir::raw_indented_ostream::write_impl(const char *ptr,
                                                   size_t size) {
  StringRef str(ptr, size);
  // Print out indented.
  auto print = [this](StringRef str) {
    if (atStartOfLine)
      os.indent(currentIndent) << currentExtraPrefix << str.substr(leadingWs);
    else
      os << str.substr(leadingWs);
  };

  while (!str.empty()) {
    size_t idx = str.find('\n');
    if (idx == StringRef::npos) {
      if (!str.substr(leadingWs).empty()) {
        print(str);
        atStartOfLine = false;
      }
      break;
    }

    auto split =
        std::make_pair(str.slice(0, idx), str.slice(idx + 1, StringRef::npos));
    // Print empty new line without spaces if line only has spaces and no extra
    // prefix is requested.
    if (!split.first.ltrim().empty() || !currentExtraPrefix.empty())
      print(split.first);
    os << '\n';
    atStartOfLine = true;
    str = split.second;
  }
}

} // namespace mlir
#endif // MLIR_SUPPORT_INDENTEDOSTREAM_H_
