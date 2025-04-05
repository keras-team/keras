/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file implements low-level decoder helpers for working with "Binary
// Executor Format" (BEF) files.

#ifndef TFRT_SUPPORT_BEF_READER_H_
#define TFRT_SUPPORT_BEF_READER_H_

#include <cassert>

#include "llvm/ADT/ArrayRef.h"
#include "tfrt/bef/bef_buffer.h"
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/support/byte_order.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// This class contains the low-level decoder helpers for processing a BEF file.
//
// The reader methods in this class pervasively return a boolean value, and
// treats "false" as failure - without returning an error.
class BEFReader {
 public:
  explicit BEFReader(ArrayRef<uint8_t> file) : file_(file) {}

  // Return true if we've read the entire file.
  bool Empty() const { return file_.empty(); }

  ArrayRef<uint8_t> file() const { return file_; }

  // Move the reading point to immediately after the specified section.
  void SkipPast(ArrayRef<uint8_t> section) {
    assert(file_.begin() == section.begin());
    assert(section.end() <= file_.end());
    file_ = file_.drop_front(section.size());
  }

  // Skip over some number of bytes.
  void SkipOffset(size_t offset) {
    assert(offset <= file_.size());
    file_ = file_.drop_front(offset);
  }

  // Read a single byte from the stream.
  bool ReadByte(uint8_t* value) {
    if (file_.empty()) return false;
    *value = file_.front();
    file_ = file_.drop_front();
    return true;
  }

  // Read a VBR encoded integer from the byte stream.
  bool ReadVbrInt(size_t* value) {
    uint8_t next_byte;
    if (!ReadByte(&next_byte)) return false;

    *value = (next_byte & 127);
    while ((next_byte & 128) != 0) {
      if (!ReadByte(&next_byte)) return false;

      *value = (*value << 7) | size_t(next_byte & 127);
    }
    return true;
  }

  bool ReadSection(uint8_t* section_id, ArrayRef<uint8_t>* data) {
    size_t length;
    if (!ReadByte(section_id) || !ReadVbrInt(&length)) return false;

    // The low bit of the size is a boolean indicating whether there is an
    // alignment byte + padding present.
    bool has_alignment = (length & 1) != 0;
    length >>= 1;

    // If there is an alignment byte, read the alignment byte, and skip over
    // padding bytes until we reach the desired alignment.
    if (has_alignment) {
      auto is_power_of_2 = [](uint8_t x) {
        return x != 0 && (x & (x - 1)) == 0;
      };

      uint8_t alignment;
      if (!ReadByte(&alignment) || !is_power_of_2(alignment) ||
          !ReadAlignment(alignment)) {
        return false;
      }
    }

    // Okay, the returned data is whatever is left now.
    if (length > file_.size()) return false;

    *data = file_.take_front(length);
    return true;
  }

  // Skip over bytes until reaching the specified alignment.
  bool ReadAlignment(unsigned alignment) {
    uint8_t padding;
    while (reinterpret_cast<uintptr_t>(file_.data()) & (alignment - 1))
      if (!ReadByte(&padding)) return false;

    return true;
  }

 private:
  ArrayRef<uint8_t> file_;
};

// This class contains helper methods for reading a kernel from kernel entries
// in a BEF function.
class BEFKernel {
  // BEFKernelHeader has the same data layout for the kernel header (excluding
  // num_used_bys as the number of results is not fixed.) in BEF. kernel_code,
  // kernel_location, num_arguments, num_attributes, num_functions, and
  // num_results in BEF can be directly mapped using this struct.
  struct BEFKernelHeader {
    uint32_t kernel_code;
    uint32_t kernel_location;
    uint32_t num_arguments;
    uint32_t num_attributes;
    uint32_t num_functions;
    uint32_t num_results;
  };
  static_assert(sizeof(BEFKernelHeader) == 24,
                "Unexpected size of BEFKernelHeader.");

 public:
  BEFKernel(const uint32_t* kernel_start)
      : header_(reinterpret_cast<const BEFKernelHeader*>(kernel_start)),
        result_table_(kernel_start + llvm::alignTo(sizeof(BEFKernelHeader),
                                                   kKernelEntryAlignment) /
                                         kKernelEntryAlignment),
        body_start_(result_table_ + header_->num_results) {
    ASSERT_LITTLE_ENDIAN();
  }

  uint32_t kernel_code() const { return header_->kernel_code; }
  uint32_t kernel_location() const { return header_->kernel_location; }
  uint32_t num_arguments() const { return header_->num_arguments; }
  uint32_t num_attributes() const { return header_->num_attributes; }
  uint32_t num_functions() const { return header_->num_functions; }
  uint32_t num_results() const { return header_->num_results; }

  uint32_t num_used_bys(int result_number) const {
    assert(result_number < header_->num_results);
    return result_table_[result_number];
  }

  // Return num_entries kernel entries starting at offset.
  ArrayRef<uint32_t> GetKernelEntries(int offset, int num_entries) const {
    return llvm::ArrayRef(body_start_ + offset, num_entries);
  }

  ArrayRef<uint32_t> GetArguments() const {
    return llvm::ArrayRef(body_start_, num_arguments());
  }

  ArrayRef<uint32_t> GetAttributes() const {
    return llvm::ArrayRef(body_start_ + num_arguments(), num_attributes());
  }

  ArrayRef<uint32_t> GetFunctions() const {
    return llvm::ArrayRef(body_start_ + num_arguments() + num_attributes(),
                          num_functions());
  }

  ArrayRef<uint32_t> GetResults() const {
    return llvm::ArrayRef(
        body_start_ + num_arguments() + num_attributes() + num_functions(),
        num_results());
  }

 private:
  const BEFKernelHeader* header_;
  // The result table contains the list of NumUsedBys.
  const uint32_t* result_table_;
  const uint32_t* body_start_;
};

}  // namespace tfrt

#endif  // TFRT_SUPPORT_BEF_READER_H_
