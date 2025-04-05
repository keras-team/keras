// Copyright 2005 and onwards Google Inc.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// A light-weight compression algorithm.  It is designed for speed of
// compression and decompression, rather than for the utmost in space
// savings.
//
// For getting better compression ratios when you are compressing data
// with long repeated sequences or compressing data that is similar to
// other data, while still compressing fast, you might look at first
// using BMDiff and then compressing the output of BMDiff with
// Snappy.

#ifndef THIRD_PARTY_SNAPPY_SNAPPY_H__
#define THIRD_PARTY_SNAPPY_SNAPPY_H__

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "snappy-stubs-public.h"

namespace snappy {
  class Source;
  class Sink;

  struct CompressionOptions {
    // Compression level.
    // Level 1 is the fastest
    // Level 2 is a little slower but provides better compression. Level 2 is
    // **EXPERIMENTAL** for the time being. It might happen that we decide to
    // fall back to level 1 in the future.
    // Levels 3+ are currently not supported. We plan to support levels up to
    // 9 in the future.
    // If you played with other compression algorithms, level 1 is equivalent to
    // fast mode (level 1) of LZ4, level 2 is equivalent to LZ4's level 2 mode
    // and compresses somewhere around zstd:-3 and zstd:-2 but generally with
    // faster decompression speeds than snappy:1 and zstd:-3.
    int level = DefaultCompressionLevel();

    constexpr CompressionOptions() = default;
    constexpr CompressionOptions(int compression_level)
        : level(compression_level) {}
    static constexpr int MinCompressionLevel() { return 1; }
    static constexpr int MaxCompressionLevel() { return 2; }
    static constexpr int DefaultCompressionLevel() { return 1; }
  };

  // ------------------------------------------------------------------------
  // Generic compression/decompression routines.
  // ------------------------------------------------------------------------

  // Compress the bytes read from "*reader" and append to "*writer". Return the
  // number of bytes written.
  size_t Compress(Source* reader, Sink* writer,
                  CompressionOptions options = {});

  // Find the uncompressed length of the given stream, as given by the header.
  // Note that the true length could deviate from this; the stream could e.g.
  // be truncated.
  //
  // Also note that this leaves "*source" in a state that is unsuitable for
  // further operations, such as RawUncompress(). You will need to rewind
  // or recreate the source yourself before attempting any further calls.
  bool GetUncompressedLength(Source* source, uint32_t* result);

  // ------------------------------------------------------------------------
  // Higher-level string based routines (should be sufficient for most users)
  // ------------------------------------------------------------------------

  // Sets "*compressed" to the compressed version of "input[0..input_length-1]".
  // Original contents of *compressed are lost.
  //
  // REQUIRES: "input[]" is not an alias of "*compressed".
  size_t Compress(const char* input, size_t input_length,
                  std::string* compressed, CompressionOptions options = {});

  // Same as `Compress` above but taking an `iovec` array as input. Note that
  // this function preprocesses the inputs to compute the sum of
  // `iov[0..iov_cnt-1].iov_len` before reading. To avoid this, use
  // `RawCompressFromIOVec` below.
  size_t CompressFromIOVec(const struct iovec* iov, size_t iov_cnt,
                           std::string* compressed,
                           CompressionOptions options = {});

  // Decompresses "compressed[0..compressed_length-1]" to "*uncompressed".
  // Original contents of "*uncompressed" are lost.
  //
  // REQUIRES: "compressed[]" is not an alias of "*uncompressed".
  //
  // returns false if the message is corrupted and could not be decompressed
  bool Uncompress(const char* compressed, size_t compressed_length,
                  std::string* uncompressed);

  // Decompresses "compressed" to "*uncompressed".
  //
  // returns false if the message is corrupted and could not be decompressed
  bool Uncompress(Source* compressed, Sink* uncompressed);

  // This routine uncompresses as much of the "compressed" as possible
  // into sink.  It returns the number of valid bytes added to sink
  // (extra invalid bytes may have been added due to errors; the caller
  // should ignore those). The emitted data typically has length
  // GetUncompressedLength(), but may be shorter if an error is
  // encountered.
  size_t UncompressAsMuchAsPossible(Source* compressed, Sink* uncompressed);

  // ------------------------------------------------------------------------
  // Lower-level character array based routines.  May be useful for
  // efficiency reasons in certain circumstances.
  // ------------------------------------------------------------------------

  // REQUIRES: "compressed" must point to an area of memory that is at
  // least "MaxCompressedLength(input_length)" bytes in length.
  //
  // Takes the data stored in "input[0..input_length]" and stores
  // it in the array pointed to by "compressed".
  //
  // "*compressed_length" is set to the length of the compressed output.
  //
  // Example:
  //    char* output = new char[snappy::MaxCompressedLength(input_length)];
  //    size_t output_length;
  //    RawCompress(input, input_length, output, &output_length);
  //    ... Process(output, output_length) ...
  //    delete [] output;
  void RawCompress(const char* input, size_t input_length, char* compressed,
                   size_t* compressed_length, CompressionOptions options = {});

  // Same as `RawCompress` above but taking an `iovec` array as input. Note that
  // `uncompressed_length` is the total number of bytes to be read from the
  // elements of `iov` (_not_ the number of elements in `iov`).
  void RawCompressFromIOVec(const struct iovec* iov, size_t uncompressed_length,
                            char* compressed, size_t* compressed_length,
                            CompressionOptions options = {});

  // Given data in "compressed[0..compressed_length-1]" generated by
  // calling the Snappy::Compress routine, this routine
  // stores the uncompressed data to
  //    uncompressed[0..GetUncompressedLength(compressed)-1]
  // returns false if the message is corrupted and could not be decrypted
  bool RawUncompress(const char* compressed, size_t compressed_length,
                     char* uncompressed);

  // Given data from the byte source 'compressed' generated by calling
  // the Snappy::Compress routine, this routine stores the uncompressed
  // data to
  //    uncompressed[0..GetUncompressedLength(compressed,compressed_length)-1]
  // returns false if the message is corrupted and could not be decrypted
  bool RawUncompress(Source* compressed, char* uncompressed);

  // Given data in "compressed[0..compressed_length-1]" generated by
  // calling the Snappy::Compress routine, this routine
  // stores the uncompressed data to the iovec "iov". The number of physical
  // buffers in "iov" is given by iov_cnt and their cumulative size
  // must be at least GetUncompressedLength(compressed). The individual buffers
  // in "iov" must not overlap with each other.
  //
  // returns false if the message is corrupted and could not be decrypted
  bool RawUncompressToIOVec(const char* compressed, size_t compressed_length,
                            const struct iovec* iov, size_t iov_cnt);

  // Given data from the byte source 'compressed' generated by calling
  // the Snappy::Compress routine, this routine stores the uncompressed
  // data to the iovec "iov". The number of physical
  // buffers in "iov" is given by iov_cnt and their cumulative size
  // must be at least GetUncompressedLength(compressed). The individual buffers
  // in "iov" must not overlap with each other.
  //
  // returns false if the message is corrupted and could not be decrypted
  bool RawUncompressToIOVec(Source* compressed, const struct iovec* iov,
                            size_t iov_cnt);

  // Returns the maximal size of the compressed representation of
  // input data that is "source_bytes" bytes in length;
  size_t MaxCompressedLength(size_t source_bytes);

  // REQUIRES: "compressed[]" was produced by RawCompress() or Compress()
  // Returns true and stores the length of the uncompressed data in
  // *result normally.  Returns false on parsing error.
  // This operation takes O(1) time.
  bool GetUncompressedLength(const char* compressed, size_t compressed_length,
                             size_t* result);

  // Returns true iff the contents of "compressed[]" can be uncompressed
  // successfully.  Does not return the uncompressed data.  Takes
  // time proportional to compressed_length, but is usually at least
  // a factor of four faster than actual decompression.
  bool IsValidCompressedBuffer(const char* compressed,
                               size_t compressed_length);

  // Returns true iff the contents of "compressed" can be uncompressed
  // successfully.  Does not return the uncompressed data.  Takes
  // time proportional to *compressed length, but is usually at least
  // a factor of four faster than actual decompression.
  // On success, consumes all of *compressed.  On failure, consumes an
  // unspecified prefix of *compressed.
  bool IsValidCompressed(Source* compressed);

  // The size of a compression block. Note that many parts of the compression
  // code assumes that kBlockSize <= 65536; in particular, the hash table
  // can only store 16-bit offsets, and EmitCopy() also assumes the offset
  // is 65535 bytes or less. Note also that if you change this, it will
  // affect the framing format (see framing_format.txt).
  //
  // Note that there might be older data around that is compressed with larger
  // block sizes, so the decompression code should not rely on the
  // non-existence of long backreferences.
  static constexpr int kBlockLog = 16;
  static constexpr size_t kBlockSize = 1 << kBlockLog;

  static constexpr int kMinHashTableBits = 8;
  static constexpr size_t kMinHashTableSize = 1 << kMinHashTableBits;

  static constexpr int kMaxHashTableBits = 15;
  static constexpr size_t kMaxHashTableSize = 1 << kMaxHashTableBits;
}  // end namespace snappy

#endif  // THIRD_PARTY_SNAPPY_SNAPPY_H__
