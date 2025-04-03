//===- File.h - Reading sparse tensors from files ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements reading sparse tensor from files in one of the
// following external formats:
//
// (1) Matrix Market Exchange (MME): *.mtx
//     https://math.nist.gov/MatrixMarket/formats.html
//
// (2) Formidable Repository of Open Sparse Tensors and Tools (FROSTT): *.tns
//     http://frostt.io/tensors/file-formats.html
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_FILE_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_FILE_H

#include "mlir/ExecutionEngine/SparseTensor/MapRef.h"
#include "mlir/ExecutionEngine/SparseTensor/Storage.h"

#include <fstream>

namespace mlir {
namespace sparse_tensor {

namespace detail {

template <typename T>
struct is_complex final : public std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> final : public std::true_type {};

/// Returns an element-value of non-complex type.  If `IsPattern` is true,
/// then returns an arbitrary value.  If `IsPattern` is false, then
/// reads the value from the current line buffer beginning at `linePtr`.
template <typename V, bool IsPattern>
inline std::enable_if_t<!is_complex<V>::value, V> readValue(char **linePtr) {
  // The external formats always store these numerical values with the type
  // double, but we cast these values to the sparse tensor object type.
  // For a pattern tensor, we arbitrarily pick the value 1 for all entries.
  if constexpr (IsPattern)
    return 1.0;
  return strtod(*linePtr, linePtr);
}

/// Returns an element-value of complex type.  If `IsPattern` is true,
/// then returns an arbitrary value.  If `IsPattern` is false, then reads
/// the value from the current line buffer beginning at `linePtr`.
template <typename V, bool IsPattern>
inline std::enable_if_t<is_complex<V>::value, V> readValue(char **linePtr) {
  // Read two values to make a complex. The external formats always store
  // numerical values with the type double, but we cast these values to the
  // sparse tensor object type. For a pattern tensor, we arbitrarily pick the
  // value 1 for all entries.
  if constexpr (IsPattern)
    return V(1.0, 1.0);
  double re = strtod(*linePtr, linePtr);
  double im = strtod(*linePtr, linePtr);
  // Avoiding brace-notation since that forbids narrowing to `float`.
  return V(re, im);
}

/// Returns an element-value.  If `isPattern` is true, then returns an
/// arbitrary value.  If `isPattern` is false, then reads the value from
/// the current line buffer beginning at `linePtr`.
template <typename V>
inline V readValue(char **linePtr, bool isPattern) {
  return isPattern ? readValue<V, true>(linePtr) : readValue<V, false>(linePtr);
}

} // namespace detail

//===----------------------------------------------------------------------===//
//
//  Reader class.
//
//===----------------------------------------------------------------------===//

/// This class abstracts over the information stored in file headers,
/// as well as providing the buffers and methods for parsing those headers.
class SparseTensorReader final {
public:
  enum class ValueKind : uint8_t {
    // The value before calling `readHeader`.
    kInvalid = 0,
    // Values that can be set by `readMMEHeader`.
    kPattern = 1,
    kReal = 2,
    kInteger = 3,
    kComplex = 4,
    // The value set by `readExtFROSTTHeader`.
    kUndefined = 5
  };

  explicit SparseTensorReader(const char *filename) : filename(filename) {
    assert(filename && "Received nullptr for filename");
  }

  // Disallows copying, to avoid duplicating the `file` pointer.
  SparseTensorReader(const SparseTensorReader &) = delete;
  SparseTensorReader &operator=(const SparseTensorReader &) = delete;

  /// Factory method to allocate a new reader, open the file, read the
  /// header, and validate that the actual contents of the file match
  /// the expected `dimShape` and `valTp`.
  static SparseTensorReader *create(const char *filename, uint64_t dimRank,
                                    const uint64_t *dimShape,
                                    PrimaryType valTp) {
    SparseTensorReader *reader = new SparseTensorReader(filename);
    reader->openFile();
    reader->readHeader();
    if (!reader->canReadAs(valTp)) {
      fprintf(stderr,
              "Tensor element type %d not compatible with values in file %s\n",
              static_cast<int>(valTp), filename);
      exit(1);
    }
    reader->assertMatchesShape(dimRank, dimShape);
    return reader;
  }

  // This dtor tries to avoid leaking the `file`.  (Though it's better
  // to call `closeFile` explicitly when possible, since there are
  // circumstances where dtors are not called reliably.)
  ~SparseTensorReader() { closeFile(); }

  /// Opens the file for reading.
  void openFile();

  /// Closes the file.
  void closeFile();

  /// Reads and parses the file's header.
  void readHeader();

  /// Returns the stored value kind.
  ValueKind getValueKind() const { return valueKind_; }

  /// Checks if a header has been successfully read.
  bool isValid() const { return valueKind_ != ValueKind::kInvalid; }

  /// Checks if the file's ValueKind can be converted into the given
  /// tensor PrimaryType.  Is only valid after parsing the header.
  bool canReadAs(PrimaryType valTy) const;

  /// Gets the MME "pattern" property setting.  Is only valid after
  /// parsing the header.
  bool isPattern() const {
    assert(isValid() && "Attempt to isPattern() before readHeader()");
    return valueKind_ == ValueKind::kPattern;
  }

  /// Gets the MME "symmetric" property setting.  Is only valid after
  /// parsing the header.
  bool isSymmetric() const {
    assert(isValid() && "Attempt to isSymmetric() before readHeader()");
    return isSymmetric_;
  }

  /// Gets the dimension-rank of the tensor.  Is only valid after parsing
  /// the header.
  uint64_t getRank() const {
    assert(isValid() && "Attempt to getRank() before readHeader()");
    return idata[0];
  }

  /// Gets the number of stored elements.  Is only valid after parsing
  /// the header.
  uint64_t getNSE() const {
    assert(isValid() && "Attempt to getNSE() before readHeader()");
    return idata[1];
  }

  /// Gets the dimension-sizes array.  The pointer itself is always
  /// valid; however, the values stored therein are only valid after
  /// parsing the header.
  const uint64_t *getDimSizes() const { return idata + 2; }

  /// Safely gets the size of the given dimension.  Is only valid
  /// after parsing the header.
  uint64_t getDimSize(uint64_t d) const {
    assert(d < getRank() && "Dimension out of bounds");
    return idata[2 + d];
  }

  /// Asserts the shape subsumes the actual dimension sizes.  Is only
  /// valid after parsing the header.
  void assertMatchesShape(uint64_t rank, const uint64_t *shape) const;

  /// Allocates a new sparse-tensor storage object with the given encoding,
  /// initializes it by reading all the elements from the file, and then
  /// closes the file. Templated on P, I, and V.
  template <typename P, typename I, typename V>
  SparseTensorStorage<P, I, V> *
  readSparseTensor(uint64_t lvlRank, const uint64_t *lvlSizes,
                   const LevelType *lvlTypes, const uint64_t *dim2lvl,
                   const uint64_t *lvl2dim) {
    const uint64_t dimRank = getRank();
    MapRef map(dimRank, lvlRank, dim2lvl, lvl2dim);
    auto *lvlCOO = readCOO<V>(map, lvlSizes);
    auto *tensor = SparseTensorStorage<P, I, V>::newFromCOO(
        dimRank, getDimSizes(), lvlRank, lvlSizes, lvlTypes, dim2lvl, lvl2dim,
        lvlCOO);
    delete lvlCOO;
    return tensor;
  }

  /// Reads the COO tensor from the file, stores the coordinates and values to
  /// the given buffers, returns a boolean value to indicate whether the COO
  /// elements are sorted.
  template <typename C, typename V>
  bool readToBuffers(uint64_t lvlRank, const uint64_t *dim2lvl,
                     const uint64_t *lvl2dim, C *lvlCoordinates, V *values);

private:
  /// Attempts to read a line from the file.
  void readLine();

  /// Reads the next line of the input file and parses the coordinates
  /// into the `dimCoords` argument.  Returns the position in the `line`
  /// buffer where the element's value should be parsed from.
  template <typename C>
  char *readCoords(C *dimCoords) {
    readLine();
    // Local variable for tracking the parser's position in the `line` buffer.
    char *linePtr = line;
    for (uint64_t dimRank = getRank(), d = 0; d < dimRank; ++d) {
      // Parse the 1-based coordinate.
      uint64_t c = strtoul(linePtr, &linePtr, 10);
      // Store the 0-based coordinate.
      dimCoords[d] = static_cast<C>(c - 1);
    }
    return linePtr;
  }

  /// Reads all the elements from the file while applying the given map.
  template <typename V>
  SparseTensorCOO<V> *readCOO(const MapRef &map, const uint64_t *lvlSizes);

  /// The implementation of `readCOO` that is templated `IsPattern` in order
  /// to perform LICM without needing to duplicate the source code.
  template <typename V, bool IsPattern>
  void readCOOLoop(const MapRef &map, SparseTensorCOO<V> *coo);

  /// The internal implementation of `readToBuffers`. We template over
  /// `IsPattern` in order to perform LICM without needing to duplicate
  /// the source code.
  template <typename C, typename V, bool IsPattern>
  bool readToBuffersLoop(const MapRef &map, C *lvlCoordinates, V *values);

  /// Reads the MME header of a general sparse matrix of type real.
  void readMMEHeader();

  /// Reads the "extended" FROSTT header. Although not part of the
  /// documented format, we assume that the file starts with optional
  /// comments followed by two lines that define the rank, the number of
  /// nonzeros, and the dimensions sizes (one per rank) of the sparse tensor.
  void readExtFROSTTHeader();

  static constexpr int kColWidth = 1025;
  const char *const filename;
  FILE *file = nullptr;
  ValueKind valueKind_ = ValueKind::kInvalid;
  bool isSymmetric_ = false;
  uint64_t idata[512];
  char line[kColWidth];
};

//===----------------------------------------------------------------------===//
//
//  Reader class methods.
//
//===----------------------------------------------------------------------===//

template <typename V>
SparseTensorCOO<V> *SparseTensorReader::readCOO(const MapRef &map,
                                                const uint64_t *lvlSizes) {
  assert(isValid() && "Attempt to readCOO() before readHeader()");
  // Prepare a COO object with the number of stored elems as initial capacity.
  auto *coo = new SparseTensorCOO<V>(map.getLvlRank(), lvlSizes, getNSE());
  // Enter the reading loop.
  if (isPattern())
    readCOOLoop<V, true>(map, coo);
  else
    readCOOLoop<V, false>(map, coo);
  // Close the file and return the COO.
  closeFile();
  return coo;
}

template <typename V, bool IsPattern>
void SparseTensorReader::readCOOLoop(const MapRef &map,
                                     SparseTensorCOO<V> *coo) {
  const uint64_t dimRank = map.getDimRank();
  const uint64_t lvlRank = map.getLvlRank();
  assert(dimRank == getRank());
  std::vector<uint64_t> dimCoords(dimRank);
  std::vector<uint64_t> lvlCoords(lvlRank);
  for (uint64_t k = 0, nse = getNSE(); k < nse; k++) {
    char *linePtr = readCoords(dimCoords.data());
    const V value = detail::readValue<V, IsPattern>(&linePtr);
    map.pushforward(dimCoords.data(), lvlCoords.data());
    coo->add(lvlCoords, value);
  }
}

template <typename C, typename V>
bool SparseTensorReader::readToBuffers(uint64_t lvlRank,
                                       const uint64_t *dim2lvl,
                                       const uint64_t *lvl2dim,
                                       C *lvlCoordinates, V *values) {
  assert(isValid() && "Attempt to readCOO() before readHeader()");
  MapRef map(getRank(), lvlRank, dim2lvl, lvl2dim);
  bool isSorted =
      isPattern() ? readToBuffersLoop<C, V, true>(map, lvlCoordinates, values)
                  : readToBuffersLoop<C, V, false>(map, lvlCoordinates, values);
  closeFile();
  return isSorted;
}

template <typename C, typename V, bool IsPattern>
bool SparseTensorReader::readToBuffersLoop(const MapRef &map, C *lvlCoordinates,
                                           V *values) {
  const uint64_t dimRank = map.getDimRank();
  const uint64_t lvlRank = map.getLvlRank();
  const uint64_t nse = getNSE();
  assert(dimRank == getRank());
  std::vector<C> dimCoords(dimRank);
  bool isSorted = false;
  char *linePtr;
  const auto readNextElement = [&]() {
    linePtr = readCoords<C>(dimCoords.data());
    map.pushforward(dimCoords.data(), lvlCoordinates);
    *values = detail::readValue<V, IsPattern>(&linePtr);
    if (isSorted) {
      // Note that isSorted is set to false when reading the first element,
      // to guarantee the safeness of using prevLvlCoords.
      C *prevLvlCoords = lvlCoordinates - lvlRank;
      for (uint64_t l = 0; l < lvlRank; ++l) {
        if (prevLvlCoords[l] != lvlCoordinates[l]) {
          if (prevLvlCoords[l] > lvlCoordinates[l])
            isSorted = false;
          break;
        }
      }
    }
    lvlCoordinates += lvlRank;
    ++values;
  };
  readNextElement();
  isSorted = true;
  for (uint64_t n = 1; n < nse; ++n)
    readNextElement();
  return isSorted;
}

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_FILE_H
