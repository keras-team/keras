//===- ADTExtras.h - Extra ADTs for use in MLIR -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_ADTEXTRAS_H
#define MLIR_SUPPORT_ADTEXTRAS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// CopyOnWriteArrayRef<T>
//===----------------------------------------------------------------------===//

// A wrapper around an ArrayRef<T> that copies to a SmallVector<T> on
// modification. This is for use in the mlir::<Type>::Builders.
template <typename T>
class CopyOnWriteArrayRef {
public:
  CopyOnWriteArrayRef(ArrayRef<T> array) : nonOwning(array){};

  CopyOnWriteArrayRef &operator=(ArrayRef<T> array) {
    nonOwning = array;
    owningStorage = {};
    return *this;
  }

  void insert(size_t index, T value) {
    SmallVector<T> &vector = ensureCopy();
    vector.insert(vector.begin() + index, value);
  }

  void erase(size_t index) {
    // Note: A copy can be avoided when just dropping the front/back dims.
    if (isNonOwning() && index == 0) {
      nonOwning = nonOwning.drop_front();
    } else if (isNonOwning() && index == size() - 1) {
      nonOwning = nonOwning.drop_back();
    } else {
      SmallVector<T> &vector = ensureCopy();
      vector.erase(vector.begin() + index);
    }
  }

  void set(size_t index, T value) { ensureCopy()[index] = value; }

  size_t size() const { return ArrayRef<T>(*this).size(); }

  bool empty() const { return ArrayRef<T>(*this).empty(); }

  operator ArrayRef<T>() const {
    return nonOwning.empty() ? ArrayRef<T>(owningStorage) : nonOwning;
  }

private:
  bool isNonOwning() const { return !nonOwning.empty(); }

  SmallVector<T> &ensureCopy() {
    // Empty non-owning storage signals the array has been copied to the owning
    // storage (or both are empty). Note: `nonOwning` should never reference
    // `owningStorage`. This can lead to dangling references if the
    // CopyOnWriteArrayRef<T> is copied.
    if (isNonOwning()) {
      owningStorage = SmallVector<T>(nonOwning);
      nonOwning = {};
    }
    return owningStorage;
  }

  ArrayRef<T> nonOwning;
  SmallVector<T> owningStorage;
};

} // namespace mlir

#endif
