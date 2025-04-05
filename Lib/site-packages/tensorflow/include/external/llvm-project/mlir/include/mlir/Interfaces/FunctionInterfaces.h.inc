/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class FunctionOpInterface;
namespace detail {
struct FunctionOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Type (*getFunctionType)(const Concept *impl, ::mlir::Operation *);
    void (*setFunctionTypeAttr)(const Concept *impl, ::mlir::Operation *, ::mlir::TypeAttr);
    ::mlir::Type (*cloneTypeWith)(const Concept *impl, ::mlir::Operation *, ::mlir::TypeRange, ::mlir::TypeRange);
    ::llvm::LogicalResult (*verifyBody)(const Concept *impl, ::mlir::Operation *);
    ::llvm::LogicalResult (*verifyType)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::SymbolOpInterface::Concept *implSymbolOpInterface = nullptr;
    const ::mlir::CallableOpInterface::Concept *implCallableOpInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implSymbolOpInterface = interfaceMap.lookup<::mlir::SymbolOpInterface>();
      assert(implSymbolOpInterface && "`::mlir::FunctionOpInterface` expected its base interface `::mlir::SymbolOpInterface` to be registered");
      implCallableOpInterface = interfaceMap.lookup<::mlir::CallableOpInterface>();
      assert(implCallableOpInterface && "`::mlir::FunctionOpInterface` expected its base interface `::mlir::CallableOpInterface` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::FunctionOpInterface;
    Model() : Concept{getFunctionType, setFunctionTypeAttr, cloneTypeWith, verifyBody, verifyType} {}

    static inline ::mlir::Type getFunctionType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setFunctionTypeAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeAttr type);
    static inline ::mlir::Type cloneTypeWith(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results);
    static inline ::llvm::LogicalResult verifyBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::LogicalResult verifyType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::FunctionOpInterface;
    FallbackModel() : Concept{getFunctionType, setFunctionTypeAttr, cloneTypeWith, verifyBody, verifyType} {}

    static inline ::mlir::Type getFunctionType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setFunctionTypeAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeAttr type);
    static inline ::mlir::Type cloneTypeWith(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results);
    static inline ::llvm::LogicalResult verifyBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::LogicalResult verifyType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::Type cloneTypeWith(::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results) const;
    ::llvm::LogicalResult verifyBody(::mlir::Operation *tablegen_opaque_val) const;
    ::llvm::LogicalResult verifyType(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct FunctionOpInterfaceTrait;

} // namespace detail
class FunctionOpInterface : public ::mlir::OpInterface<FunctionOpInterface, detail::FunctionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FunctionOpInterface, detail::FunctionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::FunctionOpInterfaceTrait<ConcreteOp> {};
  /// Returns the type of the function.
  ::mlir::Type getFunctionType();
  /// Set the type of the function. This method should perform an unsafe
  /// modification to the function type; it should not update argument or
  /// result attributes.
  void setFunctionTypeAttr(::mlir::TypeAttr type);
  /// Returns a clone of the function type with the given argument and
  /// result types.
  /// 
  /// Note: The default implementation assumes the function type has
  ///       an appropriate clone method:
  ///         `Type clone(ArrayRef<Type> inputs, ArrayRef<Type> results)`
  ::mlir::Type cloneTypeWith(::mlir::TypeRange inputs, ::mlir::TypeRange results);
  /// Verify the contents of the body of this function.
  /// 
  /// Note: The default implementation merely checks that if the entry block
  /// exists, it has the same number and type of arguments as the function type.
  ::llvm::LogicalResult verifyBody();
  /// Verify the type attribute of the function for derived op-specific
  /// invariants.
  ::llvm::LogicalResult verifyType();

    /// Block list iterator types.
    using BlockListType = ::mlir::Region::BlockListType;
    using iterator = BlockListType::iterator;
    using reverse_iterator = BlockListType::reverse_iterator;

    /// Block argument iterator types.
    using BlockArgListType = ::mlir::Region::BlockArgListType;
    using args_iterator = BlockArgListType::iterator;

    //===------------------------------------------------------------------===//
    // Body Handling
    //===------------------------------------------------------------------===//

    /// Returns true if this function is external, i.e. it has no body.
    bool isExternal() { return empty(); }

    /// Return the region containing the body of this function.
    ::mlir::Region &getFunctionBody() { return (*this)->getRegion(0); }

    /// Delete all blocks from this function.
    void eraseBody() {
      getFunctionBody().dropAllReferences();
      getFunctionBody().getBlocks().clear();
    }

    /// Return the list of blocks within the function body.
    BlockListType &getBlocks() { return getFunctionBody().getBlocks(); }

    iterator begin() { return getFunctionBody().begin(); }
    iterator end() { return getFunctionBody().end(); }
    reverse_iterator rbegin() { return getFunctionBody().rbegin(); }
    reverse_iterator rend() { return getFunctionBody().rend(); }

    /// Returns true if this function has no blocks within the body.
    bool empty() { return getFunctionBody().empty(); }

    /// Push a new block to the back of the body region.
    void push_back(::mlir::Block *block) { getFunctionBody().push_back(block); }

    /// Push a new block to the front of the body region.
    void push_front(::mlir::Block *block) { getFunctionBody().push_front(block); }

    /// Return the last block in the body region.
    ::mlir::Block &back() { return getFunctionBody().back(); }

    /// Return the first block in the body region.
    ::mlir::Block &front() { return getFunctionBody().front(); }

    /// Add an entry block to an empty function, and set up the block arguments
    /// to match the signature of the function. The newly inserted entry block
    /// is returned.
    ::mlir::Block *addEntryBlock() {
      assert(empty() && "function already has an entry block");
      ::mlir::Block *entry = new ::mlir::Block();
      push_back(entry);

      // FIXME: Allow for passing in locations for these arguments instead of using
      // the operations location.
      ::llvm::ArrayRef<::mlir::Type> inputTypes = (*this).getArgumentTypes();
      ::llvm::SmallVector<::mlir::Location> locations(inputTypes.size(),
                                              (*this).getOperation()->getLoc());
      entry->addArguments(inputTypes, locations);
      return entry;
    }

    /// Add a normal block to the end of the function's block list. The function
    /// should at least already have an entry block.
    ::mlir::Block *addBlock() {
      assert(!empty() && "function should at least have an entry block");
      push_back(new ::mlir::Block());
      return &back();
    }

    //===------------------------------------------------------------------===//
    // Type Attribute Handling
    //===------------------------------------------------------------------===//

    /// Change the type of this function in place. This is an extremely dangerous
    /// operation and it is up to the caller to ensure that this is legal for
    /// this function, and to restore invariants:
    ///  - the entry block args must be updated to match the function params.
    ///  - the argument/result attributes may need an update: if the new type
    ///    has less parameters we drop the extra attributes, if there are more
    ///    parameters they won't have any attributes.
    void setType(::mlir::Type newType) {
      ::mlir::function_interface_impl::setFunctionType((*this), newType);
    }

    //===------------------------------------------------------------------===//
    // Argument and Result Handling
    //===------------------------------------------------------------------===//

    /// Returns the number of function arguments.
    unsigned getNumArguments() { return (*this).getArgumentTypes().size(); }

    /// Returns the number of function results.
    unsigned getNumResults() { return (*this).getResultTypes().size(); }

    /// Returns the entry block function argument at the given index.
    ::mlir::BlockArgument getArgument(unsigned idx) {
      return getFunctionBody().getArgument(idx);
    }

    /// Support argument iteration.
    args_iterator args_begin() { return getFunctionBody().args_begin(); }
    args_iterator args_end() { return getFunctionBody().args_end(); }
    BlockArgListType getArguments() { return getFunctionBody().getArguments(); }

    /// Insert a single argument of type `argType` with attributes `argAttrs` and
    /// location `argLoc` at `argIndex`.
    void insertArgument(unsigned argIndex, ::mlir::Type argType, ::mlir::DictionaryAttr argAttrs,
                        ::mlir::Location argLoc) {
      insertArguments({argIndex}, {argType}, {argAttrs}, {argLoc});
    }

    /// Inserts arguments with the listed types, attributes, and locations at the
    /// listed indices. `argIndices` must be sorted. Arguments are inserted in the
    /// order they are listed, such that arguments with identical index will
    /// appear in the same order that they were listed here.
    void insertArguments(::llvm::ArrayRef<unsigned> argIndices, ::mlir::TypeRange argTypes,
                        ::llvm::ArrayRef<::mlir::DictionaryAttr> argAttrs,
                        ::llvm::ArrayRef<::mlir::Location> argLocs) {
      unsigned originalNumArgs = (*this).getNumArguments();
      ::mlir::Type newType = (*this).getTypeWithArgsAndResults(
          argIndices, argTypes, /*resultIndices=*/{}, /*resultTypes=*/{});
      ::mlir::function_interface_impl::insertFunctionArguments(
          (*this), argIndices, argTypes, argAttrs, argLocs,
          originalNumArgs, newType);
    }

    /// Insert a single result of type `resultType` at `resultIndex`.
    void insertResult(unsigned resultIndex, ::mlir::Type resultType,
                      ::mlir::DictionaryAttr resultAttrs) {
      insertResults({resultIndex}, {resultType}, {resultAttrs});
    }

    /// Inserts results with the listed types at the listed indices.
    /// `resultIndices` must be sorted. Results are inserted in the order they are
    /// listed, such that results with identical index will appear in the same
    /// order that they were listed here.
    void insertResults(::llvm::ArrayRef<unsigned> resultIndices, ::mlir::TypeRange resultTypes,
                       ::llvm::ArrayRef<::mlir::DictionaryAttr> resultAttrs) {
      unsigned originalNumResults = (*this).getNumResults();
      ::mlir::Type newType = (*this).getTypeWithArgsAndResults(
        /*argIndices=*/{}, /*argTypes=*/{}, resultIndices, resultTypes);
      ::mlir::function_interface_impl::insertFunctionResults(
          (*this), resultIndices, resultTypes, resultAttrs,
          originalNumResults, newType);
    }

    /// Erase a single argument at `argIndex`.
    void eraseArgument(unsigned argIndex) {
      ::llvm::BitVector argsToErase((*this).getNumArguments());
      argsToErase.set(argIndex);
      eraseArguments(argsToErase);
    }

    /// Erases the arguments listed in `argIndices`.
    void eraseArguments(const ::llvm::BitVector &argIndices) {
      ::mlir::Type newType = (*this).getTypeWithoutArgs(argIndices);
      ::mlir::function_interface_impl::eraseFunctionArguments(
        (*this), argIndices, newType);
    }

    /// Erase a single result at `resultIndex`.
    void eraseResult(unsigned resultIndex) {
      ::llvm::BitVector resultsToErase((*this).getNumResults());
      resultsToErase.set(resultIndex);
      eraseResults(resultsToErase);
    }

    /// Erases the results listed in `resultIndices`.
    void eraseResults(const ::llvm::BitVector &resultIndices) {
      ::mlir::Type newType = (*this).getTypeWithoutResults(resultIndices);
      ::mlir::function_interface_impl::eraseFunctionResults(
          (*this), resultIndices, newType);
    }

    /// Return the type of this function with the specified arguments and
    /// results inserted. This is used to update the function's signature in
    /// the `insertArguments` and `insertResults` methods. The arrays must be
    /// sorted by increasing index.
    ::mlir::Type getTypeWithArgsAndResults(
      ::llvm::ArrayRef<unsigned> argIndices, ::mlir::TypeRange argTypes,
      ::llvm::ArrayRef<unsigned> resultIndices, ::mlir::TypeRange resultTypes) {
      ::llvm::SmallVector<::mlir::Type> argStorage, resultStorage;
      ::mlir::TypeRange newArgTypes = insertTypesInto(
          (*this).getArgumentTypes(), argIndices, argTypes, argStorage);
      ::mlir::TypeRange newResultTypes = insertTypesInto(
          (*this).getResultTypes(), resultIndices, resultTypes, resultStorage);
      return (*this).cloneTypeWith(newArgTypes, newResultTypes);
    }

    /// Return the type of this function without the specified arguments and
    /// results. This is used to update the function's signature in the
    /// `eraseArguments` and `eraseResults` methods.
    ::mlir::Type getTypeWithoutArgsAndResults(
      const ::llvm::BitVector &argIndices, const ::llvm::BitVector &resultIndices) {
      ::llvm::SmallVector<::mlir::Type> argStorage, resultStorage;
      ::mlir::TypeRange newArgTypes = filterTypesOut(
          (*this).getArgumentTypes(), argIndices, argStorage);
      ::mlir::TypeRange newResultTypes = filterTypesOut(
          (*this).getResultTypes(), resultIndices, resultStorage);
      return (*this).cloneTypeWith(newArgTypes, newResultTypes);
    }
    ::mlir::Type getTypeWithoutArgs(const ::llvm::BitVector &argIndices) {
      ::llvm::SmallVector<::mlir::Type> argStorage;
      ::mlir::TypeRange newArgTypes = filterTypesOut(
          (*this).getArgumentTypes(), argIndices, argStorage);
      return (*this).cloneTypeWith(newArgTypes, (*this).getResultTypes());
    }
    ::mlir::Type getTypeWithoutResults(const ::llvm::BitVector &resultIndices) {
      ::llvm::SmallVector<::mlir::Type> resultStorage;
      ::mlir::TypeRange newResultTypes = filterTypesOut(
          (*this).getResultTypes(), resultIndices, resultStorage);
      return (*this).cloneTypeWith((*this).getArgumentTypes(), newResultTypes);
    }

    //===------------------------------------------------------------------===//
    // Argument Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the argument at 'index'.
    ::llvm::ArrayRef<::mlir::NamedAttribute> getArgAttrs(unsigned index) {
      return ::mlir::function_interface_impl::getArgAttrs((*this), index);
    }

    /// Return an ArrayAttr containing all argument attribute dictionaries of
    /// this function, or nullptr if no arguments have attributes.
    ::mlir::ArrayAttr getAllArgAttrs() { return (*this).getArgAttrsAttr(); }

    /// Return all argument attributes of this function.
    void getAllArgAttrs(::llvm::SmallVectorImpl<::mlir::DictionaryAttr> &result) {
      if (::mlir::ArrayAttr argAttrs = getAllArgAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<::mlir::DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append((*this).getNumArguments(),
                      ::mlir::DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the argument at 'index',
    /// null otherwise.
    ::mlir::Attribute getArgAttr(unsigned index, ::mlir::StringAttr name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    ::mlir::Attribute getArgAttr(unsigned index, ::llvm::StringRef name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, ::mlir::StringAttr name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getArgAttr(index, name));
    }
    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, ::llvm::StringRef name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getArgAttr(index, name));
    }

    /// Set the attributes held by the argument at 'index'.
    void setArgAttrs(unsigned index, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
      ::mlir::function_interface_impl::setArgAttrs((*this), index, attributes);
    }

    /// Set the attributes held by the argument at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setArgAttrs(unsigned index, ::mlir::DictionaryAttr attributes) {
      ::mlir::function_interface_impl::setArgAttrs((*this), index, attributes);
    }
    void setAllArgAttrs(::llvm::ArrayRef<::mlir::DictionaryAttr> attributes) {
      assert(attributes.size() == (*this).getNumArguments());
      ::mlir::function_interface_impl::setAllArgAttrDicts((*this), attributes);
    }
    void setAllArgAttrs(::llvm::ArrayRef<::mlir::Attribute> attributes) {
      assert(attributes.size() == (*this).getNumArguments());
      ::mlir::function_interface_impl::setAllArgAttrDicts((*this), attributes);
    }
    void setAllArgAttrs(::mlir::ArrayAttr attributes) {
      assert(attributes.size() == (*this).getNumArguments());
      (*this).setArgAttrsAttr(attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setArgAttr(unsigned index, ::mlir::StringAttr name, ::mlir::Attribute value) {
      ::mlir::function_interface_impl::setArgAttr((*this), index, name, value);
    }
    void setArgAttr(unsigned index, ::llvm::StringRef name, ::mlir::Attribute value) {
      setArgAttr(index,
                 ::mlir::StringAttr::get(this->getOperation()->getContext(), name),
                 value);
    }

    /// Remove the attribute 'name' from the argument at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    ::mlir::Attribute removeArgAttr(unsigned index, ::mlir::StringAttr name) {
      return ::mlir::function_interface_impl::removeArgAttr((*this), index, name);
    }
    ::mlir::Attribute removeArgAttr(unsigned index, ::llvm::StringRef name) {
      return removeArgAttr(
          index, ::mlir::StringAttr::get(this->getOperation()->getContext(), name));
    }

    //===------------------------------------------------------------------===//
    // Result Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the result at 'index'.
    ::llvm::ArrayRef<::mlir::NamedAttribute> getResultAttrs(unsigned index) {
      return ::mlir::function_interface_impl::getResultAttrs((*this), index);
    }

    /// Return an ArrayAttr containing all result attribute dictionaries of this
    /// function, or nullptr if no result have attributes.
    ::mlir::ArrayAttr getAllResultAttrs() { return (*this).getResAttrsAttr(); }

    /// Return all result attributes of this function.
    void getAllResultAttrs(::llvm::SmallVectorImpl<::mlir::DictionaryAttr> &result) {
      if (::mlir::ArrayAttr argAttrs = getAllResultAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<::mlir::DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append((*this).getNumResults(),
                      ::mlir::DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the result at 'index',
    /// null otherwise.
    ::mlir::Attribute getResultAttr(unsigned index, ::mlir::StringAttr name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    ::mlir::Attribute getResultAttr(unsigned index, ::llvm::StringRef name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, ::mlir::StringAttr name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getResultAttr(index, name));
    }
    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, ::llvm::StringRef name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getResultAttr(index, name));
    }

    /// Set the attributes held by the result at 'index'.
    void setResultAttrs(unsigned index, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
      ::mlir::function_interface_impl::setResultAttrs((*this), index, attributes);
    }

    /// Set the attributes held by the result at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setResultAttrs(unsigned index, ::mlir::DictionaryAttr attributes) {
      ::mlir::function_interface_impl::setResultAttrs((*this), index, attributes);
    }
    void setAllResultAttrs(::llvm::ArrayRef<::mlir::DictionaryAttr> attributes) {
      assert(attributes.size() == (*this).getNumResults());
      ::mlir::function_interface_impl::setAllResultAttrDicts(
        (*this), attributes);
    }
    void setAllResultAttrs(::llvm::ArrayRef<::mlir::Attribute> attributes) {
      assert(attributes.size() == (*this).getNumResults());
      ::mlir::function_interface_impl::setAllResultAttrDicts(
        (*this), attributes);
    }
    void setAllResultAttrs(::mlir::ArrayAttr attributes) {
      assert(attributes.size() == (*this).getNumResults());
      (*this).setResAttrsAttr(attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setResultAttr(unsigned index, ::mlir::StringAttr name, ::mlir::Attribute value) {
      ::mlir::function_interface_impl::setResultAttr((*this), index, name, value);
    }
    void setResultAttr(unsigned index, ::llvm::StringRef name, ::mlir::Attribute value) {
      setResultAttr(index,
                    ::mlir::StringAttr::get(this->getOperation()->getContext(), name),
                    value);
    }

    /// Remove the attribute 'name' from the result at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    ::mlir::Attribute removeResultAttr(unsigned index, ::mlir::StringAttr name) {
      return ::mlir::function_interface_impl::removeResultAttr((*this), index, name);
    }

    /// Returns the dictionary attribute corresponding to the argument at
    /// 'index'. If there are no argument attributes at 'index', a null
    /// attribute is returned.
    ::mlir::DictionaryAttr getArgAttrDict(unsigned index) {
      assert(index < (*this).getNumArguments() && "invalid argument number");
      return ::mlir::function_interface_impl::getArgAttrDict((*this), index);
    }

    /// Returns the dictionary attribute corresponding to the result at 'index'.
    /// If there are no result attributes at 'index', a null attribute is
    /// returned.
    ::mlir::DictionaryAttr getResultAttrDict(unsigned index) {
      assert(index < (*this).getNumResults() && "invalid result number");
      return ::mlir::function_interface_impl::getResultAttrDict((*this), index);
    }
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::SymbolOpInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::SymbolOpInterface () const {
    if (!*this) return nullptr;
    return ::mlir::SymbolOpInterface(*this, getImpl()->implSymbolOpInterface);
  }

  /// Returns the name of this symbol.
  ::mlir::StringAttr getNameAttr();
  /// Sets the name of this symbol.
  void setName(::mlir::StringAttr name);
  /// Gets the visibility of this symbol.
  mlir::SymbolTable::Visibility getVisibility();
  /// Returns true if this symbol has nested visibility.
  bool isNested();
  /// Returns true if this symbol has private visibility.
  bool isPrivate();
  /// Returns true if this symbol has public visibility.
  bool isPublic();
  /// Sets the visibility of this symbol.
  void setVisibility(mlir::SymbolTable::Visibility vis);
  /// Sets the visibility of this symbol to be nested.
  void setNested();
  /// Sets the visibility of this symbol to be private.
  void setPrivate();
  /// Sets the visibility of this symbol to be public.
  void setPublic();
  /// Get all of the uses of the current symbol that are nested within the
  /// given operation 'from'.
  /// Note: See mlir::SymbolTable::getSymbolUses for more details.
  ::std::optional<::mlir::SymbolTable::UseRange> getSymbolUses(::mlir::Operation * from);
  /// Return if the current symbol is known to have no uses that are nested
  /// within the given operation 'from'.
  /// Note: See mlir::SymbolTable::symbolKnownUseEmpty for more details.
  bool symbolKnownUseEmpty(::mlir::Operation * from);
  /// Attempt to replace all uses of the current symbol with the provided
  /// symbol 'newSymbol' that are nested within the given operation 'from'.
  /// Note: See mlir::SymbolTable::replaceAllSymbolUses for more details.
  ::llvm::LogicalResult replaceAllSymbolUses(::mlir::StringAttr newSymbol, ::mlir::Operation * from);
  /// Returns true if this operation optionally defines a symbol based on the
  /// presence of the symbol name.
  bool isOptionalSymbol();
  /// Returns true if this operation can be discarded if it has no remaining
  /// symbol uses.
  bool canDiscardOnUseEmpty();
  /// Returns true if this operation is a declaration of a symbol (as opposed
  /// to a definition).
  bool isDeclaration();

    using Visibility = mlir::SymbolTable::Visibility;

    /// Convenience version of `getNameAttr` that returns a StringRef.
    ::mlir::StringRef getName() {
      return getNameAttr().getValue();
    }

    /// Convenience version of `setName` that take a StringRef.
    void setName(::mlir::StringRef name) {
      setName(::mlir::StringAttr::get((*this)->getContext(), name));
    }
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::CallableOpInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::CallableOpInterface () const {
    if (!*this) return nullptr;
    return ::mlir::CallableOpInterface(*this, getImpl()->implCallableOpInterface);
  }

  /// Returns the region on the current operation that is callable. This may
  /// return null in the case of an external callable object, e.g. an external
  /// function.
  ::mlir::Region *getCallableRegion();
  /// Returns the callable's argument types based exclusively on the type (to
  /// allow for this method may be called on function declarations).
  ::llvm::ArrayRef<::mlir::Type> getArgumentTypes();
  /// Returns the callable's result types based exclusively on the type (to
  /// allow for this method may be called on function declarations).
  ::llvm::ArrayRef<::mlir::Type> getResultTypes();
  /// Get the array of argument attribute dictionaries. The method should
  /// return an array attribute containing only dictionary attributes equal in
  /// number to the number of region arguments. Alternatively, the method can
  /// return null to indicate that the region has no argument attributes.
  ::mlir::ArrayAttr getArgAttrsAttr();
  /// Get the array of result attribute dictionaries. The method should return
  /// an array attribute containing only dictionary attributes equal in number
  /// to the number of region results. Alternatively, the method can return
  /// null to indicate that the region has no result attributes.
  ::mlir::ArrayAttr getResAttrsAttr();
  /// Set the array of argument attribute dictionaries.
  void setArgAttrsAttr(::mlir::ArrayAttr attrs);
  /// Set the array of result attribute dictionaries.
  void setResAttrsAttr(::mlir::ArrayAttr attrs);
  /// Remove the array of argument attribute dictionaries. This is the same as
  /// setting all argument attributes to an empty dictionary. The method should
  /// return the removed attribute.
  ::mlir::Attribute removeArgAttrsAttr();
  /// Remove the array of result attribute dictionaries. This is the same as
  /// setting all result attributes to an empty dictionary. The method should
  /// return the removed attribute.
  ::mlir::Attribute removeResAttrsAttr();
};
namespace detail {
  template <typename ConcreteOp>
  struct FunctionOpInterfaceTrait : public ::mlir::OpInterface<FunctionOpInterface, detail::FunctionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns a clone of the function type with the given argument and
    /// result types.
    /// 
    /// Note: The default implementation assumes the function type has
    ///       an appropriate clone method:
    ///         `Type clone(ArrayRef<Type> inputs, ArrayRef<Type> results)`
    ::mlir::Type cloneTypeWith(::mlir::TypeRange inputs, ::mlir::TypeRange results) {
      return (*static_cast<ConcreteOp *>(this)).getFunctionType().clone(inputs, results);
    }
    /// Verify the contents of the body of this function.
    /// 
    /// Note: The default implementation merely checks that if the entry block
    /// exists, it has the same number and type of arguments as the function type.
    ::llvm::LogicalResult verifyBody() {
      if ((*static_cast<ConcreteOp *>(this)).isExternal())
        return success();
      ArrayRef<Type> fnInputTypes = (*static_cast<ConcreteOp *>(this)).getArgumentTypes();
      // NOTE: This should just be (*static_cast<ConcreteOp *>(this)).front() but access generically
      // because the interface methods defined here may be shadowed in
      // arbitrary ways. https://github.com/llvm/llvm-project/issues/54807
      Block &entryBlock = (*static_cast<ConcreteOp *>(this))->getRegion(0).front();

      unsigned numArguments = fnInputTypes.size();
      if (entryBlock.getNumArguments() != numArguments)
        return (*static_cast<ConcreteOp *>(this)).emitOpError("entry block must have ")
              << numArguments << " arguments to match function signature";

      for (unsigned i = 0, e = fnInputTypes.size(); i != e; ++i) {
        Type argType = entryBlock.getArgument(i).getType();
        if (fnInputTypes[i] != argType) {
          return (*static_cast<ConcreteOp *>(this)).emitOpError("type of entry block argument #")
                << i << '(' << argType
                << ") must match the type of the corresponding argument in "
                << "function signature(" << fnInputTypes[i] << ')';
        }
      }

      return success();
    }
    /// Verify the type attribute of the function for derived op-specific
    /// invariants.
    ::llvm::LogicalResult verifyType() {
      return success();
    }
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return function_interface_impl::verifyTrait(cast<ConcreteOp>(op));
    }

    //===------------------------------------------------------------------===//
    // Builders
    //===------------------------------------------------------------------===//

    /// Build the function with the given name, attributes, and type. This
    /// builder also inserts an entry block into the function body with the
    /// given argument types.
    static void buildWithEntryBlock(
        OpBuilder &builder, OperationState &state, StringRef name, Type type,
        ArrayRef<NamedAttribute> attrs, TypeRange inputTypes) {
      OpBuilder::InsertionGuard g(builder);
      state.addAttribute(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
      state.addAttribute(ConcreteOp::getFunctionTypeAttrName(state.name),
                        TypeAttr::get(type));
      state.attributes.append(attrs.begin(), attrs.end());

      // Add the function body.
      Region *bodyRegion = state.addRegion();
      Block *body = builder.createBlock(bodyRegion);
      for (Type input : inputTypes)
        body->addArgument(input, state.location);
    }
  

    /// Block list iterator types.
    using BlockListType = ::mlir::Region::BlockListType;
    using iterator = BlockListType::iterator;
    using reverse_iterator = BlockListType::reverse_iterator;

    /// Block argument iterator types.
    using BlockArgListType = ::mlir::Region::BlockArgListType;
    using args_iterator = BlockArgListType::iterator;

    //===------------------------------------------------------------------===//
    // Body Handling
    //===------------------------------------------------------------------===//

    /// Returns true if this function is external, i.e. it has no body.
    bool isExternal() { return empty(); }

    /// Return the region containing the body of this function.
    ::mlir::Region &getFunctionBody() { return (*static_cast<ConcreteOp *>(this))->getRegion(0); }

    /// Delete all blocks from this function.
    void eraseBody() {
      getFunctionBody().dropAllReferences();
      getFunctionBody().getBlocks().clear();
    }

    /// Return the list of blocks within the function body.
    BlockListType &getBlocks() { return getFunctionBody().getBlocks(); }

    iterator begin() { return getFunctionBody().begin(); }
    iterator end() { return getFunctionBody().end(); }
    reverse_iterator rbegin() { return getFunctionBody().rbegin(); }
    reverse_iterator rend() { return getFunctionBody().rend(); }

    /// Returns true if this function has no blocks within the body.
    bool empty() { return getFunctionBody().empty(); }

    /// Push a new block to the back of the body region.
    void push_back(::mlir::Block *block) { getFunctionBody().push_back(block); }

    /// Push a new block to the front of the body region.
    void push_front(::mlir::Block *block) { getFunctionBody().push_front(block); }

    /// Return the last block in the body region.
    ::mlir::Block &back() { return getFunctionBody().back(); }

    /// Return the first block in the body region.
    ::mlir::Block &front() { return getFunctionBody().front(); }

    /// Add an entry block to an empty function, and set up the block arguments
    /// to match the signature of the function. The newly inserted entry block
    /// is returned.
    ::mlir::Block *addEntryBlock() {
      assert(empty() && "function already has an entry block");
      ::mlir::Block *entry = new ::mlir::Block();
      push_back(entry);

      // FIXME: Allow for passing in locations for these arguments instead of using
      // the operations location.
      ::llvm::ArrayRef<::mlir::Type> inputTypes = (*static_cast<ConcreteOp *>(this)).getArgumentTypes();
      ::llvm::SmallVector<::mlir::Location> locations(inputTypes.size(),
                                              (*static_cast<ConcreteOp *>(this)).getOperation()->getLoc());
      entry->addArguments(inputTypes, locations);
      return entry;
    }

    /// Add a normal block to the end of the function's block list. The function
    /// should at least already have an entry block.
    ::mlir::Block *addBlock() {
      assert(!empty() && "function should at least have an entry block");
      push_back(new ::mlir::Block());
      return &back();
    }

    //===------------------------------------------------------------------===//
    // Type Attribute Handling
    //===------------------------------------------------------------------===//

    /// Change the type of this function in place. This is an extremely dangerous
    /// operation and it is up to the caller to ensure that this is legal for
    /// this function, and to restore invariants:
    ///  - the entry block args must be updated to match the function params.
    ///  - the argument/result attributes may need an update: if the new type
    ///    has less parameters we drop the extra attributes, if there are more
    ///    parameters they won't have any attributes.
    void setType(::mlir::Type newType) {
      ::mlir::function_interface_impl::setFunctionType((*static_cast<ConcreteOp *>(this)), newType);
    }

    //===------------------------------------------------------------------===//
    // Argument and Result Handling
    //===------------------------------------------------------------------===//

    /// Returns the number of function arguments.
    unsigned getNumArguments() { return (*static_cast<ConcreteOp *>(this)).getArgumentTypes().size(); }

    /// Returns the number of function results.
    unsigned getNumResults() { return (*static_cast<ConcreteOp *>(this)).getResultTypes().size(); }

    /// Returns the entry block function argument at the given index.
    ::mlir::BlockArgument getArgument(unsigned idx) {
      return getFunctionBody().getArgument(idx);
    }

    /// Support argument iteration.
    args_iterator args_begin() { return getFunctionBody().args_begin(); }
    args_iterator args_end() { return getFunctionBody().args_end(); }
    BlockArgListType getArguments() { return getFunctionBody().getArguments(); }

    /// Insert a single argument of type `argType` with attributes `argAttrs` and
    /// location `argLoc` at `argIndex`.
    void insertArgument(unsigned argIndex, ::mlir::Type argType, ::mlir::DictionaryAttr argAttrs,
                        ::mlir::Location argLoc) {
      insertArguments({argIndex}, {argType}, {argAttrs}, {argLoc});
    }

    /// Inserts arguments with the listed types, attributes, and locations at the
    /// listed indices. `argIndices` must be sorted. Arguments are inserted in the
    /// order they are listed, such that arguments with identical index will
    /// appear in the same order that they were listed here.
    void insertArguments(::llvm::ArrayRef<unsigned> argIndices, ::mlir::TypeRange argTypes,
                        ::llvm::ArrayRef<::mlir::DictionaryAttr> argAttrs,
                        ::llvm::ArrayRef<::mlir::Location> argLocs) {
      unsigned originalNumArgs = (*static_cast<ConcreteOp *>(this)).getNumArguments();
      ::mlir::Type newType = (*static_cast<ConcreteOp *>(this)).getTypeWithArgsAndResults(
          argIndices, argTypes, /*resultIndices=*/{}, /*resultTypes=*/{});
      ::mlir::function_interface_impl::insertFunctionArguments(
          (*static_cast<ConcreteOp *>(this)), argIndices, argTypes, argAttrs, argLocs,
          originalNumArgs, newType);
    }

    /// Insert a single result of type `resultType` at `resultIndex`.
    void insertResult(unsigned resultIndex, ::mlir::Type resultType,
                      ::mlir::DictionaryAttr resultAttrs) {
      insertResults({resultIndex}, {resultType}, {resultAttrs});
    }

    /// Inserts results with the listed types at the listed indices.
    /// `resultIndices` must be sorted. Results are inserted in the order they are
    /// listed, such that results with identical index will appear in the same
    /// order that they were listed here.
    void insertResults(::llvm::ArrayRef<unsigned> resultIndices, ::mlir::TypeRange resultTypes,
                       ::llvm::ArrayRef<::mlir::DictionaryAttr> resultAttrs) {
      unsigned originalNumResults = (*static_cast<ConcreteOp *>(this)).getNumResults();
      ::mlir::Type newType = (*static_cast<ConcreteOp *>(this)).getTypeWithArgsAndResults(
        /*argIndices=*/{}, /*argTypes=*/{}, resultIndices, resultTypes);
      ::mlir::function_interface_impl::insertFunctionResults(
          (*static_cast<ConcreteOp *>(this)), resultIndices, resultTypes, resultAttrs,
          originalNumResults, newType);
    }

    /// Erase a single argument at `argIndex`.
    void eraseArgument(unsigned argIndex) {
      ::llvm::BitVector argsToErase((*static_cast<ConcreteOp *>(this)).getNumArguments());
      argsToErase.set(argIndex);
      eraseArguments(argsToErase);
    }

    /// Erases the arguments listed in `argIndices`.
    void eraseArguments(const ::llvm::BitVector &argIndices) {
      ::mlir::Type newType = (*static_cast<ConcreteOp *>(this)).getTypeWithoutArgs(argIndices);
      ::mlir::function_interface_impl::eraseFunctionArguments(
        (*static_cast<ConcreteOp *>(this)), argIndices, newType);
    }

    /// Erase a single result at `resultIndex`.
    void eraseResult(unsigned resultIndex) {
      ::llvm::BitVector resultsToErase((*static_cast<ConcreteOp *>(this)).getNumResults());
      resultsToErase.set(resultIndex);
      eraseResults(resultsToErase);
    }

    /// Erases the results listed in `resultIndices`.
    void eraseResults(const ::llvm::BitVector &resultIndices) {
      ::mlir::Type newType = (*static_cast<ConcreteOp *>(this)).getTypeWithoutResults(resultIndices);
      ::mlir::function_interface_impl::eraseFunctionResults(
          (*static_cast<ConcreteOp *>(this)), resultIndices, newType);
    }

    /// Return the type of this function with the specified arguments and
    /// results inserted. This is used to update the function's signature in
    /// the `insertArguments` and `insertResults` methods. The arrays must be
    /// sorted by increasing index.
    ::mlir::Type getTypeWithArgsAndResults(
      ::llvm::ArrayRef<unsigned> argIndices, ::mlir::TypeRange argTypes,
      ::llvm::ArrayRef<unsigned> resultIndices, ::mlir::TypeRange resultTypes) {
      ::llvm::SmallVector<::mlir::Type> argStorage, resultStorage;
      ::mlir::TypeRange newArgTypes = insertTypesInto(
          (*static_cast<ConcreteOp *>(this)).getArgumentTypes(), argIndices, argTypes, argStorage);
      ::mlir::TypeRange newResultTypes = insertTypesInto(
          (*static_cast<ConcreteOp *>(this)).getResultTypes(), resultIndices, resultTypes, resultStorage);
      return (*static_cast<ConcreteOp *>(this)).cloneTypeWith(newArgTypes, newResultTypes);
    }

    /// Return the type of this function without the specified arguments and
    /// results. This is used to update the function's signature in the
    /// `eraseArguments` and `eraseResults` methods.
    ::mlir::Type getTypeWithoutArgsAndResults(
      const ::llvm::BitVector &argIndices, const ::llvm::BitVector &resultIndices) {
      ::llvm::SmallVector<::mlir::Type> argStorage, resultStorage;
      ::mlir::TypeRange newArgTypes = filterTypesOut(
          (*static_cast<ConcreteOp *>(this)).getArgumentTypes(), argIndices, argStorage);
      ::mlir::TypeRange newResultTypes = filterTypesOut(
          (*static_cast<ConcreteOp *>(this)).getResultTypes(), resultIndices, resultStorage);
      return (*static_cast<ConcreteOp *>(this)).cloneTypeWith(newArgTypes, newResultTypes);
    }
    ::mlir::Type getTypeWithoutArgs(const ::llvm::BitVector &argIndices) {
      ::llvm::SmallVector<::mlir::Type> argStorage;
      ::mlir::TypeRange newArgTypes = filterTypesOut(
          (*static_cast<ConcreteOp *>(this)).getArgumentTypes(), argIndices, argStorage);
      return (*static_cast<ConcreteOp *>(this)).cloneTypeWith(newArgTypes, (*static_cast<ConcreteOp *>(this)).getResultTypes());
    }
    ::mlir::Type getTypeWithoutResults(const ::llvm::BitVector &resultIndices) {
      ::llvm::SmallVector<::mlir::Type> resultStorage;
      ::mlir::TypeRange newResultTypes = filterTypesOut(
          (*static_cast<ConcreteOp *>(this)).getResultTypes(), resultIndices, resultStorage);
      return (*static_cast<ConcreteOp *>(this)).cloneTypeWith((*static_cast<ConcreteOp *>(this)).getArgumentTypes(), newResultTypes);
    }

    //===------------------------------------------------------------------===//
    // Argument Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the argument at 'index'.
    ::llvm::ArrayRef<::mlir::NamedAttribute> getArgAttrs(unsigned index) {
      return ::mlir::function_interface_impl::getArgAttrs((*static_cast<ConcreteOp *>(this)), index);
    }

    /// Return an ArrayAttr containing all argument attribute dictionaries of
    /// this function, or nullptr if no arguments have attributes.
    ::mlir::ArrayAttr getAllArgAttrs() { return (*static_cast<ConcreteOp *>(this)).getArgAttrsAttr(); }

    /// Return all argument attributes of this function.
    void getAllArgAttrs(::llvm::SmallVectorImpl<::mlir::DictionaryAttr> &result) {
      if (::mlir::ArrayAttr argAttrs = getAllArgAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<::mlir::DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append((*static_cast<ConcreteOp *>(this)).getNumArguments(),
                      ::mlir::DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the argument at 'index',
    /// null otherwise.
    ::mlir::Attribute getArgAttr(unsigned index, ::mlir::StringAttr name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    ::mlir::Attribute getArgAttr(unsigned index, ::llvm::StringRef name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, ::mlir::StringAttr name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getArgAttr(index, name));
    }
    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, ::llvm::StringRef name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getArgAttr(index, name));
    }

    /// Set the attributes held by the argument at 'index'.
    void setArgAttrs(unsigned index, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
      ::mlir::function_interface_impl::setArgAttrs((*static_cast<ConcreteOp *>(this)), index, attributes);
    }

    /// Set the attributes held by the argument at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setArgAttrs(unsigned index, ::mlir::DictionaryAttr attributes) {
      ::mlir::function_interface_impl::setArgAttrs((*static_cast<ConcreteOp *>(this)), index, attributes);
    }
    void setAllArgAttrs(::llvm::ArrayRef<::mlir::DictionaryAttr> attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumArguments());
      ::mlir::function_interface_impl::setAllArgAttrDicts((*static_cast<ConcreteOp *>(this)), attributes);
    }
    void setAllArgAttrs(::llvm::ArrayRef<::mlir::Attribute> attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumArguments());
      ::mlir::function_interface_impl::setAllArgAttrDicts((*static_cast<ConcreteOp *>(this)), attributes);
    }
    void setAllArgAttrs(::mlir::ArrayAttr attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumArguments());
      (*static_cast<ConcreteOp *>(this)).setArgAttrsAttr(attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setArgAttr(unsigned index, ::mlir::StringAttr name, ::mlir::Attribute value) {
      ::mlir::function_interface_impl::setArgAttr((*static_cast<ConcreteOp *>(this)), index, name, value);
    }
    void setArgAttr(unsigned index, ::llvm::StringRef name, ::mlir::Attribute value) {
      setArgAttr(index,
                 ::mlir::StringAttr::get(this->getOperation()->getContext(), name),
                 value);
    }

    /// Remove the attribute 'name' from the argument at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    ::mlir::Attribute removeArgAttr(unsigned index, ::mlir::StringAttr name) {
      return ::mlir::function_interface_impl::removeArgAttr((*static_cast<ConcreteOp *>(this)), index, name);
    }
    ::mlir::Attribute removeArgAttr(unsigned index, ::llvm::StringRef name) {
      return removeArgAttr(
          index, ::mlir::StringAttr::get(this->getOperation()->getContext(), name));
    }

    //===------------------------------------------------------------------===//
    // Result Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the result at 'index'.
    ::llvm::ArrayRef<::mlir::NamedAttribute> getResultAttrs(unsigned index) {
      return ::mlir::function_interface_impl::getResultAttrs((*static_cast<ConcreteOp *>(this)), index);
    }

    /// Return an ArrayAttr containing all result attribute dictionaries of this
    /// function, or nullptr if no result have attributes.
    ::mlir::ArrayAttr getAllResultAttrs() { return (*static_cast<ConcreteOp *>(this)).getResAttrsAttr(); }

    /// Return all result attributes of this function.
    void getAllResultAttrs(::llvm::SmallVectorImpl<::mlir::DictionaryAttr> &result) {
      if (::mlir::ArrayAttr argAttrs = getAllResultAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<::mlir::DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append((*static_cast<ConcreteOp *>(this)).getNumResults(),
                      ::mlir::DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the result at 'index',
    /// null otherwise.
    ::mlir::Attribute getResultAttr(unsigned index, ::mlir::StringAttr name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    ::mlir::Attribute getResultAttr(unsigned index, ::llvm::StringRef name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, ::mlir::StringAttr name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getResultAttr(index, name));
    }
    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, ::llvm::StringRef name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getResultAttr(index, name));
    }

    /// Set the attributes held by the result at 'index'.
    void setResultAttrs(unsigned index, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
      ::mlir::function_interface_impl::setResultAttrs((*static_cast<ConcreteOp *>(this)), index, attributes);
    }

    /// Set the attributes held by the result at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setResultAttrs(unsigned index, ::mlir::DictionaryAttr attributes) {
      ::mlir::function_interface_impl::setResultAttrs((*static_cast<ConcreteOp *>(this)), index, attributes);
    }
    void setAllResultAttrs(::llvm::ArrayRef<::mlir::DictionaryAttr> attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumResults());
      ::mlir::function_interface_impl::setAllResultAttrDicts(
        (*static_cast<ConcreteOp *>(this)), attributes);
    }
    void setAllResultAttrs(::llvm::ArrayRef<::mlir::Attribute> attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumResults());
      ::mlir::function_interface_impl::setAllResultAttrDicts(
        (*static_cast<ConcreteOp *>(this)), attributes);
    }
    void setAllResultAttrs(::mlir::ArrayAttr attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumResults());
      (*static_cast<ConcreteOp *>(this)).setResAttrsAttr(attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setResultAttr(unsigned index, ::mlir::StringAttr name, ::mlir::Attribute value) {
      ::mlir::function_interface_impl::setResultAttr((*static_cast<ConcreteOp *>(this)), index, name, value);
    }
    void setResultAttr(unsigned index, ::llvm::StringRef name, ::mlir::Attribute value) {
      setResultAttr(index,
                    ::mlir::StringAttr::get(this->getOperation()->getContext(), name),
                    value);
    }

    /// Remove the attribute 'name' from the result at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    ::mlir::Attribute removeResultAttr(unsigned index, ::mlir::StringAttr name) {
      return ::mlir::function_interface_impl::removeResultAttr((*static_cast<ConcreteOp *>(this)), index, name);
    }

    /// Returns the dictionary attribute corresponding to the argument at
    /// 'index'. If there are no argument attributes at 'index', a null
    /// attribute is returned.
    ::mlir::DictionaryAttr getArgAttrDict(unsigned index) {
      assert(index < (*static_cast<ConcreteOp *>(this)).getNumArguments() && "invalid argument number");
      return ::mlir::function_interface_impl::getArgAttrDict((*static_cast<ConcreteOp *>(this)), index);
    }

    /// Returns the dictionary attribute corresponding to the result at 'index'.
    /// If there are no result attributes at 'index', a null attribute is
    /// returned.
    ::mlir::DictionaryAttr getResultAttrDict(unsigned index) {
      assert(index < (*static_cast<ConcreteOp *>(this)).getNumResults() && "invalid result number");
      return ::mlir::function_interface_impl::getResultAttrDict((*static_cast<ConcreteOp *>(this)), index);
    }
  
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getFunctionType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFunctionType();
}
template<typename ConcreteOp>
void detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::setFunctionTypeAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeAttr type) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setFunctionTypeAttr(type);
}
template<typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::cloneTypeWith(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).cloneTypeWith(inputs, results);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyBody();
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyType();
}
template<typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getFunctionType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getFunctionType(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setFunctionTypeAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeAttr type) {
  return static_cast<const ConcreteOp *>(impl)->setFunctionTypeAttr(tablegen_opaque_val, type);
}
template<typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::cloneTypeWith(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results) {
  return static_cast<const ConcreteOp *>(impl)->cloneTypeWith(tablegen_opaque_val, inputs, results);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyBody(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyType(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::cloneTypeWith(::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFunctionType().clone(inputs, results);
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyBody(::mlir::Operation *tablegen_opaque_val) const {
if ((llvm::cast<ConcreteOp>(tablegen_opaque_val)).isExternal())
        return success();
      ArrayRef<Type> fnInputTypes = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArgumentTypes();
      // NOTE: This should just be (llvm::cast<ConcreteOp>(tablegen_opaque_val)).front() but access generically
      // because the interface methods defined here may be shadowed in
      // arbitrary ways. https://github.com/llvm/llvm-project/issues/54807
      Block &entryBlock = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getRegion(0).front();

      unsigned numArguments = fnInputTypes.size();
      if (entryBlock.getNumArguments() != numArguments)
        return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitOpError("entry block must have ")
              << numArguments << " arguments to match function signature";

      for (unsigned i = 0, e = fnInputTypes.size(); i != e; ++i) {
        Type argType = entryBlock.getArgument(i).getType();
        if (fnInputTypes[i] != argType) {
          return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitOpError("type of entry block argument #")
                << i << '(' << argType
                << ") must match the type of the corresponding argument in "
                << "function signature(" << fnInputTypes[i] << ')';
        }
      }

      return success();
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyType(::mlir::Operation *tablegen_opaque_val) const {
return success();
}
} // namespace mlir
