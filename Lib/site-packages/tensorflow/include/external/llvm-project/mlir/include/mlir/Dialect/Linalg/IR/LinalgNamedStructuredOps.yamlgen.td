
//===----------------------------------------------------------------------===//
// Op definition for CopyOp
//===----------------------------------------------------------------------===//

def CopyOp : LinalgStructuredBase_Op<"copy", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Copies the tensor elementwise.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, CopyOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, CopyOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$cast,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("cast", cast);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, CopyOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    let hasCanonicalizer = 1;


    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for ElemwiseUnaryOp
//===----------------------------------------------------------------------===//

def ElemwiseUnaryOp : LinalgStructuredBase_Op<"elemwise_unary", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies the unary function fun elementwise.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<UnaryFnAttr, "UnaryFn::exp">:$fun,
DefaultValuedOptionalAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, ElemwiseUnaryOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, ElemwiseUnaryOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$fun, "Attribute":$cast,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("fun", fun);
$_state.addAttribute("cast", cast);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, ElemwiseUnaryOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for ExpOp
//===----------------------------------------------------------------------===//

def ExpOp : LinalgStructuredBase_Op<"exp", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies exp(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, ExpOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, ExpOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for LogOp
//===----------------------------------------------------------------------===//

def LogOp : LinalgStructuredBase_Op<"log", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies log(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, LogOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, LogOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for AbsOp
//===----------------------------------------------------------------------===//

def AbsOp : LinalgStructuredBase_Op<"abs", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies abs(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, AbsOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, AbsOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for CeilOp
//===----------------------------------------------------------------------===//

def CeilOp : LinalgStructuredBase_Op<"ceil", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies ceil(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, CeilOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, CeilOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for FloorOp
//===----------------------------------------------------------------------===//

def FloorOp : LinalgStructuredBase_Op<"floor", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies floor(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, FloorOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, FloorOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for NegFOp
//===----------------------------------------------------------------------===//

def NegFOp : LinalgStructuredBase_Op<"negf", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies negf(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, NegFOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, NegFOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for ReciprocalOp
//===----------------------------------------------------------------------===//

def ReciprocalOp : LinalgStructuredBase_Op<"reciprocal", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies reciprocal(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, ReciprocalOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, ReciprocalOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for RoundOp
//===----------------------------------------------------------------------===//

def RoundOp : LinalgStructuredBase_Op<"round", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies round(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, RoundOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, RoundOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for SqrtOp
//===----------------------------------------------------------------------===//

def SqrtOp : LinalgStructuredBase_Op<"sqrt", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies sqrt(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, SqrtOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, SqrtOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for RsqrtOp
//===----------------------------------------------------------------------===//

def RsqrtOp : LinalgStructuredBase_Op<"rsqrt", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies rsqrt(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, RsqrtOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, RsqrtOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for SquareOp
//===----------------------------------------------------------------------===//

def SquareOp : LinalgStructuredBase_Op<"square", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies square(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, SquareOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, SquareOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for TanhOp
//===----------------------------------------------------------------------===//

def TanhOp : LinalgStructuredBase_Op<"tanh", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies tanh(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, TanhOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, TanhOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for ErfOp
//===----------------------------------------------------------------------===//

def ErfOp : LinalgStructuredBase_Op<"erf", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies erf(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, ErfOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, ErfOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for ElemwiseBinaryOp
//===----------------------------------------------------------------------===//

def ElemwiseBinaryOp : LinalgStructuredBase_Op<"elemwise_binary", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Applies the binary function fun elementwise.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<BinaryFnAttr, "BinaryFn::add">:$fun,
DefaultValuedOptionalAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, ElemwiseBinaryOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, ElemwiseBinaryOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$fun, "Attribute":$cast,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("fun", fun);
$_state.addAttribute("cast", cast);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, ElemwiseBinaryOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for AddOp
//===----------------------------------------------------------------------===//

def AddOp : LinalgStructuredBase_Op<"add", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Adds two tensors elementwise.}];
  let description = [{The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.add` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, AddOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, AddOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for SubOp
//===----------------------------------------------------------------------===//

def SubOp : LinalgStructuredBase_Op<"sub", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Subtracts two tensors elementwise.}];
  let description = [{The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.sub` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, SubOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, SubOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MulOp
//===----------------------------------------------------------------------===//

def MulOp : LinalgStructuredBase_Op<"mul", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Multiplies two tensors elementwise.}];
  let description = [{The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.mul` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, MulOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, MulOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DivOp
//===----------------------------------------------------------------------===//

def DivOp : LinalgStructuredBase_Op<"div", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Divides the first tensor by the second tensor, elementwise.}];
  let description = [{The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.div` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DivOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DivOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DivUnsignedOp
//===----------------------------------------------------------------------===//

def DivUnsignedOp : LinalgStructuredBase_Op<"div_unsigned", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Divides the first tensor by the second tensor, elementwise. For integer
types, performs an unsigned division.}];
  let description = [{The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.div` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DivUnsignedOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DivUnsignedOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MaxOp
//===----------------------------------------------------------------------===//

def MaxOp : LinalgStructuredBase_Op<"max", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Takes the max (signed) between two inputs, elementwise.}];
  let description = [{The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.max` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, MaxOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, MaxOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MinOp
//===----------------------------------------------------------------------===//

def MinOp : LinalgStructuredBase_Op<"min", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Takes the min (signed) between two inputs, elementwise.}];
  let description = [{The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.min` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, MinOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, MinOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PowFOp
//===----------------------------------------------------------------------===//

def PowFOp : LinalgStructuredBase_Op<"powf", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Takes the powf(lhs, rhs) between two inputs, elementwise. For powf(arg, 2) use `linalg.square`.}];
  let description = [{Only applies to floating point values.

The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.powf` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PowFOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PowFOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for SelectOp
//===----------------------------------------------------------------------===//

def SelectOp : LinalgStructuredBase_Op<"select", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Chooses one value based on a binary condition supplied as its first operand.}];
  let description = [{The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.select` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, SelectOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, SelectOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MatmulOp
//===----------------------------------------------------------------------===//

def MatmulOp : LinalgStructuredBase_Op<"matmul", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a matrix multiplication of two 2D inputs.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, MatmulOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, MatmulOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$cast,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("cast", cast);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, MatmulOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for QuantizedMatmulOp
//===----------------------------------------------------------------------===//

def QuantizedMatmulOp : LinalgStructuredBase_Op<"quantized_matmul", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Performs a matrix multiplication of two 2D inputs.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. The quantized variant
includes zero-point adjustments for the left and right operands of the
matmul.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, QuantizedMatmulOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, QuantizedMatmulOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MatmulTransposeAOp
//===----------------------------------------------------------------------===//

def MatmulTransposeAOp : LinalgStructuredBase_Op<"matmul_transpose_a", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a matrix multiplication of two 2D inputs with lhs operand
transposed.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, MatmulTransposeAOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, MatmulTransposeAOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$cast,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("cast", cast);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, MatmulTransposeAOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MatmulTransposeBOp
//===----------------------------------------------------------------------===//

def MatmulTransposeBOp : LinalgStructuredBase_Op<"matmul_transpose_b", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a matrix multiplication of two 2D inputs with rhs operand
transposed.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, MatmulTransposeBOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, MatmulTransposeBOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$cast,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("cast", cast);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, MatmulTransposeBOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Mmt4DOp
//===----------------------------------------------------------------------===//

def Mmt4DOp : LinalgStructuredBase_Op<"mmt4d", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a matrix-matrix-transpose multiplication of two 4D inputs.}];
  let description = [{Differences from linalg.matmul:
* The right hand side is transposed, whence the 't' in 'mmt'.
* The input and output tensors have a 4D shape instead of a 2D shape. They
  are interpreted as 2D matrices with one level of 2D tile subdivision,
  whence the 2+2=4 dimensions. The inner tile dimensions are identified with
  '0' suffixes below, for instance the LHS matrix shape (M, K, M0, K0) reads
  as: MxK tiles, each of shape M0xK0.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Mmt4DOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Mmt4DOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchMmt4DOp
//===----------------------------------------------------------------------===//

def BatchMmt4DOp : LinalgStructuredBase_Op<"batch_mmt4d", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a batched matrix-matrix-transpose multiplication of two
batched-4D (5D) inputs.}];
  let description = [{Besides the outermost batch dimension has the same semantic as
linalg.batch_matmul, the differences from linalg.batch_matmul in the
non-batch dimensions are the same as linalg.mmt4d vs. linalg.matmul. See the
description of lingalg.mmt4d.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, BatchMmt4DOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, BatchMmt4DOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchMatmulOp
//===----------------------------------------------------------------------===//

def BatchMatmulOp : LinalgStructuredBase_Op<"batch_matmul", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a batched matrix multiplication of two 3D inputs.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, BatchMatmulOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, BatchMatmulOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchMatmulTransposeAOp
//===----------------------------------------------------------------------===//

def BatchMatmulTransposeAOp : LinalgStructuredBase_Op<"batch_matmul_transpose_a", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a batched matrix multiplication of two 3D inputs where lhs operand
has its non-batch dimensions transposed.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, BatchMatmulTransposeAOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, BatchMatmulTransposeAOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchMatmulTransposeBOp
//===----------------------------------------------------------------------===//

def BatchMatmulTransposeBOp : LinalgStructuredBase_Op<"batch_matmul_transpose_b", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a batched matrix multiplication of two 3D inputs where rhs operand
has its non-batch dimensions transposed.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, BatchMatmulTransposeBOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, BatchMatmulTransposeBOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for QuantizedBatchMatmulOp
//===----------------------------------------------------------------------===//

def QuantizedBatchMatmulOp : LinalgStructuredBase_Op<"quantized_batch_matmul", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Performs a batched matrix multiplication of two 3D inputs.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. The quantized variant
includes zero-point adjustments for the left and right operands of the
matmul.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, QuantizedBatchMatmulOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, QuantizedBatchMatmulOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchReduceMatmulOp
//===----------------------------------------------------------------------===//

def BatchReduceMatmulOp : LinalgStructuredBase_Op<"batch_reduce_matmul", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a batch-reduce matrix multiplication of two 3D inputs.
The partial multiplication results are reduced into a 2D output.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, BatchReduceMatmulOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, BatchReduceMatmulOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MatvecOp
//===----------------------------------------------------------------------===//

def MatvecOp : LinalgStructuredBase_Op<"matvec", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a matrix-vector multiplication.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, MatvecOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, MatvecOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for VecmatOp
//===----------------------------------------------------------------------===//

def VecmatOp : LinalgStructuredBase_Op<"vecmat", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a vector-matrix multiplication.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, VecmatOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, VecmatOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchMatvecOp
//===----------------------------------------------------------------------===//

def BatchMatvecOp : LinalgStructuredBase_Op<"batch_matvec", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a batched matrix-vector multiplication.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, BatchMatvecOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, BatchMatvecOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchVecmatOp
//===----------------------------------------------------------------------===//

def BatchVecmatOp : LinalgStructuredBase_Op<"batch_vecmat", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a batched matrix-vector multiplication.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, BatchVecmatOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, BatchVecmatOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DotOp
//===----------------------------------------------------------------------===//

def DotOp : LinalgStructuredBase_Op<"dot", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{Performs a dot product of two vectors to a scalar result.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DotOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DotOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv1DOp
//===----------------------------------------------------------------------===//

def Conv1DOp : LinalgStructuredBase_Op<"conv_1d", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 1-D convolution with no channels.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv1DOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv1DOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DOp
//===----------------------------------------------------------------------===//

def Conv2DOp : LinalgStructuredBase_Op<"conv_2d", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 2-D convolution with no channels.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv2DOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv2DOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv3DOp
//===----------------------------------------------------------------------===//

def Conv3DOp : LinalgStructuredBase_Op<"conv_3d", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 3-D convolution with no channels.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv3DOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv3DOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv1DNwcWcfOp
//===----------------------------------------------------------------------===//

def Conv1DNwcWcfOp : LinalgStructuredBase_Op<"conv_1d_nwc_wcf", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 1-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv1DNwcWcfOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv1DNwcWcfOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv1DNwcWcfOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv1DNcwFcwOp
//===----------------------------------------------------------------------===//

def Conv1DNcwFcwOp : LinalgStructuredBase_Op<"conv_1d_ncw_fcw", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 1-D convolution.}];
  let description = [{Layout:
  * Input: NCW.
  * Kernel: FCW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv1DNcwFcwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv1DNcwFcwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv1DNcwFcwOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNhwcHwcfOp
//===----------------------------------------------------------------------===//

def Conv2DNhwcHwcfOp : LinalgStructuredBase_Op<"conv_2d_nhwc_hwcf", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 2-D convolution.}];
  let description = [{Layout:
  * Input: NHWC.
  * Kernel: HWCF.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv2DNhwcHwcfOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNhwcHwcfOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNhwcHwcfOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNhwcFhwcOp
//===----------------------------------------------------------------------===//

def Conv2DNhwcFhwcOp : LinalgStructuredBase_Op<"conv_2d_nhwc_fhwc", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 2-D convolution.}];
  let description = [{Layout:
  * Input: NHWC.
  * Kernel: FHWC.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv2DNhwcFhwcOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNhwcFhwcOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNhwcFhwcOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNhwcHwcfQOp
//===----------------------------------------------------------------------===//

def Conv2DNhwcHwcfQOp : LinalgStructuredBase_Op<"conv_2d_nhwc_hwcf_q", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 2-D convolution with zero point offsets.}];
  let description = [{Layout:
  * Input: NHWC.
  * Kernel: HWCF.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. This includes the zero
point offsets common to quantized operations.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv2DNhwcHwcfQOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNhwcHwcfQOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNhwcHwcfQOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNhwcFhwcQOp
//===----------------------------------------------------------------------===//

def Conv2DNhwcFhwcQOp : LinalgStructuredBase_Op<"conv_2d_nhwc_fhwc_q", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 2-D convolution with zero point offsets.}];
  let description = [{Layout:
  * Input: NHWC.
  * Kernel: FHWC.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. This includes the zero
point offsets common to quantized operations.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv2DNhwcFhwcQOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNhwcFhwcQOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNhwcFhwcQOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNchwFchwOp
//===----------------------------------------------------------------------===//

def Conv2DNchwFchwOp : LinalgStructuredBase_Op<"conv_2d_nchw_fchw", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 2-D convolution.}];
  let description = [{Layout:
  * Input: NCHW.
  * Kernel: FCHW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv2DNchwFchwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNchwFchwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNchwFchwOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNgchwFgchwOp
//===----------------------------------------------------------------------===//

def Conv2DNgchwFgchwOp : LinalgStructuredBase_Op<"conv_2d_ngchw_fgchw", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 2-D grouped convolution.}];
  let description = [{Layout:
  * Input: NGCHW.
  * Kernel: FGCHW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv2DNgchwFgchwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNgchwFgchwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNgchwFgchwOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNgchwGfchwOp
//===----------------------------------------------------------------------===//

def Conv2DNgchwGfchwOp : LinalgStructuredBase_Op<"conv_2d_ngchw_gfchw", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 2-D grouped convolution.}];
  let description = [{Layout:
  * Input: NGCHW.
  * Kernel: GFCHW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv2DNgchwGfchwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNgchwGfchwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNgchwGfchwOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNgchwGfchwQOp
//===----------------------------------------------------------------------===//

def Conv2DNgchwGfchwQOp : LinalgStructuredBase_Op<"conv_2d_ngchw_gfchw_q", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 2-D grouped convolution with zero-point offsets.}];
  let description = [{Layout:
  * Input: NGCHW.
  * Kernel: GFCHW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. This includes the zero
point offsets common to quantized operations.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv2DNgchwGfchwQOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNgchwGfchwQOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNgchwGfchwQOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv3DNdhwcDhwcfOp
//===----------------------------------------------------------------------===//

def Conv3DNdhwcDhwcfOp : LinalgStructuredBase_Op<"conv_3d_ndhwc_dhwcf", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 3-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv3DNdhwcDhwcfOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv3DNdhwcDhwcfOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv3DNdhwcDhwcfOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv3DNdhwcDhwcfQOp
//===----------------------------------------------------------------------===//

def Conv3DNdhwcDhwcfQOp : LinalgStructuredBase_Op<"conv_3d_ndhwc_dhwcf_q", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 3-D convolution with zero point offsets.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. This includes the zero
point offsets common to quantized operations.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv3DNdhwcDhwcfQOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv3DNdhwcDhwcfQOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv3DNdhwcDhwcfQOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv3DNcdhwFcdhwOp
//===----------------------------------------------------------------------===//

def Conv3DNcdhwFcdhwOp : LinalgStructuredBase_Op<"conv_3d_ncdhw_fcdhw", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 3-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, Conv3DNcdhwFcdhwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Conv3DNcdhwFcdhwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, Conv3DNcdhwFcdhwOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv1DNwcWcOp
//===----------------------------------------------------------------------===//

def DepthwiseConv1DNwcWcOp : LinalgStructuredBase_Op<"depthwise_conv_1d_nwc_wc", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs depth-wise 1-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv1DNwcWcOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv1DNwcWcOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv1DNwcWcOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv1DNcwCwOp
//===----------------------------------------------------------------------===//

def DepthwiseConv1DNcwCwOp : LinalgStructuredBase_Op<"depthwise_conv_1d_ncw_cw", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs depth-wise 1-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv1DNcwCwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv1DNcwCwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv1DNcwCwOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv1DNwcWcmOp
//===----------------------------------------------------------------------===//

def DepthwiseConv1DNwcWcmOp : LinalgStructuredBase_Op<"depthwise_conv_1d_nwc_wcm", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs depth-wise 1-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv1DNwcWcmOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv1DNwcWcmOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv1DNwcWcmOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNhwcHwcOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNhwcHwcOp : LinalgStructuredBase_Op<"depthwise_conv_2d_nhwc_hwc", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs depth-wise 2-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv2DNhwcHwcOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv2DNhwcHwcOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv2DNhwcHwcOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNchwChwOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNchwChwOp : LinalgStructuredBase_Op<"depthwise_conv_2d_nchw_chw", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs depth-wise 2-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv2DNchwChwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv2DNchwChwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv2DNchwChwOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNhwcHwcQOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNhwcHwcQOp : LinalgStructuredBase_Op<"depthwise_conv_2d_nhwc_hwc_q", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs depth-wise 2-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv2DNhwcHwcQOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv2DNhwcHwcQOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv2DNhwcHwcQOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNhwcHwcmOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNhwcHwcmOp : LinalgStructuredBase_Op<"depthwise_conv_2d_nhwc_hwcm", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs depth-wise 2-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv2DNhwcHwcmOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv2DNhwcHwcmOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv2DNhwcHwcmOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNhwcHwcmQOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNhwcHwcmQOp : LinalgStructuredBase_Op<"depthwise_conv_2d_nhwc_hwcm_q", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs depth-wise 2-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv2DNhwcHwcmQOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv2DNhwcHwcmQOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv2DNhwcHwcmQOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv3DNdhwcDhwcOp
//===----------------------------------------------------------------------===//

def DepthwiseConv3DNdhwcDhwcOp : LinalgStructuredBase_Op<"depthwise_conv_3d_ndhwc_dhwc", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs depth-wise 3-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv3DNdhwcDhwcOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv3DNdhwcDhwcOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv3DNdhwcDhwcOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv3DNcdhwCdhwOp
//===----------------------------------------------------------------------===//

def DepthwiseConv3DNcdhwCdhwOp : LinalgStructuredBase_Op<"depthwise_conv_3d_ncdhw_cdhw", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs depth-wise 3-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv3DNcdhwCdhwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv3DNcdhwCdhwOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv3DNcdhwCdhwOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv3DNdhwcDhwcmOp
//===----------------------------------------------------------------------===//

def DepthwiseConv3DNdhwcDhwcmOp : LinalgStructuredBase_Op<"depthwise_conv_3d_ndhwc_dhwcm", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs depth-wise 3-D convolution.}];
  let description = [{Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv3DNdhwcDhwcmOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv3DNdhwcDhwcmOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv3DNdhwcDhwcmOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcSumOp
//===----------------------------------------------------------------------===//

def PoolingNhwcSumOp : LinalgStructuredBase_Op<"pooling_nhwc_sum", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs sum pooling.}];
  let description = [{Layout:
  * Input: NHWC.
  * Kernel: HW.

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNhwcSumOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNhwcSumOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNhwcSumOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNchwSumOp
//===----------------------------------------------------------------------===//

def PoolingNchwSumOp : LinalgStructuredBase_Op<"pooling_nchw_sum", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs sum pooling.}];
  let description = [{Layout:
  * Input: NCHW.
  * Kernel: HW.

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNchwSumOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNchwSumOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNchwSumOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcMaxOp
//===----------------------------------------------------------------------===//

def PoolingNhwcMaxOp : LinalgStructuredBase_Op<"pooling_nhwc_max", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs max pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNhwcMaxOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNhwcMaxOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNhwcMaxOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcMaxUnsignedOp
//===----------------------------------------------------------------------===//

def PoolingNhwcMaxUnsignedOp : LinalgStructuredBase_Op<"pooling_nhwc_max_unsigned", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs unsigned max pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNhwcMaxUnsignedOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNhwcMaxUnsignedOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNhwcMaxUnsignedOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNchwMaxOp
//===----------------------------------------------------------------------===//

def PoolingNchwMaxOp : LinalgStructuredBase_Op<"pooling_nchw_max", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs max pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNchwMaxOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNchwMaxOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNchwMaxOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcMinOp
//===----------------------------------------------------------------------===//

def PoolingNhwcMinOp : LinalgStructuredBase_Op<"pooling_nhwc_min", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs min pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNhwcMinOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNhwcMinOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNhwcMinOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcMinUnsignedOp
//===----------------------------------------------------------------------===//

def PoolingNhwcMinUnsignedOp : LinalgStructuredBase_Op<"pooling_nhwc_min_unsigned", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs unsigned min pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNhwcMinUnsignedOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNhwcMinUnsignedOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNhwcMinUnsignedOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNwcSumOp
//===----------------------------------------------------------------------===//

def PoolingNwcSumOp : LinalgStructuredBase_Op<"pooling_nwc_sum", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs sum pooling.}];
  let description = [{Layout:
  * Input: NWC.
  * Kernel: W.

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNwcSumOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNwcSumOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNwcSumOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNcwSumOp
//===----------------------------------------------------------------------===//

def PoolingNcwSumOp : LinalgStructuredBase_Op<"pooling_ncw_sum", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs sum pooling.}];
  let description = [{Layout:
  * Input: NCW.
  * Kernel: W.

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNcwSumOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNcwSumOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNcwSumOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNwcMaxOp
//===----------------------------------------------------------------------===//

def PoolingNwcMaxOp : LinalgStructuredBase_Op<"pooling_nwc_max", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs max pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNwcMaxOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNwcMaxOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNwcMaxOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNwcMaxUnsignedOp
//===----------------------------------------------------------------------===//

def PoolingNwcMaxUnsignedOp : LinalgStructuredBase_Op<"pooling_nwc_max_unsigned", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs unsigned max pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNwcMaxUnsignedOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNwcMaxUnsignedOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNwcMaxUnsignedOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNcwMaxOp
//===----------------------------------------------------------------------===//

def PoolingNcwMaxOp : LinalgStructuredBase_Op<"pooling_ncw_max", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs max pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNcwMaxOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNcwMaxOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNcwMaxOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNwcMinOp
//===----------------------------------------------------------------------===//

def PoolingNwcMinOp : LinalgStructuredBase_Op<"pooling_nwc_min", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs min pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNwcMinOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNwcMinOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNwcMinOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNwcMinUnsignedOp
//===----------------------------------------------------------------------===//

def PoolingNwcMinUnsignedOp : LinalgStructuredBase_Op<"pooling_nwc_min_unsigned", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs unsigned min pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNwcMinUnsignedOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNwcMinUnsignedOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNwcMinUnsignedOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNdhwcSumOp
//===----------------------------------------------------------------------===//

def PoolingNdhwcSumOp : LinalgStructuredBase_Op<"pooling_ndhwc_sum", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 3D sum pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNdhwcSumOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNdhwcSumOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNdhwcSumOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNdhwcMaxOp
//===----------------------------------------------------------------------===//

def PoolingNdhwcMaxOp : LinalgStructuredBase_Op<"pooling_ndhwc_max", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 3D max pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNdhwcMaxOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNdhwcMaxOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNdhwcMaxOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNdhwcMinOp
//===----------------------------------------------------------------------===//

def PoolingNdhwcMinOp : LinalgStructuredBase_Op<"pooling_ndhwc_min", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{Performs 3D min pooling.}];
  let description = [{Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedOptionalAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, PoolingNdhwcMinOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PoolingNdhwcMinOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, PoolingNdhwcMinOp::getRegionBuilder());
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for FillOp
//===----------------------------------------------------------------------===//

def FillOp : LinalgStructuredBase_Op<"fill", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgFillOpInterface])> {
    
  let summary = [{Fills the output tensor with the given value.}];
  let description = [{Works for arbitrary ranked output tensors since the operation performs scalar
accesses only and is thus rank polymorphic. Numeric casting is performed on
the value operand, promoting it to the same data type as the output.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, FillOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, FillOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    let hasCanonicalizer = 1;


    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for FillRng2DOp
//===----------------------------------------------------------------------===//

def FillRng2DOp : LinalgStructuredBase_Op<"fill_rng_2d", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{Fills the output tensor with pseudo random numbers.}];
  let description = [{The operation generations pseudo random numbers using a linear congruential
generator. It provides no guarantees regarding the distribution of the
generated random numbers. Instead of generating the random numbers
sequentially, it instantiates one random number generator per data element
and runs them in parallel. The seed operand and the indices of the data
element seed the random number generation. The min and max operands limit
the range of the generated random numbers.}];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, FillRng2DOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, FillRng2DOp::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}
