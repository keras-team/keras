//===- PassOptions.h - Pass Option Utilities --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities for registering options with compiler passes and
// pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PASSOPTIONS_H_
#define MLIR_PASS_PASSOPTIONS_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include <memory>

namespace mlir {
class OpPassManager;

namespace detail {
namespace pass_options {
/// Parse a string containing a list of comma-delimited elements, invoking the
/// given parser for each sub-element and passing them to the provided
/// element-append functor.
LogicalResult
parseCommaSeparatedList(llvm::cl::Option &opt, StringRef argName,
                        StringRef optionStr,
                        function_ref<LogicalResult(StringRef)> elementParseFn);
template <typename ElementParser, typename ElementAppendFn>
LogicalResult parseCommaSeparatedList(llvm::cl::Option &opt, StringRef argName,
                                      StringRef optionStr,
                                      ElementParser &elementParser,
                                      ElementAppendFn &&appendFn) {
  return parseCommaSeparatedList(
      opt, argName, optionStr, [&](StringRef valueStr) {
        typename ElementParser::parser_data_type value = {};
        if (elementParser.parse(opt, argName, valueStr, value))
          return failure();
        appendFn(value);
        return success();
      });
}

/// Trait used to detect if a type has a operator<< method.
template <typename T>
using has_stream_operator_trait =
    decltype(std::declval<raw_ostream &>() << std::declval<T>());
template <typename T>
using has_stream_operator = llvm::is_detected<has_stream_operator_trait, T>;

/// Utility methods for printing option values.
template <typename ParserT>
static void printOptionValue(raw_ostream &os, const bool &value) {
  os << (value ? StringRef("true") : StringRef("false"));
}
template <typename ParserT>
static void printOptionValue(raw_ostream &os, const std::string &str) {
  // Check if the string needs to be escaped before writing it to the ostream.
  const size_t spaceIndex = str.find_first_of(' ');
  const size_t escapeIndex =
      std::min({str.find_first_of('{'), str.find_first_of('\''),
                str.find_first_of('"')});
  const bool requiresEscape = spaceIndex < escapeIndex;
  if (requiresEscape)
    os << "{";
  os << str;
  if (requiresEscape)
    os << "}";
}
template <typename ParserT, typename DataT>
static std::enable_if_t<has_stream_operator<DataT>::value>
printOptionValue(raw_ostream &os, const DataT &value) {
  os << value;
}
template <typename ParserT, typename DataT>
static std::enable_if_t<!has_stream_operator<DataT>::value>
printOptionValue(raw_ostream &os, const DataT &value) {
  // If the value can't be streamed, fallback to checking for a print in the
  // parser.
  ParserT::print(os, value);
}
} // namespace pass_options

/// Base container class and manager for all pass options.
class PassOptions : protected llvm::cl::SubCommand {
private:
  /// This is the type-erased option base class. This provides some additional
  /// hooks into the options that are not available via llvm::cl::Option.
  class OptionBase {
  public:
    virtual ~OptionBase() = default;

    /// Out of line virtual function to provide home for the class.
    virtual void anchor();

    /// Print the name and value of this option to the given stream.
    virtual void print(raw_ostream &os) = 0;

    /// Return the argument string of this option.
    StringRef getArgStr() const { return getOption()->ArgStr; }

    /// Returns true if this option has any value assigned to it.
    bool hasValue() const { return optHasValue; }

  protected:
    /// Return the main option instance.
    virtual const llvm::cl::Option *getOption() const = 0;

    /// Copy the value from the given option into this one.
    virtual void copyValueFrom(const OptionBase &other) = 0;

    /// Flag indicating if this option has a value.
    bool optHasValue = false;

    /// Allow access to private methods.
    friend PassOptions;
  };

  /// This is the parser that is used by pass options that use literal options.
  /// This is a thin wrapper around the llvm::cl::parser, that exposes some
  /// additional methods.
  template <typename DataType>
  struct GenericOptionParser : public llvm::cl::parser<DataType> {
    using llvm::cl::parser<DataType>::parser;

    /// Returns an argument name that maps to the specified value.
    std::optional<StringRef> findArgStrForValue(const DataType &value) {
      for (auto &it : this->Values)
        if (it.V.compare(value))
          return it.Name;
      return std::nullopt;
    }
  };

  /// This is the parser that is used by pass options that wrap PassOptions
  /// instances. Like GenericOptionParser, this is a thin wrapper around
  /// llvm::cl::basic_parser.
  template <typename PassOptionsT>
  struct PassOptionsParser : public llvm::cl::basic_parser<PassOptionsT> {
    using llvm::cl::basic_parser<PassOptionsT>::basic_parser;
    // Parse the options object by delegating to
    // `PassOptionsT::parseFromString`.
    bool parse(llvm::cl::Option &, StringRef, StringRef arg,
               PassOptionsT &value) {
      return failed(value.parseFromString(arg));
    }

    // Print the options object by delegating to `PassOptionsT::print`.
    static void print(llvm::raw_ostream &os, const PassOptionsT &value) {
      value.print(os);
    }
  };

  /// Utility methods for printing option values.
  template <typename DataT>
  static void printValue(raw_ostream &os, GenericOptionParser<DataT> &parser,
                         const DataT &value) {
    if (std::optional<StringRef> argStr = parser.findArgStrForValue(value))
      os << *argStr;
    else
      llvm_unreachable("unknown data value for option");
  }
  template <typename DataT, typename ParserT>
  static void printValue(raw_ostream &os, ParserT &parser, const DataT &value) {
    detail::pass_options::printOptionValue<ParserT>(os, value);
  }

public:
  /// The specific parser to use. This is necessary because we need to provide
  /// additional methods for certain data type parsers.
  template <typename DataType>
  using OptionParser = std::conditional_t<
      // If the data type is derived from PassOptions, use the
      // PassOptionsParser.
      std::is_base_of_v<PassOptions, DataType>, PassOptionsParser<DataType>,
      // Otherwise, use GenericOptionParser where it is well formed, and fall
      // back to llvm::cl::parser otherwise.
      // TODO: We should upstream the methods in GenericOptionParser to avoid
      // the  need to do this.
      std::conditional_t<std::is_base_of<llvm::cl::generic_parser_base,
                                         llvm::cl::parser<DataType>>::value,
                         GenericOptionParser<DataType>,
                         llvm::cl::parser<DataType>>>;

  /// This class represents a specific pass option, with a provided
  /// data type.
  template <typename DataType, typename OptionParser = OptionParser<DataType>>
  class Option
      : public llvm::cl::opt<DataType, /*ExternalStorage=*/false, OptionParser>,
        public OptionBase {
  public:
    template <typename... Args>
    Option(PassOptions &parent, StringRef arg, Args &&...args)
        : llvm::cl::opt<DataType, /*ExternalStorage=*/false, OptionParser>(
              arg, llvm::cl::sub(parent), std::forward<Args>(args)...) {
      assert(!this->isPositional() && !this->isSink() &&
             "sink and positional options are not supported");
      parent.options.push_back(this);

      // Set a callback to track if this option has a value.
      this->setCallback([this](const auto &) { this->optHasValue = true; });
    }
    ~Option() override = default;
    using llvm::cl::opt<DataType, /*ExternalStorage=*/false,
                        OptionParser>::operator=;
    Option &operator=(const Option &other) {
      *this = other.getValue();
      return *this;
    }

  private:
    /// Return the main option instance.
    const llvm::cl::Option *getOption() const final { return this; }

    /// Print the name and value of this option to the given stream.
    void print(raw_ostream &os) final {
      os << this->ArgStr << '=';
      printValue(os, this->getParser(), this->getValue());
    }

    /// Copy the value from the given option into this one.
    void copyValueFrom(const OptionBase &other) final {
      this->setValue(static_cast<const Option<DataType, OptionParser> &>(other)
                         .getValue());
      optHasValue = other.optHasValue;
    }
  };

  /// This class represents a specific pass option that contains a list of
  /// values of the provided data type. The elements within the textual form of
  /// this option are parsed assuming they are comma-separated. Delimited
  /// sub-ranges within individual elements of the list may contain commas that
  /// are not treated as separators for the top-level list.
  template <typename DataType, typename OptionParser = OptionParser<DataType>>
  class ListOption
      : public llvm::cl::list<DataType, /*StorageClass=*/bool, OptionParser>,
        public OptionBase {
  public:
    template <typename... Args>
    ListOption(PassOptions &parent, StringRef arg, Args &&...args)
        : llvm::cl::list<DataType, /*StorageClass=*/bool, OptionParser>(
              arg, llvm::cl::sub(parent), std::forward<Args>(args)...),
          elementParser(*this) {
      assert(!this->isPositional() && !this->isSink() &&
             "sink and positional options are not supported");
      assert(!(this->getMiscFlags() & llvm::cl::MiscFlags::CommaSeparated) &&
             "ListOption is implicitly comma separated, specifying "
             "CommaSeparated is extraneous");
      parent.options.push_back(this);
      elementParser.initialize();
    }
    ~ListOption() override = default;
    ListOption<DataType, OptionParser> &
    operator=(const ListOption<DataType, OptionParser> &other) {
      *this = ArrayRef<DataType>(other);
      this->optHasValue = other.optHasValue;
      return *this;
    }

    bool handleOccurrence(unsigned pos, StringRef argName,
                          StringRef arg) override {
      if (this->isDefaultAssigned()) {
        this->clear();
        this->overwriteDefault();
      }
      this->optHasValue = true;
      return failed(detail::pass_options::parseCommaSeparatedList(
          *this, argName, arg, elementParser,
          [&](const DataType &value) { this->addValue(value); }));
    }

    /// Allow assigning from an ArrayRef.
    ListOption<DataType, OptionParser> &operator=(ArrayRef<DataType> values) {
      ((std::vector<DataType> &)*this).assign(values.begin(), values.end());
      optHasValue = true;
      return *this;
    }

    /// Allow accessing the data held by this option.
    MutableArrayRef<DataType> operator*() {
      return static_cast<std::vector<DataType> &>(*this);
    }
    ArrayRef<DataType> operator*() const {
      return static_cast<const std::vector<DataType> &>(*this);
    }

  private:
    /// Return the main option instance.
    const llvm::cl::Option *getOption() const final { return this; }

    /// Print the name and value of this option to the given stream.
    void print(raw_ostream &os) final {
      // Don't print the list if empty. An empty option value can be treated as
      // an element of the list in certain cases (e.g. ListOption<std::string>).
      if ((**this).empty())
        return;

      os << this->ArgStr << "={";
      auto printElementFn = [&](const DataType &value) {
        printValue(os, this->getParser(), value);
      };
      llvm::interleave(*this, os, printElementFn, ",");
      os << "}";
    }

    /// Copy the value from the given option into this one.
    void copyValueFrom(const OptionBase &other) final {
      *this = static_cast<const ListOption<DataType, OptionParser> &>(other);
    }

    /// The parser to use for parsing the list elements.
    OptionParser elementParser;
  };

  PassOptions() = default;
  /// Delete the copy constructor to avoid copying the internal options map.
  PassOptions(const PassOptions &) = delete;
  PassOptions(PassOptions &&) = delete;

  /// Copy the option values from 'other' into 'this', where 'other' has the
  /// same options as 'this'.
  void copyOptionValuesFrom(const PassOptions &other);

  /// Parse options out as key=value pairs that can then be handed off to the
  /// `llvm::cl` command line passing infrastructure. Everything is space
  /// separated.
  LogicalResult parseFromString(StringRef options,
                                raw_ostream &errorStream = llvm::errs());

  /// Print the options held by this struct in a form that can be parsed via
  /// 'parseFromString'.
  void print(raw_ostream &os) const;

  /// Print the help string for the options held by this struct. `descIndent` is
  /// the indent that the descriptions should be aligned.
  void printHelp(size_t indent, size_t descIndent) const;

  /// Return the maximum width required when printing the help string.
  size_t getOptionWidth() const;

private:
  /// A list of all of the opaque options.
  std::vector<OptionBase *> options;
};
} // namespace detail

//===----------------------------------------------------------------------===//
// PassPipelineOptions
//===----------------------------------------------------------------------===//

/// Subclasses of PassPipelineOptions provide a set of options that can be used
/// to initialize a pass pipeline. See PassPipelineRegistration for usage
/// details.
///
/// Usage:
///
/// struct MyPipelineOptions : PassPipelineOptions<MyPassOptions> {
///   ListOption<int> someListFlag{*this, "flag-name", llvm::cl::desc("...")};
/// };
template <typename T>
class PassPipelineOptions : public detail::PassOptions {
public:
  /// Factory that parses the provided options and returns a unique_ptr to the
  /// struct.
  static std::unique_ptr<T> createFromString(StringRef options) {
    auto result = std::make_unique<T>();
    if (failed(result->parseFromString(options)))
      return nullptr;
    return result;
  }
};

/// A default empty option struct to be used for passes that do not need to take
/// any options.
struct EmptyPipelineOptions : public PassPipelineOptions<EmptyPipelineOptions> {
};
} // namespace mlir

//===----------------------------------------------------------------------===//
// MLIR Options
//===----------------------------------------------------------------------===//

namespace llvm {
namespace cl {
//===----------------------------------------------------------------------===//
// std::vector+SmallVector

namespace detail {
template <typename VectorT, typename ElementT>
class VectorParserBase : public basic_parser_impl {
public:
  VectorParserBase(Option &opt) : basic_parser_impl(opt), elementParser(opt) {}

  using parser_data_type = VectorT;

  bool parse(Option &opt, StringRef argName, StringRef arg,
             parser_data_type &vector) {
    if (!arg.consume_front("[") || !arg.consume_back("]")) {
      return opt.error("expected vector option to be wrapped with '[]'",
                       argName);
    }

    return failed(mlir::detail::pass_options::parseCommaSeparatedList(
        opt, argName, arg, elementParser,
        [&](const ElementT &value) { vector.push_back(value); }));
  }

  static void print(raw_ostream &os, const VectorT &vector) {
    llvm::interleave(
        vector, os,
        [&](const ElementT &value) {
          mlir::detail::pass_options::printOptionValue<
              llvm::cl::parser<ElementT>>(os, value);
        },
        ",");
  }

  void printOptionInfo(const Option &opt, size_t globalWidth) const {
    // Add the `vector<>` qualifier to the option info.
    outs() << "  --" << opt.ArgStr;
    outs() << "=<vector<" << elementParser.getValueName() << ">>";
    Option::printHelpStr(opt.HelpStr, globalWidth, getOptionWidth(opt));
  }

  size_t getOptionWidth(const Option &opt) const {
    // Add the `vector<>` qualifier to the option width.
    StringRef vectorExt("vector<>");
    return elementParser.getOptionWidth(opt) + vectorExt.size();
  }

private:
  llvm::cl::parser<ElementT> elementParser;
};
} // namespace detail

template <typename T>
class parser<std::vector<T>>
    : public detail::VectorParserBase<std::vector<T>, T> {
public:
  parser(Option &opt) : detail::VectorParserBase<std::vector<T>, T>(opt) {}
};
template <typename T, unsigned N>
class parser<SmallVector<T, N>>
    : public detail::VectorParserBase<SmallVector<T, N>, T> {
public:
  parser(Option &opt) : detail::VectorParserBase<SmallVector<T, N>, T>(opt) {}
};

//===----------------------------------------------------------------------===//
// OpPassManager: OptionValue

template <>
struct OptionValue<mlir::OpPassManager> final : GenericOptionValue {
  using WrapperType = mlir::OpPassManager;

  OptionValue();
  OptionValue(const OptionValue<mlir::OpPassManager> &rhs);
  OptionValue(const mlir::OpPassManager &value);
  OptionValue<mlir::OpPassManager> &operator=(const mlir::OpPassManager &rhs);
  ~OptionValue();

  /// Returns if the current option has a value.
  bool hasValue() const { return value.get(); }

  /// Returns the current value of the option.
  mlir::OpPassManager &getValue() const {
    assert(hasValue() && "invalid option value");
    return *value;
  }

  /// Set the value of the option.
  void setValue(const mlir::OpPassManager &newValue);
  void setValue(StringRef pipelineStr);

  /// Compare the option with the provided value.
  bool compare(const mlir::OpPassManager &rhs) const;
  bool compare(const GenericOptionValue &rhs) const override {
    const auto &rhsOV =
        static_cast<const OptionValue<mlir::OpPassManager> &>(rhs);
    if (!rhsOV.hasValue())
      return false;
    return compare(rhsOV.getValue());
  }

private:
  void anchor() override;

  /// The underlying pass manager. We use a unique_ptr to avoid the need for the
  /// full type definition.
  std::unique_ptr<mlir::OpPassManager> value;
};

//===----------------------------------------------------------------------===//
// OpPassManager: Parser

extern template class basic_parser<mlir::OpPassManager>;

template <>
class parser<mlir::OpPassManager> : public basic_parser<mlir::OpPassManager> {
public:
  /// A utility struct used when parsing a pass manager that prevents the need
  /// for a default constructor on OpPassManager.
  struct ParsedPassManager {
    ParsedPassManager();
    ParsedPassManager(ParsedPassManager &&);
    ~ParsedPassManager();
    operator const mlir::OpPassManager &() const {
      assert(value && "parsed value was invalid");
      return *value;
    }

    std::unique_ptr<mlir::OpPassManager> value;
  };
  using parser_data_type = ParsedPassManager;
  using OptVal = OptionValue<mlir::OpPassManager>;

  parser(Option &opt) : basic_parser(opt) {}

  bool parse(Option &, StringRef, StringRef arg, ParsedPassManager &value);

  /// Print an instance of the underling option value to the given stream.
  static void print(raw_ostream &os, const mlir::OpPassManager &value);

  // Overload in subclass to provide a better default value.
  StringRef getValueName() const override { return "pass-manager"; }

  void printOptionDiff(const Option &opt, mlir::OpPassManager &pm,
                       const OptVal &defaultValue, size_t globalWidth) const;

  // An out-of-line virtual method to provide a 'home' for this class.
  void anchor() override;
};

} // namespace cl
} // namespace llvm

#endif // MLIR_PASS_PASSOPTIONS_H_
