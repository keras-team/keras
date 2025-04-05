/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: OpenMPOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseCancellationConstructType(ClauseCancellationConstructType val) {
  switch (val) {
    case ClauseCancellationConstructType::Parallel: return "parallel";
    case ClauseCancellationConstructType::Loop: return "loop";
    case ClauseCancellationConstructType::Sections: return "sections";
    case ClauseCancellationConstructType::Taskgroup: return "taskgroup";
  }
  return "";
}

::std::optional<ClauseCancellationConstructType> symbolizeClauseCancellationConstructType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ClauseCancellationConstructType>>(str)
      .Case("parallel", ClauseCancellationConstructType::Parallel)
      .Case("loop", ClauseCancellationConstructType::Loop)
      .Case("sections", ClauseCancellationConstructType::Sections)
      .Case("taskgroup", ClauseCancellationConstructType::Taskgroup)
      .Default(::std::nullopt);
}
::std::optional<ClauseCancellationConstructType> symbolizeClauseCancellationConstructType(uint32_t value) {
  switch (value) {
  case 0: return ClauseCancellationConstructType::Parallel;
  case 1: return ClauseCancellationConstructType::Loop;
  case 2: return ClauseCancellationConstructType::Sections;
  case 3: return ClauseCancellationConstructType::Taskgroup;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseDepend(ClauseDepend val) {
  switch (val) {
    case ClauseDepend::dependsource: return "dependsource";
    case ClauseDepend::dependsink: return "dependsink";
  }
  return "";
}

::std::optional<ClauseDepend> symbolizeClauseDepend(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ClauseDepend>>(str)
      .Case("dependsource", ClauseDepend::dependsource)
      .Case("dependsink", ClauseDepend::dependsink)
      .Default(::std::nullopt);
}
::std::optional<ClauseDepend> symbolizeClauseDepend(uint32_t value) {
  switch (value) {
  case 0: return ClauseDepend::dependsource;
  case 1: return ClauseDepend::dependsink;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
std::string stringifyClauseRequires(ClauseRequires symbol) {
  auto val = static_cast<uint32_t>(symbol);
  assert(15u == (15u | val) && "invalid bits set in bit enum");
  // Special case for all bits unset.
  if (val == 0) return "none";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;

  if (1u == (1u & val))
    strs.push_back("reverse_offload");

  if (2u == (2u & val))
    strs.push_back("unified_address");

  if (4u == (4u & val))
    strs.push_back("unified_shared_memory");

  if (8u == (8u & val))
    strs.push_back("dynamic_allocators");
  return ::llvm::join(strs, "|");
}

::std::optional<ClauseRequires> symbolizeClauseRequires(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "none") return ClauseRequires::none;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::std::optional<uint32_t>>(symbol.trim())
      .Case("reverse_offload", 1)
      .Case("unified_address", 2)
      .Case("unified_shared_memory", 4)
      .Case("dynamic_allocators", 8)
      .Default(::std::nullopt);
    if (bit) { val |= *bit; } else { return ::std::nullopt; }
  }
  return static_cast<ClauseRequires>(val);
}

::std::optional<ClauseRequires> symbolizeClauseRequires(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return ClauseRequires::none;

  if (value & ~static_cast<uint32_t>(15u)) return std::nullopt;
  return static_cast<ClauseRequires>(value);
}
} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseTaskDepend(ClauseTaskDepend val) {
  switch (val) {
    case ClauseTaskDepend::taskdependin: return "taskdependin";
    case ClauseTaskDepend::taskdependout: return "taskdependout";
    case ClauseTaskDepend::taskdependinout: return "taskdependinout";
  }
  return "";
}

::std::optional<ClauseTaskDepend> symbolizeClauseTaskDepend(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ClauseTaskDepend>>(str)
      .Case("taskdependin", ClauseTaskDepend::taskdependin)
      .Case("taskdependout", ClauseTaskDepend::taskdependout)
      .Case("taskdependinout", ClauseTaskDepend::taskdependinout)
      .Default(::std::nullopt);
}
::std::optional<ClauseTaskDepend> symbolizeClauseTaskDepend(uint32_t value) {
  switch (value) {
  case 0: return ClauseTaskDepend::taskdependin;
  case 1: return ClauseTaskDepend::taskdependout;
  case 2: return ClauseTaskDepend::taskdependinout;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyDataSharingClauseType(DataSharingClauseType val) {
  switch (val) {
    case DataSharingClauseType::Private: return "private";
    case DataSharingClauseType::FirstPrivate: return "firstprivate";
  }
  return "";
}

::std::optional<DataSharingClauseType> symbolizeDataSharingClauseType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<DataSharingClauseType>>(str)
      .Case("private", DataSharingClauseType::Private)
      .Case("firstprivate", DataSharingClauseType::FirstPrivate)
      .Default(::std::nullopt);
}
::std::optional<DataSharingClauseType> symbolizeDataSharingClauseType(uint32_t value) {
  switch (value) {
  case 0: return DataSharingClauseType::Private;
  case 1: return DataSharingClauseType::FirstPrivate;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyDeclareTargetCaptureClause(DeclareTargetCaptureClause val) {
  switch (val) {
    case DeclareTargetCaptureClause::to: return "to";
    case DeclareTargetCaptureClause::link: return "link";
    case DeclareTargetCaptureClause::enter: return "enter";
  }
  return "";
}

::std::optional<DeclareTargetCaptureClause> symbolizeDeclareTargetCaptureClause(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<DeclareTargetCaptureClause>>(str)
      .Case("to", DeclareTargetCaptureClause::to)
      .Case("link", DeclareTargetCaptureClause::link)
      .Case("enter", DeclareTargetCaptureClause::enter)
      .Default(::std::nullopt);
}
::std::optional<DeclareTargetCaptureClause> symbolizeDeclareTargetCaptureClause(uint32_t value) {
  switch (value) {
  case 0: return DeclareTargetCaptureClause::to;
  case 1: return DeclareTargetCaptureClause::link;
  case 2: return DeclareTargetCaptureClause::enter;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyDeclareTargetDeviceType(DeclareTargetDeviceType val) {
  switch (val) {
    case DeclareTargetDeviceType::any: return "any";
    case DeclareTargetDeviceType::host: return "host";
    case DeclareTargetDeviceType::nohost: return "nohost";
  }
  return "";
}

::std::optional<DeclareTargetDeviceType> symbolizeDeclareTargetDeviceType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<DeclareTargetDeviceType>>(str)
      .Case("any", DeclareTargetDeviceType::any)
      .Case("host", DeclareTargetDeviceType::host)
      .Case("nohost", DeclareTargetDeviceType::nohost)
      .Default(::std::nullopt);
}
::std::optional<DeclareTargetDeviceType> symbolizeDeclareTargetDeviceType(uint32_t value) {
  switch (value) {
  case 0: return DeclareTargetDeviceType::any;
  case 1: return DeclareTargetDeviceType::host;
  case 2: return DeclareTargetDeviceType::nohost;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseGrainsizeType(ClauseGrainsizeType val) {
  switch (val) {
    case ClauseGrainsizeType::Strict: return "strict";
  }
  return "";
}

::std::optional<ClauseGrainsizeType> symbolizeClauseGrainsizeType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ClauseGrainsizeType>>(str)
      .Case("strict", ClauseGrainsizeType::Strict)
      .Default(::std::nullopt);
}
::std::optional<ClauseGrainsizeType> symbolizeClauseGrainsizeType(uint32_t value) {
  switch (value) {
  case 0: return ClauseGrainsizeType::Strict;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseMemoryOrderKind(ClauseMemoryOrderKind val) {
  switch (val) {
    case ClauseMemoryOrderKind::Seq_cst: return "seq_cst";
    case ClauseMemoryOrderKind::Acq_rel: return "acq_rel";
    case ClauseMemoryOrderKind::Acquire: return "acquire";
    case ClauseMemoryOrderKind::Release: return "release";
    case ClauseMemoryOrderKind::Relaxed: return "relaxed";
  }
  return "";
}

::std::optional<ClauseMemoryOrderKind> symbolizeClauseMemoryOrderKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ClauseMemoryOrderKind>>(str)
      .Case("seq_cst", ClauseMemoryOrderKind::Seq_cst)
      .Case("acq_rel", ClauseMemoryOrderKind::Acq_rel)
      .Case("acquire", ClauseMemoryOrderKind::Acquire)
      .Case("release", ClauseMemoryOrderKind::Release)
      .Case("relaxed", ClauseMemoryOrderKind::Relaxed)
      .Default(::std::nullopt);
}
::std::optional<ClauseMemoryOrderKind> symbolizeClauseMemoryOrderKind(uint32_t value) {
  switch (value) {
  case 0: return ClauseMemoryOrderKind::Seq_cst;
  case 1: return ClauseMemoryOrderKind::Acq_rel;
  case 2: return ClauseMemoryOrderKind::Acquire;
  case 3: return ClauseMemoryOrderKind::Release;
  case 4: return ClauseMemoryOrderKind::Relaxed;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseNumTasksType(ClauseNumTasksType val) {
  switch (val) {
    case ClauseNumTasksType::Strict: return "strict";
  }
  return "";
}

::std::optional<ClauseNumTasksType> symbolizeClauseNumTasksType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ClauseNumTasksType>>(str)
      .Case("strict", ClauseNumTasksType::Strict)
      .Default(::std::nullopt);
}
::std::optional<ClauseNumTasksType> symbolizeClauseNumTasksType(uint32_t value) {
  switch (value) {
  case 0: return ClauseNumTasksType::Strict;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseOrderKind(ClauseOrderKind val) {
  switch (val) {
    case ClauseOrderKind::Concurrent: return "concurrent";
  }
  return "";
}

::std::optional<ClauseOrderKind> symbolizeClauseOrderKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ClauseOrderKind>>(str)
      .Case("concurrent", ClauseOrderKind::Concurrent)
      .Default(::std::nullopt);
}
::std::optional<ClauseOrderKind> symbolizeClauseOrderKind(uint32_t value) {
  switch (value) {
  case 1: return ClauseOrderKind::Concurrent;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyOrderModifier(OrderModifier val) {
  switch (val) {
    case OrderModifier::reproducible: return "reproducible";
    case OrderModifier::unconstrained: return "unconstrained";
  }
  return "";
}

::std::optional<OrderModifier> symbolizeOrderModifier(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<OrderModifier>>(str)
      .Case("reproducible", OrderModifier::reproducible)
      .Case("unconstrained", OrderModifier::unconstrained)
      .Default(::std::nullopt);
}
::std::optional<OrderModifier> symbolizeOrderModifier(uint32_t value) {
  switch (value) {
  case 0: return OrderModifier::reproducible;
  case 1: return OrderModifier::unconstrained;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseProcBindKind(ClauseProcBindKind val) {
  switch (val) {
    case ClauseProcBindKind::Primary: return "primary";
    case ClauseProcBindKind::Master: return "master";
    case ClauseProcBindKind::Close: return "close";
    case ClauseProcBindKind::Spread: return "spread";
  }
  return "";
}

::std::optional<ClauseProcBindKind> symbolizeClauseProcBindKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ClauseProcBindKind>>(str)
      .Case("primary", ClauseProcBindKind::Primary)
      .Case("master", ClauseProcBindKind::Master)
      .Case("close", ClauseProcBindKind::Close)
      .Case("spread", ClauseProcBindKind::Spread)
      .Default(::std::nullopt);
}
::std::optional<ClauseProcBindKind> symbolizeClauseProcBindKind(uint32_t value) {
  switch (value) {
  case 0: return ClauseProcBindKind::Primary;
  case 1: return ClauseProcBindKind::Master;
  case 2: return ClauseProcBindKind::Close;
  case 3: return ClauseProcBindKind::Spread;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseScheduleKind(ClauseScheduleKind val) {
  switch (val) {
    case ClauseScheduleKind::Static: return "static";
    case ClauseScheduleKind::Dynamic: return "dynamic";
    case ClauseScheduleKind::Guided: return "guided";
    case ClauseScheduleKind::Auto: return "auto";
    case ClauseScheduleKind::Runtime: return "runtime";
  }
  return "";
}

::std::optional<ClauseScheduleKind> symbolizeClauseScheduleKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ClauseScheduleKind>>(str)
      .Case("static", ClauseScheduleKind::Static)
      .Case("dynamic", ClauseScheduleKind::Dynamic)
      .Case("guided", ClauseScheduleKind::Guided)
      .Case("auto", ClauseScheduleKind::Auto)
      .Case("runtime", ClauseScheduleKind::Runtime)
      .Default(::std::nullopt);
}
::std::optional<ClauseScheduleKind> symbolizeClauseScheduleKind(uint32_t value) {
  switch (value) {
  case 0: return ClauseScheduleKind::Static;
  case 1: return ClauseScheduleKind::Dynamic;
  case 2: return ClauseScheduleKind::Guided;
  case 3: return ClauseScheduleKind::Auto;
  case 4: return ClauseScheduleKind::Runtime;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyScheduleModifier(ScheduleModifier val) {
  switch (val) {
    case ScheduleModifier::none: return "none";
    case ScheduleModifier::monotonic: return "monotonic";
    case ScheduleModifier::nonmonotonic: return "nonmonotonic";
    case ScheduleModifier::simd: return "simd";
  }
  return "";
}

::std::optional<ScheduleModifier> symbolizeScheduleModifier(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ScheduleModifier>>(str)
      .Case("none", ScheduleModifier::none)
      .Case("monotonic", ScheduleModifier::monotonic)
      .Case("nonmonotonic", ScheduleModifier::nonmonotonic)
      .Case("simd", ScheduleModifier::simd)
      .Default(::std::nullopt);
}
::std::optional<ScheduleModifier> symbolizeScheduleModifier(uint32_t value) {
  switch (value) {
  case 0: return ScheduleModifier::none;
  case 1: return ScheduleModifier::monotonic;
  case 2: return ScheduleModifier::nonmonotonic;
  case 3: return ScheduleModifier::simd;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyVariableCaptureKind(VariableCaptureKind val) {
  switch (val) {
    case VariableCaptureKind::This: return "This";
    case VariableCaptureKind::ByRef: return "ByRef";
    case VariableCaptureKind::ByCopy: return "ByCopy";
    case VariableCaptureKind::VLAType: return "VLAType";
  }
  return "";
}

::std::optional<VariableCaptureKind> symbolizeVariableCaptureKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<VariableCaptureKind>>(str)
      .Case("This", VariableCaptureKind::This)
      .Case("ByRef", VariableCaptureKind::ByRef)
      .Case("ByCopy", VariableCaptureKind::ByCopy)
      .Case("VLAType", VariableCaptureKind::VLAType)
      .Default(::std::nullopt);
}
::std::optional<VariableCaptureKind> symbolizeVariableCaptureKind(uint32_t value) {
  switch (value) {
  case 0: return VariableCaptureKind::This;
  case 1: return VariableCaptureKind::ByRef;
  case 2: return VariableCaptureKind::ByCopy;
  case 3: return VariableCaptureKind::VLAType;
  default: return ::std::nullopt;
  }
}

} // namespace omp
} // namespace mlir

