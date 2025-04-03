//
//  Copyright 2021 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Do not include this file directly.
// Include absl/flags/flag.h instead.

// MSVC debug builds do not implement initialization with constexpr constructors
// correctly. To work around this we add a level of indirection, so that the
// class `absl::Flag` contains an `internal::Flag*` (instead of being an alias
// to that class) and dynamically allocates an instance when necessary. We also
// forward all calls to internal::Flag methods via trampoline methods. In this
// setup the `absl::Flag` class does not have constructor and virtual methods,
// all the data members are public and thus MSVC is able to initialize it at
// link time. To deal with multiple threads accessing the flag for the first
// time concurrently we use an atomic boolean indicating if flag object is
// initialized. We also employ the double-checked locking pattern where the
// second level of protection is a global Mutex, so if two threads attempt to
// construct the flag concurrently only one wins.
//
// This solution is based on a recommendation here:
// https://developercommunity.visualstudio.com/content/problem/336946/class-with-constexpr-constructor-not-using-static.html?childToView=648454#comment-648454

namespace flags_internal {
absl::Mutex* GetGlobalConstructionGuard();
}  // namespace flags_internal

// Public methods of `absl::Flag<T>` are NOT part of the Abseil Flags API.
// See https://abseil.io/docs/cpp/guides/flags
template <typename T>
class Flag {
 public:
  // No constructor and destructor to ensure this is an aggregate type.
  // Visual Studio 2015 still requires the constructor for class to be
  // constexpr initializable.
#if _MSC_VER <= 1900
  constexpr Flag(const char* name, const char* filename,
                 const flags_internal::HelpGenFunc help_gen,
                 const flags_internal::FlagDfltGenFunc default_value_gen)
      : name_(name),
        filename_(filename),
        help_gen_(help_gen),
        default_value_gen_(default_value_gen),
        inited_(false),
        impl_(nullptr) {}
#endif

  flags_internal::Flag<T>& GetImpl() const {
    if (!inited_.load(std::memory_order_acquire)) {
      absl::MutexLock l(flags_internal::GetGlobalConstructionGuard());

      if (inited_.load(std::memory_order_acquire)) {
        return *impl_;
      }

      impl_ = new flags_internal::Flag<T>(
          name_, filename_,
          {flags_internal::FlagHelpMsg(help_gen_),
           flags_internal::FlagHelpKind::kGenFunc},
          {flags_internal::FlagDefaultSrc(default_value_gen_),
           flags_internal::FlagDefaultKind::kGenFunc});
      inited_.store(true, std::memory_order_release);
    }

    return *impl_;
  }

  // Public methods of `absl::Flag<T>` are NOT part of the Abseil Flags API.
  // See https://abseil.io/docs/cpp/guides/flags
  bool IsRetired() const { return GetImpl().IsRetired(); }
  absl::string_view Name() const { return GetImpl().Name(); }
  std::string Help() const { return GetImpl().Help(); }
  bool IsModified() const { return GetImpl().IsModified(); }
  bool IsSpecifiedOnCommandLine() const {
    return GetImpl().IsSpecifiedOnCommandLine();
  }
  std::string Filename() const { return GetImpl().Filename(); }
  std::string DefaultValue() const { return GetImpl().DefaultValue(); }
  std::string CurrentValue() const { return GetImpl().CurrentValue(); }
  template <typename U>
  inline bool IsOfType() const {
    return GetImpl().template IsOfType<U>();
  }
  T Get() const {
    return flags_internal::FlagImplPeer::InvokeGet<T>(GetImpl());
  }
  void Set(const T& v) {
    flags_internal::FlagImplPeer::InvokeSet(GetImpl(), v);
  }
  void InvokeCallback() { GetImpl().InvokeCallback(); }

  const CommandLineFlag& Reflect() const {
    return flags_internal::FlagImplPeer::InvokeReflect(GetImpl());
  }

  // The data members are logically private, but they need to be public for
  // this to be an aggregate type.
  const char* name_;
  const char* filename_;
  const flags_internal::HelpGenFunc help_gen_;
  const flags_internal::FlagDfltGenFunc default_value_gen_;

  mutable std::atomic<bool> inited_;
  mutable flags_internal::Flag<T>* impl_;
};
