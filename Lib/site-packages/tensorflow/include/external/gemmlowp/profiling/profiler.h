// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// profiler.h: a simple sampling profiler that's always just one #include away!
//
// Overview
// ========
//
// This profiler only samples a pseudo-stack, not the actual call stack.
// The code to be profiled needs to be instrumented with
// pseudo-stack "labels", see ScopedProfilingLabel.
// Using pseudo-stacks allows this profiler to be very simple, low-overhead,
// portable, and independent of compilation details such as function inlining
// and frame pointers. The granularity of instrumentation can be freely chosen,
// and it is possible to get some annotate-like detail, i.e. detail within one
// function without splitting it into multiple functions.
//
// This profiler should remain small and simple; its key feature is to fit in
// a single header file so that there should never be a reason to refrain
// from profiling. More complex and feature-rich alternatives are
// readily available. This one offers a strict superset of its
// functionality: https://github.com/bgirard/GeckoProfiler, including
// intertwining pseudostacks with real call stacks, more annotation options,
// and advanced visualization.
//
// Usage
// =====
//
// 0. Enable profiling by defining GEMMLOWP_PROFILING. When profiling is
//    not enabled, profiling instrumentation from instrumentation.h
//    (ScopedProfilingLabel, RegisterCurrentThreadForProfiling)
//    is still defined but does nothing. On the other hand,
//    when profiling is not enabled, it is an error to #include the
//    present file.
//
// 1. Each thread can opt in to profiling by calling
//    RegisterCurrentThreadForProfiling() defined in instrumentation.h.
//    This can be done at any time, before or during profiling.
//    No sample will be collected from a thread until
//    it has called RegisterCurrentThreadForProfiling().
//
// 2. Instrument your code to be profiled with ScopedProfilingLabel,
//    which is a RAII helper defined in instrumentation.h. The identifier
//    names (some_label, etc) do not matter; what will show up
//    in the profile is the string passed to the constructor, which
//    must be a literal string. See the full example below.
//
//    Note: the overhead of ScopedProfilingLabel is zero when not
//    enabling profiling (when not defining GEMMLOWP_PROFILING).
//
// 3. Use the profiler.h interface to control profiling. There are two
//    functions: StartProfiling() and FinishProfiling(). They must be
//    called on the same thread. FinishProfiling() prints the profile
//    on stdout.
//
// Full example
// ============
/*
    #define GEMMLOWP_PROFILING
    #include "profiling/instrumentation.h"
    using namespace gemmlowp;

    const int iters = 100000000;
    volatile int i;

    void Bar() {
      ScopedProfilingLabel label("Bar");
      for (i = 0; i < iters; i++) {}
    }

    void Foo() {
      ScopedProfilingLabel label("Foo");
      for (i = 0; i < iters; i++) {}
      Bar();
    }

    void Init() {
      RegisterCurrentThreadForProfiling();
    }

    #include "profiling/profiler.h"

    int main() {
      Init();
      StartProfiling();
      Foo();
      FinishProfiling();
    }
*
* Output:
*
    gemmlowp profile (1 threads, 304 samples)
    100.00% Foo
        51.32% other
        48.68% Bar
    0.00% other (outside of any label)
*/
//
// Interpreting results
// ====================
//
//  Each node shows the absolute percentage, among all the samples,
//  of the number of samples that recorded the given pseudo-stack.
//  The percentages are *NOT* relative to the parent node. In addition
//  to your own labels, you will also see 'other' nodes that collect
//  the remainder of samples under the parent node that didn't fall into
//  any of the labelled child nodes. Example:
//
//  20% Foo
//      12% Bar
//      6% Xyz
//      2% other
//
//  This means that 20% of all labels were under Foo, of which 12%/20%==60%
//  were under Bar, 6%/20%==30% were under Xyz, and 2%/20%==10% were not
//  under either Bar or Xyz.
//
//  Typically, one wants to keep adding ScopedProfilingLabel's until
//  the 'other' nodes show low percentages.
//
// Interpreting results with multiple threads
// ==========================================
//
// At each sample, each thread registered for profiling gets sampled once.
// So if there is one "main thread" spending its time in MainFunc() and
// 4 "worker threads" spending time in WorkerFunc(), then 80% (=4/5) of the
// samples will be in WorkerFunc, so the profile will look like this:
//
// 80% WorkerFunc
// 20% MainFunc

#ifndef GEMMLOWP_PROFILING_PROFILER_H_
#define GEMMLOWP_PROFILING_PROFILER_H_

#ifndef GEMMLOWP_PROFILING
#error Profiling is not enabled!
#endif

#include <vector>

#include "instrumentation.h"

namespace gemmlowp {

// A tree view of a profile.
class ProfileTreeView {
  struct Node {
    std::vector<Node*> children;
    const char* label;
    std::size_t weight;
    Node() : label(nullptr), weight(0) {}
    ~Node() {
      for (auto child : children) {
        delete child;
      }
    }
  };

  static bool CompareNodes(Node* n1, Node* n2) {
    return n1->weight > n2->weight;
  }

  Node root_;

  void PrintNode(const Node* node, int level) const {
    if (level) {
      for (int i = 1; i < level; i++) {
        printf("    ");
      }
      printf("%.2f%% %s\n", 100.0f * node->weight / root_.weight, node->label);
    }
    for (auto child : node->children) {
      PrintNode(child, level + 1);
    }
  }

  static void AddStackToNode(const ProfilingStack& stack, Node* node,
                             std::size_t level) {
    node->weight++;
    if (stack.size == level) {
      return;
    }
    Node* child_to_add_to = nullptr;
    for (auto child : node->children) {
      if (child->label == stack.labels[level]) {
        child_to_add_to = child;
        break;
      }
    }
    if (!child_to_add_to) {
      child_to_add_to = new Node;
      child_to_add_to->label = stack.labels[level];
      node->children.push_back(child_to_add_to);
    }
    AddStackToNode(stack, child_to_add_to, level + 1);
    return;
  }

  void AddStack(const ProfilingStack& stack) {
    AddStackToNode(stack, &root_, 0);
  }

  void AddOtherChildrenToNode(Node* node) {
    std::size_t top_level_children_weight = 0;
    for (auto c : node->children) {
      AddOtherChildrenToNode(c);
      top_level_children_weight += c->weight;
    }
    if (top_level_children_weight) {
      Node* other_child = new Node;
      other_child->label =
          node == &root_ ? "other (outside of any label)" : "other";
      other_child->weight = node->weight - top_level_children_weight;
      node->children.push_back(other_child);
    }
  }

  void AddOtherNodes() { AddOtherChildrenToNode(&root_); }

  void SortNode(Node* node) {
    std::sort(node->children.begin(), node->children.end(), CompareNodes);
    for (auto child : node->children) {
      SortNode(child);
    }
  }

  void Sort() { SortNode(&root_); }

 public:
  explicit ProfileTreeView(const std::vector<ProfilingStack>& stacks) {
    for (auto stack : stacks) {
      AddStack(stack);
    }
    AddOtherNodes();
    Sort();
  }

  void Print() const {
    printf("\n");
    printf("gemmlowp profile (%d threads, %d samples)\n",
           static_cast<int>(ThreadsUnderProfiling().size()),
           static_cast<int>(root_.weight));
    PrintNode(&root_, 0);
    printf("\n");
  }
};

// This function is the only place that determines our sampling frequency.
inline void WaitOneProfilerTick() {
  static const int millisecond = 1000000;

#if defined __arm__ || defined __aarch64__
  // Reduced sampling frequency on mobile devices helps limit time and memory
  // overhead there.
  static const int interval = 10 * millisecond;
#else
  static const int interval = 1 * millisecond;
#endif

  timespec ts;
  ts.tv_sec = 0;
  ts.tv_nsec = interval;
  nanosleep(&ts, nullptr);
}

// This is how we track whether we've already started profiling,
// to guard against misuse of the API.
inline bool& IsProfiling() {
  static bool b;
  return b;
}

// This is how we tell the profiler thread to finish.
inline bool& ProfilerThreadShouldFinish() {
  static bool b;
  return b;
}

// The profiler thread. See ProfilerThreadFunc.
inline pthread_t& ProfilerThread() {
  static pthread_t t;
  return t;
}

// Records a stack from a running thread.
// The tricky part is that we're not interrupting the thread.
// This is OK because we're looking at a pseudo-stack of labels,
// not at the real thread stack, and if the pseudo-stack changes
// while we're recording it, we are OK with getting either the
// old or the new stack. Note that ProfilingStack::Pop
// only decrements the size, and doesn't null the popped label,
// so if we're concurrently recording it, it shouldn't change
// under our feet until another label is pushed, at which point
// we are OK with getting either this new label or the old one.
// In the end, the key atomicity property that we are relying on
// here is that pointers are changed atomically, and the labels
// are pointers (to literal strings).
inline void RecordStack(ThreadInfo* thread, ProfilingStack* dst) {
  ScopedLock sl(thread->stack.lock);
  assert(!dst->size);
  while (dst->size < thread->stack.size) {
    dst->labels[dst->size] = thread->stack.labels[dst->size];
    dst->size++;
  }
}

// The profiler thread's entry point.
// Note that a separate thread is to be started each time we call
// StartProfiling(), and finishes when we call FinishProfiling().
// So here we only need to handle the recording and reporting of
// a single profile.
inline void* ProfilerThreadFunc(void*) {
  assert(ProfilerThread() == pthread_self());

  // Since we only handle one profile per profiler thread, the
  // profile data (the array of recorded stacks) can be a local variable here.
  std::vector<ProfilingStack> stacks;

  while (!ProfilerThreadShouldFinish()) {
    WaitOneProfilerTick();
    {
      ScopedLock sl(GlobalMutexes::Profiler());
      for (auto t : ThreadsUnderProfiling()) {
        ProfilingStack s;
        RecordStack(t, &s);
        stacks.push_back(s);
      }
    }
  }

  // Profiling is finished and we now report the results.
  ProfileTreeView(stacks).Print();

  return nullptr;
}

// Starts recording samples.
inline void StartProfiling() {
  ScopedLock sl(GlobalMutexes::Profiler());
  ReleaseBuildAssertion(!IsProfiling(), "We're already profiling!");
  IsProfiling() = true;
  ProfilerThreadShouldFinish() = false;
  pthread_create(&ProfilerThread(), nullptr, ProfilerThreadFunc, nullptr);
}

// Stops recording samples, and prints a profile tree-view on stdout.
inline void FinishProfiling() {
  {
    ScopedLock sl(GlobalMutexes::Profiler());
    ReleaseBuildAssertion(IsProfiling(), "We weren't profiling!");
    // The ProfilerThreadShouldFinish() mechanism here is really naive and bad,
    // as the scary comments below should make clear.
    // Should we use a condition variable?
    ProfilerThreadShouldFinish() = true;
  }  // must release the lock here to avoid deadlock with profiler thread.
  pthread_join(ProfilerThread(), nullptr);
  IsProfiling() = false;  // yikes, this should be guarded by the lock!
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_PROFILING_PROFILER_H_
