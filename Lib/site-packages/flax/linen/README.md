# Linen: A comfortable evolution of Flax

Linen is a neural network API developed based on learning from our users and the broader JAX community. Linen improves on much of the former `flax.nn` API (removed since v0.4.0), such as submodule sharing and better support for non-trainable variables.
Moreover, Linen builds on a "functional core", enabling direct usage of JAX transformations such as `vmap`, `remat` or `scan` inside your modules.

In Linen, Modules behave much closer to vanilla Python objects, while still letting you opt-in to the concise single-method pattern many of our users love.

The Linen Module API is stable and currently recommended for new projects. We are already supporting users in the OSS community and within Google. Minor changes may come to the top-level `apply` and `init` patterns, which we will communicate clearly. We plan a few improvements, including writing up short design notes, adding more design tests (see last link below), and an API for interactive module instances.

Please open a [discussion](https://github.com/google/flax/discussions) if you have any questions or thoughts.

**See the [Linen API reference docs](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/index.html)**, or take a look at our additional material:

* 2-page intro to the [Linen Design Principles](https://docs.google.com/document/d/1ZlL_4bXCw5Xl0WstQw1GpnZqfb9JFOeUGAPcBVk-kn8)
* [Slides from a talk to the JAX core team](https://docs.google.com/presentation/d/1ngKWUwsSqAwPRvATG8sAxMzu9ujv4N__cKsUofdNno0)
* [Brief Intro to Linen](https://colab.research.google.com/github/google/flax/blob/main/docs/linen_intro.ipynb) in Colab
* An [upgrade guide](https://docs.google.com/document/d/1hYavTVPaKVVe9Be8pCB7yW7r6dDv3RALVNit8NZca4c) + some additional questions we're considering
* Ported [examples](https://github.com/google/flax/tree/main/examples)
* "Design tests" used to ensure that our "functional core" supports [various advanced use-cases](https://github.com/google/flax/tree/main/tests/core/design), and that the mostly-syntactic-sugar Module abstraction
  [doesn't get in the way](https://github.com/google/flax/tree/main/examples/linen_design_test)
