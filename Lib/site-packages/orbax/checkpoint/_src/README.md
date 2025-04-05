# `_src` package

This package contains (or is intended to contain in the future) the majority of
actual orbax-checkpoint implementations. Code from this directory should not be
directly relied upon by outside users. Instead, depend on symbols exported by
`orbax.checkpoint` or any other subpackages.