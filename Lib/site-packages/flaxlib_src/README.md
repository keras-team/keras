# flaxlib

## Build flaxlib from source

Install necessary dependencies to build the C++ based package.

```shell
pip install meson-python ninja build
```

Clone the Flax repository, navigate to the flaxlib source directory.

```shell
git clone git@github.com:google/flax.git
cd flax/flaxlib_src
```

Configure the build.

```shell
mkdir -p subprojects
meson wrap install robin-map
meson wrap install nanobind
meson setup builddir
```

Compile the code. You'll need to run this repeatedly if you modify the source
code. Note that the actual wheel name will differ depending on your system.

```shell
meson compile -C builddir
python -m build . -w
pip install dist/flaxlib-0.0.1-cp311-cp311-macosx_14_0_arm64.whl --force-reinstall
```
