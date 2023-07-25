"""Script to create (and optionally install) a `.whl` archive for Keras Core.

Usage:

1. Create a `.whl` file in `dist/`:

```
python3 pip_build.py
```

2. Also install the new package immediately after:

```
python3 pip_build.py --install
```
"""
import argparse
import glob
import os
import pathlib
import shutil

import namex

# Needed because importing torch after TF causes the runtime to crash
import torch  # noqa: F401

package = "keras_core"
build_directory = "tmp_build_dir"
dist_directory = "dist"
to_copy = ["setup.py", "README.md"]


def ignore_files(_, filenames):
    return [f for f in filenames if f.endswith("_test.py")]


def build():
    if os.path.exists(build_directory):
        raise ValueError(f"Directory already exists: {build_directory}")

    whl_path = None
    try:
        # Copy sources (`keras_core/` directory and setup files) to build
        # directory
        root_path = pathlib.Path(__file__).parent.resolve()
        os.chdir(root_path)
        os.mkdir(build_directory)
        shutil.copytree(
            package, os.path.join(build_directory, package), ignore=ignore_files
        )
        for fname in to_copy:
            shutil.copy(fname, os.path.join(f"{build_directory}", fname))
        os.chdir(build_directory)

        # Restructure the codebase so that source files live in `keras_core/src`
        namex.convert_codebase(package, code_directory="src")

        # Generate API __init__.py files in `keras_core/`
        namex.generate_api_files(package, code_directory="src", verbose=True)

        # Make keras_core/_tf_keras/ by copying keras_core/
        tf_keras_dirpath = os.path.join(package, "_tf_keras")
        os.makedirs(tf_keras_dirpath)
        with open(os.path.join(package, "__init__.py")) as f:
            init_file = f.read()
            init_file = init_file.replace(
                "from keras_core import _legacy",
                "from keras_core import _tf_keras",
            )
        with open(os.path.join(package, "__init__.py"), "w") as f:
            f.write(init_file)
        with open(os.path.join(tf_keras_dirpath, "__init__.py"), "w") as f:
            f.write(init_file)
        for dirname in os.listdir(package):
            dirpath = os.path.join(package, dirname)
            if os.path.isdir(dirpath) and dirname not in (
                "_legacy",
                "_tf_keras",
                "src",
            ):
                shutil.copytree(
                    dirpath,
                    os.path.join(tf_keras_dirpath, dirname),
                    ignore=ignore_files,
                )

        # Copy keras_core/_legacy/ file contents to keras_core/_tf_keras/
        legacy_submodules = [
            path[:-3]
            for path in os.listdir(os.path.join(package, "src", "legacy"))
            if path.endswith(".py")
        ]
        legacy_submodules += [
            path
            for path in os.listdir(os.path.join(package, "src", "legacy"))
            if os.path.isdir(os.path.join(package, "src", "legacy", path))
        ]

        for root, _, fnames in os.walk(os.path.join(package, "_legacy")):
            for fname in fnames:
                if fname.endswith(".py"):
                    legacy_fpath = os.path.join(root, fname)
                    tf_keras_root = root.replace("/_legacy", "/_tf_keras")
                    core_api_fpath = os.path.join(
                        root.replace("/_legacy", ""), fname
                    )
                    if not os.path.exists(tf_keras_root):
                        os.makedirs(tf_keras_root)
                    tf_keras_fpath = os.path.join(tf_keras_root, fname)
                    with open(legacy_fpath) as f:
                        legacy_contents = f.read()
                        legacy_contents = legacy_contents.replace(
                            "keras_core._legacy", "keras_core._tf_keras"
                        )
                    if os.path.exists(core_api_fpath):
                        with open(core_api_fpath) as f:
                            core_api_contents = f.read()
                        core_api_contents = core_api_contents.replace(
                            "from keras_core import _tf_keras\n", ""
                        )
                        for legacy_submodule in legacy_submodules:
                            core_api_contents = core_api_contents.replace(
                                f"from keras_core import {legacy_submodule}\n",
                                "",
                            )
                            core_api_contents = core_api_contents.replace(
                                f"keras_core.{legacy_submodule}",
                                f"keras_core._tf_keras.{legacy_submodule}",
                            )
                        legacy_contents = (
                            core_api_contents + "\n" + legacy_contents
                        )
                    with open(tf_keras_fpath, "w") as f:
                        f.write(legacy_contents)

        # Delete keras_core/_legacy/
        shutil.rmtree(os.path.join(package, "_legacy"))

        # Make sure to export the __version__ string
        from keras_core.src.version import __version__  # noqa: E402

        with open(os.path.join(package, "__init__.py")) as f:
            init_contents = f.read()
        with open(os.path.join(package, "__init__.py"), "w") as f:
            f.write(init_contents + "\n\n" + f'__version__ = "{__version__}"\n')

        # Build the package
        os.system("python3 -m build")

        # Save the dist files generated by the build process
        os.chdir(root_path)
        if not os.path.exists(dist_directory):
            os.mkdir(dist_directory)
        for fpath in glob.glob(
            os.path.join(build_directory, dist_directory, "*.*")
        ):
            shutil.copy(fpath, dist_directory)

        # Find the .whl file path
        for fname in os.listdir(dist_directory):
            if __version__ in fname and fname.endswith(".whl"):
                whl_path = os.path.abspath(os.path.join(dist_directory, fname))
        print(f"Build successful. Wheel file available at {whl_path}")
    finally:
        # Clean up: remove the build directory (no longer needed)
        shutil.rmtree(build_directory)
    return whl_path


def install_whl(whl_fpath):
    print(f"Installing wheel file: {whl_fpath}")
    os.system(f"pip3 install {whl_fpath} --force-reinstall --no-dependencies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--install",
        action="store_true",
        help="Whether to install the generated wheel file.",
    )
    args = parser.parse_args()
    whl_path = build()
    if whl_path and args.install:
        install_whl(whl_path)
