"""Script to generate keras public API in `keras/api` directory.

Usage:

Run via `./shell/api_gen.sh`.
It generates API and formats user and generated APIs.
"""

import os
import re
import shutil

import namex

PACKAGE = "keras"
BUILD_DIR_NAME = "tmp_build_dir"

# Path fragments used repeatedly when rewriting `_legacy` paths/contents
# into their `_tf_keras/keras` equivalents.
_LEGACY_SEGMENT = os.path.join(os.path.sep, "_legacy")
_TF_KERAS_SEGMENT = os.path.join(os.path.sep, "_tf_keras", "keras")


def ignore_files(_, filenames):
    """`shutil.copytree` ignore-hook that skips `*_test.py` files."""
    return [f for f in filenames if f.endswith("_test.py")]


def copy_source_to_build_directory(root_path):
    """Copies `keras/src` into a fresh temp build directory.

    Args:
        root_path: Absolute path to the repository root.

    Returns:
        The path to the newly created build directory.
    """
    build_dir = os.path.join(root_path, BUILD_DIR_NAME)
    build_package_dir = os.path.join(build_dir, PACKAGE)
    build_src_dir = os.path.join(build_package_dir, "src")
    root_src_dir = os.path.join(root_path, PACKAGE, "src")

    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_package_dir)
    shutil.copytree(root_src_dir, build_src_dir)
    return build_dir


def create_legacy_directory(package_dir):
    """Builds `keras/_tf_keras/keras`, a full copy of the public API.

    `_tf_keras` mirrors the generated public API (`keras/`) but with the
    `_legacy` symbols folded back in, for backwards compatibility with
    `tf.keras`. After this runs, `keras/_legacy` is deleted since its
    contents have been merged into `_tf_keras`.

    Args:
        package_dir: Path to the generated `keras` package directory
            (inside the build directory), containing `src/`, `_legacy/`,
            and the generated API modules.
    """
    src_dir = os.path.join(package_dir, "src")

    # Make keras/_tf_keras/ by copying keras/
    tf_keras_dirpath_parent = os.path.join(package_dir, "_tf_keras")
    tf_keras_dirpath = os.path.join(tf_keras_dirpath_parent, "keras")
    os.makedirs(tf_keras_dirpath, exist_ok=True)

    with open(os.path.join(tf_keras_dirpath_parent, "__init__.py"), "w") as f:
        f.write("from keras._tf_keras import keras\n")

    # Point `keras/__init__.py` at `_tf_keras` instead of `_legacy`.
    with open(os.path.join(package_dir, "__init__.py")) as f:
        init_file = f.read()
        init_file = init_file.replace(
            "from keras import _legacy as _legacy",
            "from keras import _tf_keras as _tf_keras",
        )
    with open(os.path.join(package_dir, "__init__.py"), "w") as f:
        f.write(init_file)

    # `keras/_tf_keras/keras/__init__.py` shouldn't re-import itself, so
    # strip out the `_tf_keras` import before reusing `init_file` there.
    init_file = init_file.replace("from keras import _tf_keras\n", "\n")
    with open(os.path.join(tf_keras_dirpath, "__init__.py"), "w") as f:
        f.write(init_file)

    # Copy every generated API subpackage (except `_legacy`/`_tf_keras`/
    # `src`) into `_tf_keras/keras`, so it starts as a full mirror of the
    # public API before the legacy contents are merged in below.
    for dirname in os.listdir(package_dir):
        dirpath = os.path.join(package_dir, dirname)
        if os.path.isdir(dirpath) and dirname not in (
            "_legacy",
            "_tf_keras",
            "src",
        ):
            destpath = os.path.join(tf_keras_dirpath, dirname)
            if os.path.exists(destpath):
                shutil.rmtree(destpath)
            shutil.copytree(
                dirpath,
                destpath,
                ignore=ignore_files,
            )

    # Names of the top-level submodules under `src/legacy`, used below to
    # find and remove the corresponding non-legacy imports/references in
    # the generated core API files.
    legacy_submodules = [
        path[:-3]
        for path in os.listdir(os.path.join(src_dir, "legacy"))
        if path.endswith(".py")
    ]
    legacy_submodules += [
        path
        for path in os.listdir(os.path.join(src_dir, "legacy"))
        if os.path.isdir(os.path.join(src_dir, "legacy", path))
    ]

    # Merge each `_legacy` file into its `_tf_keras/keras` counterpart,
    # preferring legacy definitions over the core API's when both define
    # the same symbol.
    for root, _, fnames in os.walk(os.path.join(package_dir, "_legacy")):
        for fname in fnames:
            if not fname.endswith(".py"):
                continue

            legacy_fpath = os.path.join(root, fname)
            tf_keras_root = root.replace(_LEGACY_SEGMENT, _TF_KERAS_SEGMENT)
            core_api_fpath = os.path.join(
                root.replace(_LEGACY_SEGMENT, ""), fname
            )

            if not os.path.exists(tf_keras_root):
                os.makedirs(tf_keras_root)
            tf_keras_fpath = os.path.join(tf_keras_root, fname)

            with open(legacy_fpath) as f:
                legacy_contents = f.read()
                legacy_contents = legacy_contents.replace(
                    "keras._legacy", "keras._tf_keras.keras"
                )

            if os.path.exists(core_api_fpath):
                with open(core_api_fpath) as f:
                    core_api_contents = f.read()
                core_api_contents = core_api_contents.replace(
                    "from keras import _tf_keras as _tf_keras\n", ""
                )
                for legacy_submodule in legacy_submodules:
                    core_api_contents = core_api_contents.replace(
                        f"from keras import {legacy_submodule} as "
                        f"{legacy_submodule}\n",
                        "",
                    )
                    core_api_contents = core_api_contents.replace(
                        f"keras.{legacy_submodule}",
                        f"keras._tf_keras.keras.{legacy_submodule}",
                    )
                # Remove duplicate generated comments string. (Docstrings
                # are stripped with a non-greedy-unsafe pattern, so
                # newlines are escaped first to keep the match on one
                # logical line and avoid `re.DOTALL` surprises.)
                # Remove duplicate generated comments/docstrings safely using a non-greedy pattern.
                legacy_contents = re.sub(r'""".*?"""', "", legacy_contents, flags=re.DOTALL)

                # If the same module is imported in both legacy and
                # core_api, keep only the legacy version.
                legacy_imports = re.findall(
                    r"import (\w+)", legacy_contents
                )
                for import_name in legacy_imports:
                    core_api_contents = re.sub(
                        f"\n.* import {import_name} as {import_name}\n",
                        r"\n",
                        core_api_contents,
                    )
                legacy_contents = f"{core_api_contents}\n{legacy_contents}"

            with open(tf_keras_fpath, "w") as f:
                f.write(legacy_contents)

    # Delete keras/api/_legacy/ now that it has been merged into
    # `_tf_keras/keras`.
    shutil.rmtree(os.path.join(package_dir, "_legacy"))


def export_version_string(api_init_fname):
    """Appends the `__version__` re-export to a generated `__init__.py`."""
    with open(api_init_fname) as f:
        contents = f.read()
    with open(api_init_fname, "w") as f:
        contents += (
            "from keras.src.version import __version__ as __version__\n"
        )
        f.write(contents)


def build():
    """Generates the public `keras.api` package and `_tf_keras` mirror.

    Sources are first copied into a scratch build directory so that
    `namex` can generate the API there without touching the checked-out
    `keras/api` tree until generation succeeds. The build directory is
    always cleaned up afterwards.
    """
    root_path = os.path.dirname(os.path.abspath(__file__))
    code_api_dir = os.path.join(root_path, PACKAGE, "api")

    # Create temp build dir
    build_dir = copy_source_to_build_directory(root_path)
    build_api_dir = os.path.join(build_dir, PACKAGE)
    build_src_dir = os.path.join(build_api_dir, "src")
    build_api_init_fname = os.path.join(build_api_dir, "__init__.py")

    try:
        os.chdir(build_dir)
        open(build_api_init_fname, "w").close()
        namex.generate_api_files(
            "keras",
            code_directory="src",
            exclude_directories=[
                os.path.join("src", "backend", "jax"),
                os.path.join("src", "backend", "openvino"),
                os.path.join("src", "backend", "tensorflow"),
                os.path.join("src", "backend", "torch"),
            ],
        )
        # Add __version__ to `api/`.
        export_version_string(build_api_init_fname)
        # Creates `_tf_keras` with full keras API
        create_legacy_directory(package_dir=os.path.join(build_dir, PACKAGE))
        # Copy back the keras/api and keras/__init__.py from build directory
        if os.path.exists(build_src_dir):
            shutil.rmtree(build_src_dir)
        if os.path.exists(code_api_dir):
            shutil.rmtree(code_api_dir)
        shutil.copytree(
            build_api_dir, code_api_dir, ignore=shutil.ignore_patterns("src/")
        )
    finally:
        # Clean up: remove the build directory (no longer needed)
        shutil.rmtree(build_dir)


if __name__ == "__main__":
    build()
