"""Script to generate keras public API in `keras/api` directory.

Usage:

Run via `./shell/api_gen.sh`.
It generates API and formats user and generated APIs.
"""

import os
import re
import shutil

import namex

package = "keras"


def ignore_files(_, filenames):
    return [f for f in filenames if f.endswith("_test.py")]


def copy_source_to_build_directory(root_path):
    # Copy sources (`keras/` directory and setup files) to build dir
    build_dir = os.path.join(root_path, "tmp_build_dir")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.mkdir(build_dir)
    shutil.copytree(
        package, os.path.join(build_dir, package), ignore=ignore_files
    )
    return build_dir


def create_legacy_directory(package_dir):
    src_dir = os.path.join(package_dir, "src")
    api_dir = os.path.join(package_dir, "api")
    # Make keras/_tf_keras/ by copying keras/
    tf_keras_dirpath_parent = os.path.join(api_dir, "_tf_keras")
    tf_keras_dirpath = os.path.join(tf_keras_dirpath_parent, "keras")
    os.makedirs(tf_keras_dirpath, exist_ok=True)
    with open(os.path.join(tf_keras_dirpath_parent, "__init__.py"), "w") as f:
        f.write("from keras.api._tf_keras import keras\n")
    with open(os.path.join(api_dir, "__init__.py")) as f:
        init_file = f.read()
        init_file = init_file.replace(
            "from keras.api import _legacy",
            "from keras.api import _tf_keras",
        )
    with open(os.path.join(api_dir, "__init__.py"), "w") as f:
        f.write(init_file)
    # Remove the import of `_tf_keras` in `keras/_tf_keras/keras/__init__.py`
    init_file = init_file.replace("from keras.api import _tf_keras\n", "\n")
    with open(os.path.join(tf_keras_dirpath, "__init__.py"), "w") as f:
        f.write(init_file)
    for dirname in os.listdir(api_dir):
        dirpath = os.path.join(api_dir, dirname)
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

    # Copy keras/_legacy/ file contents to keras/_tf_keras/keras
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
    for root, _, fnames in os.walk(os.path.join(api_dir, "_legacy")):
        for fname in fnames:
            if fname.endswith(".py"):
                legacy_fpath = os.path.join(root, fname)
                tf_keras_root = root.replace("/_legacy", "/_tf_keras/keras")
                core_api_fpath = os.path.join(
                    root.replace("/_legacy", ""), fname
                )
                if not os.path.exists(tf_keras_root):
                    os.makedirs(tf_keras_root)
                tf_keras_fpath = os.path.join(tf_keras_root, fname)
                with open(legacy_fpath) as f:
                    legacy_contents = f.read()
                    legacy_contents = legacy_contents.replace(
                        "keras.api._legacy", "keras.api._tf_keras.keras"
                    )
                if os.path.exists(core_api_fpath):
                    with open(core_api_fpath) as f:
                        core_api_contents = f.read()
                    core_api_contents = core_api_contents.replace(
                        "from keras.api import _tf_keras\n", ""
                    )
                    for legacy_submodule in legacy_submodules:
                        core_api_contents = core_api_contents.replace(
                            f"from keras.api import {legacy_submodule}\n",
                            "",
                        )
                        core_api_contents = core_api_contents.replace(
                            f"keras.api.{legacy_submodule}",
                            f"keras.api._tf_keras.keras.{legacy_submodule}",
                        )
                    # Remove duplicate generated comments string.
                    legacy_contents = re.sub(r"\n", r"\\n", legacy_contents)
                    legacy_contents = re.sub('""".*"""', "", legacy_contents)
                    legacy_contents = re.sub(r"\\n", r"\n", legacy_contents)
                    # If the same module is in legacy and core_api, use legacy
                    legacy_imports = re.findall(
                        r"import (\w+)", legacy_contents
                    )
                    for import_name in legacy_imports:
                        core_api_contents = re.sub(
                            f"\n.* import {import_name}\n",
                            r"\n",
                            core_api_contents,
                        )
                    legacy_contents = core_api_contents + "\n" + legacy_contents
                with open(tf_keras_fpath, "w") as f:
                    f.write(legacy_contents)

    # Delete keras/api/_legacy/
    shutil.rmtree(os.path.join(api_dir, "_legacy"))


def export_version_string(api_init_fname):
    with open(api_init_fname) as f:
        contents = f.read()
    with open(api_init_fname, "w") as f:
        contents += "from keras.src.version import __version__\n"
        f.write(contents)


def update_package_init(init_fname):
    contents = """
# Import everything from /api/ into keras.
from keras.api import *  # noqa: F403
from keras.api import __version__  # Import * ignores names start with "_".

import os

# Add everything in /api/ to the module search path.
__path__.append(os.path.join(os.path.dirname(__file__), "api"))  # noqa: F405

# Don't pollute namespace.
del os

# Never autocomplete `.src` or `.api` on an imported keras object.
def __dir__():
    keys = dict.fromkeys((globals().keys()))
    keys.pop("src")
    keys.pop("api")
    return list(keys)


# Don't import `.src` or `.api` during `from keras import *`.
__all__ = [
    name
    for name in globals().keys()
    if not (name.startswith("_") or name in ("src", "api"))
]"""
    with open(init_fname) as f:
        init_contents = f.read()
    with open(init_fname, "w") as f:
        f.write(init_contents.replace("\nfrom keras import api", contents))


def build():
    # Backup the `keras/__init__.py` and restore it on error in api gen.
    root_path = os.path.dirname(os.path.abspath(__file__))
    code_api_dir = os.path.join(root_path, package, "api")
    code_init_fname = os.path.join(root_path, package, "__init__.py")
    # Create temp build dir
    build_dir = copy_source_to_build_directory(root_path)
    build_api_dir = os.path.join(build_dir, package, "api")
    build_init_fname = os.path.join(build_dir, package, "__init__.py")
    build_api_init_fname = os.path.join(build_api_dir, "__init__.py")
    try:
        os.chdir(build_dir)
        # Generates `keras/api` directory.
        if os.path.exists(build_api_dir):
            shutil.rmtree(build_api_dir)
        if os.path.exists(build_init_fname):
            os.remove(build_init_fname)
        os.makedirs(build_api_dir)
        namex.generate_api_files(
            "keras", code_directory="src", target_directory="api"
        )
        # Creates `keras/__init__.py` importing from `keras/api`
        update_package_init(build_init_fname)
        # Add __version__ to keras package
        export_version_string(build_api_init_fname)
        # Creates `_tf_keras` with full keras API
        create_legacy_directory(package_dir=os.path.join(build_dir, package))
        # Copy back the keras/api and keras/__init__.py from build directory
        if os.path.exists(code_api_dir):
            shutil.rmtree(code_api_dir)
        shutil.copytree(build_api_dir, code_api_dir)
        shutil.copy(build_init_fname, code_init_fname)
    finally:
        # Clean up: remove the build directory (no longer needed)
        shutil.rmtree(build_dir)


if __name__ == "__main__":
    build()
