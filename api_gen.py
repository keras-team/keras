"""Script to generate keras public API in `keras/api` directory.

Usage:

Run via `./shell/api_gen.sh`.
It generates API and formats user and generated APIs.
"""

import os
import shutil

import namex

package = "keras"


def ignore_files(_, filenames):
    return [f for f in filenames if f.endswith("_test.py")]


def create_legacy_directory():
    API_DIR = os.path.join(package, "api")
    # Make keras/_tf_keras/ by copying keras/
    tf_keras_dirpath_parent = os.path.join(API_DIR, "_tf_keras")
    tf_keras_dirpath = os.path.join(tf_keras_dirpath_parent, "keras")
    os.makedirs(tf_keras_dirpath, exist_ok=True)
    with open(os.path.join(tf_keras_dirpath_parent, "__init__.py"), "w") as f:
        f.write("from keras.api._tf_keras import keras\n")
    with open(os.path.join(API_DIR, "__init__.py")) as f:
        init_file = f.read()
        init_file = init_file.replace(
            "from keras.api import _legacy",
            "from keras.api import _tf_keras",
        )
    with open(os.path.join(API_DIR, "__init__.py"), "w") as f:
        f.write(init_file)
    # Remove the import of `_tf_keras` in `keras/_tf_keras/keras/__init__.py`
    init_file = init_file.replace("from keras.api import _tf_keras\n", "\n")
    with open(os.path.join(tf_keras_dirpath, "__init__.py"), "w") as f:
        f.write(init_file)
    for dirname in os.listdir(API_DIR):
        dirpath = os.path.join(API_DIR, dirname)
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
                    legacy_contents = core_api_contents + "\n" + legacy_contents
                with open(tf_keras_fpath, "w") as f:
                    f.write(legacy_contents)

    # Delete keras/api/_legacy/
    shutil.rmtree(os.path.join(API_DIR, "_legacy"))


def export_version_string():
    API_INIT = os.path.join(package, "api", "__init__.py")
    with open(API_INIT) as f:
        contents = f.read()
    with open(API_INIT, "w") as f:
        contents += "from keras.src.version import __version__\n"
        f.write(contents)


def update_package_init():
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
    with open(os.path.join(package, "__init__.py")) as f:
        init_contents = f.read()
    with open(os.path.join(package, "__init__.py"), "w") as f:
        f.write(init_contents.replace("\nfrom keras import api", contents))


if __name__ == "__main__":
    # Backup the `keras/__init__.py` and restore it on error in api gen.
    os.makedirs(os.path.join(package, "api"), exist_ok=True)
    init_fname = os.path.join(package, "__init__.py")
    backup_init_fname = os.path.join(package, "__init__.py.bak")
    try:
        if os.path.exists(init_fname):
            shutil.move(init_fname, backup_init_fname)
        # Generates `keras/api` directory.
        namex.generate_api_files(
            "keras", code_directory="src", target_directory="api"
        )
        # Creates `keras/__init__.py` importing from `keras/api`
        update_package_init()
    except Exception as e:
        if os.path.exists(backup_init_fname):
            shutil.move(backup_init_fname, init_fname)
        raise e
    finally:
        if os.path.exists(backup_init_fname):
            os.remove(backup_init_fname)
    # Add __version__ to keras package
    export_version_string()
    # Creates `_tf_keras` with full keras API
    create_legacy_directory()
