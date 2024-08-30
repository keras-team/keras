"""Script to create (and optionally install) a `.whl` archive for Keras 3.

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
import datetime
import glob
import os
import pathlib
import re
import shutil
import sys

# Needed because importing torch after TF causes the runtime to crash
import torch  # noqa: F401

package = "keras"
build_directory = "tmp_build_dir"
dist_directory = "dist"
to_copy = ["setup.py", "README.md"]

def export_version_string(version, is_nightly=False, rc_index=None):
    """Export Version and Package Name."""
    if is_nightly:
        date = datetime.datetime.now()
        version += f".dev{date.strftime('%Y%m%d%H')}"
        # Replaces `name="keras"` string in `setup.py` with `keras-nightly`
        setup_path = "setup.py"
        if os.path.isfile(setup_path):
            with open(setup_path) as f:
                setup_contents = f.read()
            setup_contents = setup_contents.replace(
                'name="keras"', 'name="keras-nightly"'
            )
            with open(setup_path, "w") as f:
                f.write(setup_contents)
        else:
            print(f"Warning: {setup_path} not found.")
    elif rc_index is not None:
        version += f"rc{rc_index}"

    # Export the __version__ string
    version_path = os.path.join(package, "src", "version.py")
    if os.path.isfile(version_path):
        with open(version_path) as f:
            init_contents = f.read()
        init_contents = re.sub(
            r"\n__version__ = .*\n",
            f'\n__version__ = "{version}"\n',
            init_contents,
        )
        with open(version_path, "w") as f:
            f.write(init_contents)
    else:
        print(f"Warning: {version_path} not found.")

def ignore_files(_, filenames):
    return [f for f in filenames if f.endswith("_test.py")]

def copy_source_to_build_directory(root_path):
    """Copy sources and setup files to the build directory."""
    os.makedirs(build_directory, exist_ok=True)
    shutil.copytree(
        package, os.path.join(build_directory, package), ignore=ignore_files
    )
    for fname in to_copy:
        src = os.path.join(root_path, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(build_directory, fname))
        else:
            print(f"Warning: {fname} not found in the root path.")

def move_tf_keras_directory():
    """Move `keras/api/_tf_keras` to `keras/_tf_keras` and update references."""
    tf_keras_src = os.path.join(package, "api", "_tf_keras")
    tf_keras_dest = os.path.join(package, "_tf_keras")
    if os.path.isdir(tf_keras_src):
        shutil.move(tf_keras_src, tf_keras_dest)
        init_path = os.path.join(package, "api", "__init__.py")
        if os.path.isfile(init_path):
            with open(init_path) as f:
                contents = f.read()
            contents = contents.replace("from keras.api import _tf_keras", "")
            with open(init_path, "w") as f:
                f.write(contents)
        else:
            print(f"Warning: {init_path} not found.")

        for root, _, fnames in os.walk(tf_keras_dest):
            for fname in fnames:
                if fname.endswith(".py"):
                    tf_keras_fpath = os.path.join(root, fname)
                    with open(tf_keras_fpath) as f:
                        contents = f.read()
                    contents = contents.replace(
                        "keras.api._tf_keras", "keras._tf_keras"
                    )
                    with open(tf_keras_fpath, "w") as f:
                        f.write(contents)
    else:
        print(f"Warning: Source directory {tf_keras_src} not found.")

def build(root_path, is_nightly=False, rc_index=None):
    """Build the package and save the distribution files."""
    if os.path.exists(build_directory):
        raise ValueError(f"Directory already exists: {build_directory}")

    try:
        copy_source_to_build_directory(root_path)
        move_tf_keras_directory()

        from keras.src.version import __version__  # noqa: E402

        export_version_string(__version__, is_nightly, rc_index)
        return build_and_save_output(root_path, __version__)
    finally:
        # Clean up: remove the build directory (no longer needed)
        shutil.rmtree(build_directory)

def build_and_save_output(root_path, __version__):
    """Build the package and save the distribution files."""
    if os.system("python3 -m build") != 0:
        print("Build failed.")
        return None

    dist_path = os.path.join(root_path, dist_directory)
    os.makedirs(dist_path, exist_ok=True)
    for fpath in glob.glob(os.path.join(build_directory, dist_directory, "*.*")):
        shutil.copy(fpath, dist_path)

    whl_path = None
    for fname in os.listdir(dist_path):
        if __version__ in fname and fname.endswith(".whl"):
            whl_path = os.path.abspath(os.path.join(dist_path, fname))
            break

    if whl_path:
        print(f"Build successful. Wheel file available at {whl_path}")
    else:
        print("Build failed. No wheel file found.")
    return whl_path

def install_whl(whl_fpath):
    """Install the wheel file."""
    print(f"Installing wheel file: {whl_fpath}")
    if os.system(f"pip3 install {whl_fpath} --force-reinstall --no-dependencies") != 0:
        print("Installation failed.")
    else:
        print("Installation successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--install",
        action="store_true",
        help="Whether to install the generated wheel file.",
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Whether to generate nightly wheel file.",
    )
    parser.add_argument(
        "--rc",
        type=int,
        help="Specify [0-9] when generating RC wheels.",
    )
    args = parser.parse_args()
    root_path = pathlib.Path(__file__).parent.resolve()
    whl_path = build(root_path, args.nightly, args.rc)
    if whl_path and args.install:
        install_whl(whl_path)
