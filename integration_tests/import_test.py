import os
import re
import subprocess

from keras_core import backend

BACKEND_REQ = {
    "tensorflow": "tensorflow",
    "torch": "torch torchvision",
    "jax": "jax jaxlib",
}


def setup_package():
    subprocess.run("rm -rf tmp_build_dir", shell=True)
    build_process = subprocess.run(
        "python3 pip_build.py",
        capture_output=True,
        text=True,
        shell=True,
    )
    print(build_process.stdout)
    match = re.search(
        r"\s[^\s]*\.whl",
        build_process.stdout,
    )
    if not match:
        raise ValueError("Installing Keras Core package unsuccessful. ")
        print(build_process.stderr)
    whl_path = match.group()
    return whl_path


def create_virtualenv():
    env_setup = [
        # Create and activate virtual environment
        "python3 -m venv test_env",
        # "source ./test_env/bin/activate",
    ]
    os.environ["PATH"] = (
        "/test_env/bin/" + os.pathsep + os.environ.get("PATH", "")
    )
    run_commands_local(env_setup)


def manage_venv_installs(whl_path):
    other_backends = list(set(BACKEND_REQ.keys()) - {backend.backend()})
    install_setup = [
        # Installs the backend's package and common requirements
        "pip install " + BACKEND_REQ[backend.backend()],
        "pip install -r requirements-common.txt",
        "pip install pytest",
        # Ensure other backends are uninstalled
        "pip uninstall -y "
        + BACKEND_REQ[other_backends[0]]
        + " "
        + BACKEND_REQ[other_backends[1]],
        # Install `.whl` package
        "pip install " + whl_path + " --force-reinstall --no-dependencies",
    ]
    run_commands_venv(install_setup)


def run_keras_core_flow():
    test_script = [
        # Runs the example script
        "python -m pytest integration_tests/basic_full_flow.py",
    ]
    run_commands_venv(test_script)


def cleanup():
    cleanup_script = [
        # Exits virtual environment, deletes files, and any
        # miscellaneous install logs
        "exit",
        "rm -rf test_env",
        "rm -rf tmp_build_dir",
        "rm -f *+cpu",
    ]
    run_commands_local(cleanup_script)


def run_commands_local(commands):
    for command in commands:
        subprocess.run(command, shell=True)


def run_commands_venv(commands):
    for command in commands:
        cmd_with_args = command.split(" ")
        cmd_with_args[0] = "test_env/bin/" + cmd_with_args[0]
        p = subprocess.Popen(cmd_with_args)
        p.wait()


def test_keras_core_imports():
    # Ensures packages from all backends are installed.
    # Builds Keras core package and returns package file path.
    whl_path = setup_package()

    # Creates and activates a virtual environment.
    create_virtualenv()

    # Ensures the backend's package is installed
    # and the other backends are uninstalled.
    manage_venv_installs(whl_path)

    # Runs test of basic flow in Keras Core.
    # Tests for backend-specific imports and `model.fit()`.
    run_keras_core_flow()

    # Removes virtual environment and associated files
    cleanup()


if __name__ == "__main__":
    test_keras_core_imports()