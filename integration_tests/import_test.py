import os
import re
import subprocess

from keras import backend

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
    whl_path = re.findall(
        r"[^\s]*\.whl",
        build_process.stdout,
    )[-1]
    if not whl_path:
        print(build_process.stderr)
        raise ValueError("Installing Keras package unsuccessful. ")
    return whl_path


def create_virtualenv():
    env_setup = [
        # Create virtual environment
        "python3 -m venv test_env",
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
        "pip install " + whl_path,
    ]
    run_commands_venv(install_setup)


def run_keras_flow():
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
        print(f"Running command: {command}")
        subprocess.run(command, shell=True)


def run_commands_venv(commands):
    for command in commands:
        print(f"Running command: {command}")
        cmd_with_args = command.split(" ")
        cmd_with_args[0] = "test_env/bin/" + cmd_with_args[0]
        p = subprocess.Popen(cmd_with_args)
        assert p.wait() == 0


def test_keras_imports():
    try:
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
        run_keras_flow()

        # Removes virtual environment and associated files
    finally:
        cleanup()


if __name__ == "__main__":
    test_keras_imports()
