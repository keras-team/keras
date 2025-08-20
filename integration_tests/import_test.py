import os
import re
import subprocess

from keras.src import backend
from keras.src.backend import config

# For torch, use index url to avoid installing nvidia drivers for the test.
BACKEND_REQ = {
    "tensorflow": ("tensorflow-cpu", ""),
    "torch": (
        "torch",
        "--extra-index-url https://download.pytorch.org/whl/cpu ",
    ),
    "jax": ("jax[cpu]", ""),
    "openvino": ("openvino", ""),
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
    )
    if not whl_path:
        print(build_process.stdout)
        print(build_process.stderr)
        raise ValueError("Installing Keras package unsuccessful. ")
    return whl_path[-1]


def create_virtualenv():
    env_setup = [
        # Create virtual environment
        "python3 -m venv test_env",
    ]
    os.environ["PATH"] = os.pathsep.join(
        (
            os.path.join(os.getcwd(), "test_env", "bin"),
            os.environ.get("PATH", ""),
        )
    )
    if os.name == "nt":
        os.environ["PATH"] = os.pathsep.join(
            (
                os.path.join(os.getcwd(), "test_env", "Scripts"),
                os.environ["PATH"],
            )
        )
    run_commands_local(env_setup)


def manage_venv_installs(whl_path):
    other_backends = list(set(BACKEND_REQ.keys()) - {backend.backend()})
    backend_pkg, backend_extra_url = BACKEND_REQ[backend.backend()]
    install_setup = [
        # Installs the backend's package and common requirements
        f"pip install {backend_extra_url}{backend_pkg}",
        "pip install -r requirements-common.txt",
        "pip install pytest",
        # Ensure other backends are uninstalled
        "pip uninstall -y {0} {1} {2}".format(
            BACKEND_REQ[other_backends[0]][0],
            BACKEND_REQ[other_backends[1]][0],
            BACKEND_REQ[other_backends[2]][0],
        ),
        # Install `.whl` package
        f"pip install {whl_path}",
    ]
    # Install flax for JAX when NNX is enabled
    if backend.backend() == "jax" and config.is_nnx_enabled():
        install_setup.append("pip install flax>=0.10.1")
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
        cmd_with_args[0] = os.path.join(
            "test_env",
            "Scripts" if os.name == "nt" else "bin",
            cmd_with_args[0],
        )
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
