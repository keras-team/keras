#!/usr/bin/env -S uv run --script
"""Return only non-reusable runner, Kubernetes, and GCE identity metadata."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import ssl
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

RUNNER_ROOT = Path("/home/runner")
KSA_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
TOKEN_PATH = KSA_DIR / "token"
CA_PATH = KSA_DIR / "ca.crt"
NAMESPACE_PATH = KSA_DIR / "namespace"
MAX_METADATA_ITEMS = 50


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def opaque_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def file_metadata(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
    except OSError as exc:
        return {"present": False, "error": type(exc).__name__}
    return {
        "present": True,
        "mode": stat.filemode(info.st_mode),
        "mode_octal": format(stat.S_IMODE(info.st_mode), "04o"),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "bytes": info.st_size,
    }


def decode_jwt_claims(token: str) -> dict[str, Any] | None:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        value = json.loads(base64.urlsafe_b64decode(payload))
        return value if isinstance(value, dict) else None
    except (ValueError, UnicodeError, json.JSONDecodeError):
        return None


def project_runner_files() -> dict[str, Any]:
    runner_path = RUNNER_ROOT / ".runner"
    credentials_path = RUNNER_ROOT / ".credentials"
    rsa_path = RUNNER_ROOT / ".credentials_rsaparams"
    runner = read_json(runner_path) or {}
    credentials = read_json(credentials_path) or {}
    credential_data = credentials.get("data", {})
    if not isinstance(credential_data, dict):
        credential_data = {}

    def url_host(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        try:
            return urllib.parse.urlsplit(value).netloc or None
        except ValueError:
            return None

    rsa = read_json(rsa_path) or {}
    modulus = rsa.get("Modulus") or rsa.get("modulus")
    modulus_bits = None
    if isinstance(modulus, str):
        try:
            modulus_bits = len(base64.b64decode(modulus)) * 8
        except ValueError:
            pass
    return {
        "runner": {
            "file": file_metadata(runner_path),
            "metadata": {
                key: runner[key]
                for key in (
                    "agentId",
                    "agentName",
                    "poolId",
                    "poolName",
                    "serverUrl",
                    "gitHubUrl",
                    "workFolder",
                    "ephemeral",
                    "disableUpdate",
                )
                if key in runner
            },
        },
        "credentials": {
            "file": file_metadata(credentials_path),
            "scheme": credentials.get("scheme"),
            "data_keys": sorted(str(key) for key in credential_data),
            "client_id": credential_data.get("clientId"),
            "authorization_host": url_host(
                credential_data.get("authorizationUrl")
            ),
            "authorization_v2_host": url_host(
                credential_data.get("authorizationUrlV2")
            ),
            "oauth_endpoint_host": url_host(
                credential_data.get("oauthEndpointUrl")
            ),
        },
        "rsa_parameters": {
            "file": file_metadata(rsa_path),
            "field_names": sorted(str(key) for key in rsa),
            "modulus_bits": modulus_bits,
            "values_returned": False,
        },
    }


def project_kubernetes_claims(claims: dict[str, Any] | None) -> dict[str, Any]:
    if not claims:
        return {"available": False}
    kubernetes = claims.get("kubernetes.io", {})
    if not isinstance(kubernetes, dict):
        kubernetes = {}
    serviceaccount = kubernetes.get("serviceaccount", {})
    pod = kubernetes.get("pod", {})
    node = kubernetes.get("node", {})
    return {
        "available": True,
        "iss_hash": opaque_name(claims.get("iss")),
        "sub_hash": opaque_name(claims.get("sub")),
        "aud": claims.get("aud"),
        "iat": claims.get("iat"),
        "exp": claims.get("exp"),
        "namespace_hash": opaque_name(kubernetes.get("namespace")),
        "serviceaccount": {
            "name_hash": opaque_name(
                serviceaccount.get("name")
                if isinstance(serviceaccount, dict)
                else None
            ),
            "uid_present": bool(
                serviceaccount.get("uid")
                if isinstance(serviceaccount, dict)
                else None
            ),
        },
        "pod": {
            "name_hash": opaque_name(
                pod.get("name") if isinstance(pod, dict) else None
            ),
            "uid_present": bool(
                pod.get("uid") if isinstance(pod, dict) else None
            ),
        },
        "node": {
            "name_hash": opaque_name(
                node.get("name") if isinstance(node, dict) else None
            ),
            "uid_present": bool(
                node.get("uid") if isinstance(node, dict) else None
            ),
        },
        "token_returned": False,
    }


class KubeClient:
    def __init__(self, token: str, namespace: str):
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        if not host:
            raise RuntimeError("KUBERNETES_SERVICE_HOST is absent")
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        self.base = f"https://{host}:{port}"
        self.token = token
        self.namespace = namespace
        self.context = ssl.create_default_context(cafile=str(CA_PATH))

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        accept: str = "application/json",
    ) -> dict[str, Any]:
        payload = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base + path,
            data=payload,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": accept,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request, context=self.context, timeout=8
            ) as response:
                return {"ok": True, "value": json.load(response)}
        except urllib.error.HTTPError as exc:
            return {"ok": False, "http_status": exc.code}
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": type(exc).__name__}

    def ssar(
        self,
        verb: str,
        resource: str,
        group: str = "",
        subresource: str | None = None,
    ) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            "namespace": self.namespace,
            "verb": verb,
            "group": group,
            "resource": resource,
        }
        if subresource:
            attributes["subresource"] = subresource
        response = self.request(
            "POST",
            "/apis/authorization.k8s.io/v1/selfsubjectaccessreviews",
            {
                "apiVersion": "authorization.k8s.io/v1",
                "kind": "SelfSubjectAccessReview",
                "spec": {"resourceAttributes": attributes},
            },
        )
        value = response.get("value")
        if response.get("ok") and isinstance(value, dict):
            status_value = value.get("status", {})
            if isinstance(status_value, dict):
                return {
                    "allowed": bool(status_value.get("allowed")),
                    "denied": bool(status_value.get("denied")),
                    "reason": status_value.get("reason"),
                }
        return {key: response[key] for key in response if key != "value"}

    def partial_metadata(self, path: str, kind: str) -> dict[str, Any]:
        response = self.request(
            "GET",
            path,
            accept="application/json;as=PartialObjectMetadataList;g=meta.k8s.io;v=v1",
        )
        value = response.get("value")
        if not response.get("ok") or not isinstance(value, dict):
            return {key: response[key] for key in response if key != "value"}
        raw_items = value.get("items", [])
        projected = []
        for item in raw_items[:MAX_METADATA_ITEMS]:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            labels = metadata.get("labels", {})
            annotations = metadata.get("annotations", {})
            if not isinstance(labels, dict):
                labels = {}
            if not isinstance(annotations, dict):
                annotations = {}
            safe_labels = {
                key: val
                for key, val in labels.items()
                if key.startswith("actions.github.com/")
                or key.startswith("app.kubernetes.io/")
                or key in {"job-name", "jobset.sigs.k8s.io/jobset-name"}
            }
            safe_annotations = {
                key: val
                for key, val in annotations.items()
                if key == "iam.gke.io/gcp-service-account"
            }
            owners = []
            for owner in metadata.get("ownerReferences", []):
                if isinstance(owner, dict):
                    owners.append(
                        {
                            "kind": owner.get("kind"),
                            "name_hash": opaque_name(owner.get("name")),
                            "controller": owner.get("controller"),
                        }
                    )
            projected.append(
                {
                    "name_hash": opaque_name(metadata.get("name")),
                    "labels": safe_labels,
                    "annotations": safe_annotations,
                    "ownerReferences": owners,
                    "arc_cleanup_finalizer": (
                        "actions.github.com/cleanup-protection"
                        in metadata.get("finalizers", [])
                    ),
                }
            )
        return {
            "ok": True,
            "kind": kind,
            "count": len(raw_items),
            "items": projected,
            "truncated": len(raw_items) > MAX_METADATA_ITEMS,
            "object_bodies_returned": False,
        }

    def autoscaling_runner_sets(self) -> dict[str, Any]:
        namespace = urllib.parse.quote(self.namespace, safe="")
        response = self.request(
            "GET",
            f"/apis/actions.github.com/v1alpha1/namespaces/{namespace}/autoscalingrunnersets",
        )
        value = response.get("value")
        if not response.get("ok") or not isinstance(value, dict):
            return {key: response[key] for key in response if key != "value"}
        projected = []
        for item in value.get("items", []):
            if not isinstance(item, dict):
                continue
            spec = item.get("spec", {})
            if not isinstance(spec, dict):
                spec = {}
            template = spec.get("template", {})
            pod_spec = (
                template.get("spec", {}) if isinstance(template, dict) else {}
            )
            if not isinstance(pod_spec, dict):
                pod_spec = {}
            secret_name = spec.get("githubConfigSecret")
            projected.append(
                {
                    "name_hash": opaque_name(
                        item.get("metadata", {}).get("name")
                    ),
                    "runnerScaleSetName": spec.get("runnerScaleSetName"),
                    "runnerGroup": spec.get("runnerGroup"),
                    "githubConfigUrl": spec.get("githubConfigUrl"),
                    "githubConfigSecret_name_hash": opaque_name(secret_name),
                    "githubConfigSecret_configured": bool(secret_name),
                    "vaultType": (
                        spec.get("vaultConfig", {}).get("type")
                        if isinstance(spec.get("vaultConfig"), dict)
                        else None
                    ),
                    "runner_serviceaccount_name_hash": opaque_name(
                        pod_spec.get("serviceAccountName")
                    ),
                }
            )
        return {"ok": True, "items": projected, "secret_values_returned": False}


def metadata_get(path: str) -> str | list[str] | None:
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/" + path,
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            value = response.read().decode().strip()
    except (urllib.error.URLError, OSError, UnicodeError):
        return None
    if path.endswith("/scopes"):
        return sorted(line for line in value.splitlines() if line)
    return value or None


def main() -> int:
    output: dict[str, Any] = {
        "captured_at": utc_now(),
        "safety": {
            "credential_values_returned": False,
            "credential_hashes_returned": False,
            "cloud_token_requested": False,
            "cloud_identity_document_requested": False,
            "kubernetes_secret_data_requested": False,
            "mutating_request_sent": False,
        },
        "process": {
            "uid": os.getuid(),
            "gid": os.getgid(),
            "environment_variable_names": sorted(
                key
                for key in os.environ
                if any(
                    word in key.upper()
                    for word in (
                        "TOKEN",
                        "SECRET",
                        "CREDENTIAL",
                        "KUBERNETES",
                        "RUNNER",
                    )
                )
            ),
        },
        "runner_files": project_runner_files(),
        "gce_metadata": {
            "project_id": metadata_get("project/project-id"),
            "service_account_email": metadata_get(
                "instance/service-accounts/default/email"
            ),
            "scopes": metadata_get("instance/service-accounts/default/scopes"),
            "access_token_requested": False,
        },
    }
    try:
        token = TOKEN_PATH.read_text().strip()
        namespace = NAMESPACE_PATH.read_text().strip()
    except OSError as exc:
        output["kubernetes"] = {"available": False, "error": type(exc).__name__}
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    output["kubernetes_identity"] = project_kubernetes_claims(
        decode_jwt_claims(token)
    )
    try:
        client = KubeClient(token, namespace)
    except RuntimeError as exc:
        output["kubernetes"] = {"available": False, "error": str(exc)}
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    checks = [
        ("pods", "list", "", None),
        ("pods", "create", "", None),
        ("pods", "create", "", "exec"),
        ("pods", "get", "", "log"),
        ("jobs", "list", "batch", None),
        ("secrets", "get", "", None),
        ("secrets", "list", "", None),
        ("serviceaccounts", "list", "", None),
        ("serviceaccounts", "create", "", "token"),
        ("roles", "list", "rbac.authorization.k8s.io", None),
        ("rolebindings", "list", "rbac.authorization.k8s.io", None),
        ("nodes", "list", "", None),
        ("autoscalingrunnersets", "list", "actions.github.com", None),
        ("ephemeralrunners", "list", "actions.github.com", None),
    ]
    authorization: dict[str, Any] = {}
    for resource, verb, group, subresource in checks:
        key = "/".join(
            part
            for part in (group or "core", resource, subresource, verb)
            if part
        )
        authorization[key] = client.ssar(verb, resource, group, subresource)

    namespace = urllib.parse.quote(namespace, safe="")
    kubernetes: dict[str, Any] = {
        "available": True,
        "namespace_hash": opaque_name(client.namespace),
        "authorization": authorization,
        "pod_metadata": client.partial_metadata(
            f"/api/v1/namespaces/{namespace}/pods", "Pod"
        ),
        "job_metadata": client.partial_metadata(
            f"/apis/batch/v1/namespaces/{namespace}/jobs", "Job"
        ),
        "secret_metadata": client.partial_metadata(
            f"/api/v1/namespaces/{namespace}/secrets", "Secret"
        ),
        "serviceaccount_metadata": client.partial_metadata(
            f"/api/v1/namespaces/{namespace}/serviceaccounts", "ServiceAccount"
        ),
    }
    if authorization.get(
        "actions.github.com/autoscalingrunnersets/list", {}
    ).get("allowed"):
        kubernetes["autoscaling_runner_sets"] = client.autoscaling_runner_sets()
    output["kubernetes"] = kubernetes
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
