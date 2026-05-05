import ipaddress
import json
import socket
import urllib.parse
import warnings

import numpy as np

from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback

try:
    import requests
except ImportError:
    requests = None


def _validate_url_for_ssrf(root, path):
    """Validate the final request URL for SSRF risks.

    Protects against:
    - Non-http(s) schemes.
    - Link-local IP targets (e.g. the AWS/GCP/Azure metadata endpoint
      169.254.169.254) reached either directly or via DNS rebinding.
    - Host-injection attacks where a crafted ``path`` value (e.g.
      ``@attacker.com``) redirects the request to a different host.
    """
    # --- scheme check on root ---
    root_parsed = urllib.parse.urlparse(root)
    if root_parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid URL scheme '{root_parsed.scheme}'. "
            "Only 'http' and 'https' are allowed."
        )

    # --- build and re-parse the full URL to catch host-injection via path ---
    full_url = root + path
    full_parsed = urllib.parse.urlparse(full_url)

    # Detect host injection: path must not alter the authority component.
    if full_parsed.hostname != root_parsed.hostname:
        raise ValueError(
            f"Host injection detected: the combined URL hostname "
            f"'{full_parsed.hostname}' does not match the root hostname "
            f"'{root_parsed.hostname}'. Ensure 'path' does not contain "
            f"'@', scheme separators, or other authority-altering characters."
        )

    hostname = full_parsed.hostname
    if not hostname:
        raise ValueError("URL must contain a valid hostname.")

    # --- DNS resolution check (catches rebinding attacks) ---
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(
            f"Unable to resolve hostname '{hostname}': {exc}"
        ) from exc

    resolved_ips = {info[4][0] for info in addr_infos}
    for ip_str in resolved_ips:
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        # Block only link-local addresses; these are used by cloud
        # instance-metadata services (AWS 169.254.169.254, GCP, Azure).
        # Loopback (127.x.x.x) and private RFC-1918 ranges are intentionally
        # allowed so that the default localhost target and intranet monitoring
        # configurations continue to work.
        if ip.is_link_local:
            raise ValueError(
                f"Requests to link-local IP addresses are not allowed "
                f"(SSRF protection): '{ip_str}' resolved from '{hostname}'. "
                f"This range is used by cloud instance-metadata services."
            )


@keras_export("keras.callbacks.RemoteMonitor")
class RemoteMonitor(Callback):
    """Callback used to stream events to a server.

    Requires the `requests` library.
    Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
    HTTP POST, with a `data` argument which is a
    JSON-encoded dictionary of event data.
    If `send_as_json=True`, the content type of the request will be
    `"application/json"`.
    Otherwise the serialized JSON will be sent within a form.

    Args:
        root: String; root url of the target server.
        path: String; path relative to `root` to which the events will be sent.
        field: String; JSON field under which the data will be stored.
            The field is used only if the payload is sent within a form
            (i.e. when `send_as_json=False`).
        headers: Dictionary; optional custom HTTP headers.
        send_as_json: Boolean; whether the request should be
            sent as `"application/json"`.
    """

    def __init__(
        self,
        root="http://localhost:9000",
        path="/publish/epoch/end/",
        field="data",
        headers=None,
        send_as_json=False,
    ):
        super().__init__()

        _validate_url_for_ssrf(root, path)

        self.root = root
        self.path = path
        self.field = field
        self.headers = headers
        self.send_as_json = send_as_json

    def on_epoch_end(self, epoch, logs=None):
        if requests is None:
            raise ImportError("RemoteMonitor requires the `requests` library.")
        logs = logs or {}
        send = {}
        send["epoch"] = epoch
        for k, v in logs.items():
            # np.ndarray and np.generic are not scalar types
            # therefore we must unwrap their scalar values and
            # pass to the json-serializable dict 'send'
            if isinstance(v, (np.ndarray, np.generic)):
                send[k] = v.item()
            else:
                send[k] = v
        try:
            if self.send_as_json:
                requests.post(
                    self.root + self.path,
                    json=send,
                    headers=self.headers,
                    timeout=10,
                )
            else:
                requests.post(
                    self.root + self.path,
                    {self.field: json.dumps(send)},
                    headers=self.headers,
                    timeout=10,
                )
        except requests.exceptions.RequestException:
            warnings.warn(
                f"Could not reach RemoteMonitor root server at {self.root}",
                stacklevel=2,
            )
