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


def _validate_url_structure(root, path):
    """Validate the URL structure for SSRF risks at configuration time.

    Checks performed (no network I/O):

    - The ``root`` scheme must be ``http`` or ``https``.
    - Concatenating ``root + path`` must not alter the URL's host
      (host-injection guard: prevents crafted ``path`` values such as
      ``@attacker.com`` from redirecting requests to a different host).

    DNS resolution is intentionally deferred to request time — see
    :func:`_check_resolved_address` — so that DNS-rebinding attacks are
    also caught.
    """
    root_parsed = urllib.parse.urlparse(root)
    if root_parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid URL scheme '{root_parsed.scheme}'. "
            "Only 'http' and 'https' are allowed."
        )

    # Re-parse the fully concatenated URL to detect host injection via path.
    full_url = root + path
    full_parsed = urllib.parse.urlparse(full_url)

    if full_parsed.hostname != root_parsed.hostname:
        raise ValueError(
            f"Host injection detected: the combined URL hostname "
            f"'{full_parsed.hostname}' does not match the root hostname "
            f"'{root_parsed.hostname}'. Ensure 'path' does not contain "
            f"'@', scheme separators, or other authority-altering characters."
        )

    if not full_parsed.hostname:
        raise ValueError("URL must contain a valid hostname.")


def _check_resolved_address(hostname):
    """Resolve *hostname* and block link-local targets (SSRF protection).

    This function is called at **request time** (inside ``on_epoch_end``)
    so that DNS-rebinding attacks are caught: an attacker who controls DNS
    could serve a benign address during the initial ``__init__`` check and
    later switch to a link-local address for subsequent requests.

    Only link-local addresses (``169.254.0.0/16`` for IPv4,
    ``fe80::/10`` for IPv6) are blocked — these are the ranges used by
    cloud instance-metadata services (e.g. AWS IMDSv1 at
    ``169.254.169.254``, GCP, and Azure).

    Loopback addresses (``127.x.x.x`` / ``::1``) and private RFC-1918
    ranges (``10.x``, ``172.16.x``, ``192.168.x``) are intentionally
    **allowed** so that the default localhost target
    (``http://localhost:9000``) and intranet monitoring configurations
    continue to work.
    """
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
        if ip.is_link_local:
            raise ValueError(
                f"Requests to link-local IP addresses are not allowed "
                f"(SSRF protection): '{ip_str}' resolved from '{hostname}'. "
                f"This range is used by cloud instance-metadata services "
                f"(e.g. AWS IMDSv1 at 169.254.169.254)."
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

        # Validate URL structure (scheme + host-injection) at construction
        # time. DNS resolution is deferred to on_epoch_end so that
        # DNS-rebinding attacks are also caught at request time.
        _validate_url_structure(root, path)

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

        # Re-validate URL structure and DNS at request time to catch
        # both DNS-rebinding attacks and any post-init host injection
        # via mutated self.root or self.path.
        _validate_url_structure(self.root, self.path)
        hostname = urllib.parse.urlparse(self.root).hostname
        _check_resolved_address(hostname)

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
