"use strict";

const assert = require("assert");
const { collect, podSpec } = require("./vrp_k8s_dryrun_probe.js");

function response(value, status = 200) {
  return { ok: status >= 200 && status < 300, status, value };
}

async function testAdmissionProjection() {
  const calls = [];
  const readFile = (filePath, encoding) => {
    if (filePath.endsWith("/namespace")) return "test-namespace\n";
    if (filePath.endsWith("/token")) return "SENSITIVE-KSA-TOKEN\n";
    if (filePath.endsWith("/ca.crt")) return Buffer.from("TEST-CA");
    throw new Error(`unexpected file: ${filePath} ${encoding || ""}`);
  };
  const call = async (options) => {
    calls.push(options);
    if (options.method === "GET") {
      return response({ reason: "NotFound" }, 404);
    }
    const pod = JSON.parse(options.body.toString("utf8"));
    if (pod.spec.hostPID) {
      return response(
        {
          reason: "Forbidden",
          message: "combined boundary denied",
          details: {
            causes: [{ reason: "FieldValueForbidden", field: "spec.hostPID" }],
          },
        },
        403,
      );
    }
    return response(pod, 201);
  };
  const originalHost = process.env.KUBERNETES_SERVICE_HOST;
  process.env.KUBERNETES_SERVICE_HOST = "kubernetes.test";
  try {
    const result = await collect(call, readFile, 123456789);
    assert.equal(result.phase, "keras-kubernetes-admission-dry-run");
    assert.equal(result.safety.server_side_dry_run_only, true);
    assert.equal(result.safety.persistent_create_sent, false);
    assert.equal(result.variants.length, 3);
    assert.equal(result.variants[0].create.accepted, true);
    assert.equal(result.variants[1].create.accepted, true);
    assert.equal(result.variants[1].create.returned_spec_projection.host_network, true);
    assert.equal(result.variants[2].create.accepted, false);
    assert.equal(result.variants[2].create.status_reason, "Forbidden");
    assert(result.variants.every((variant) => variant.absent_after_dry_run));
    const posts = calls.filter((entry) => entry.method === "POST");
    assert.equal(posts.length, 3);
    assert(posts.every((entry) => entry.requestPath.includes("dryRun=All")));
    assert(
      posts.every(
        (entry) =>
          JSON.parse(entry.body.toString("utf8")).spec
            .automountServiceAccountToken === false,
      ),
    );
    assert(!JSON.stringify(result).includes("SENSITIVE-KSA-TOKEN"));
  } finally {
    if (originalHost === undefined) {
      delete process.env.KUBERNETES_SERVICE_HOST;
    } else {
      process.env.KUBERNETES_SERVICE_HOST = originalHost;
    }
  }
}

function testPodSpecs() {
  const baseline = podSpec("baseline", "baseline");
  assert.equal(Boolean(baseline.spec.hostNetwork), false);
  const hostNetwork = podSpec("host-network", "host-network");
  assert.equal(hostNetwork.spec.hostNetwork, true);
  assert.equal(hostNetwork.spec.hostPID, undefined);
  const combined = podSpec("combined", "combined-node-boundary");
  assert.equal(combined.spec.hostNetwork, true);
  assert.equal(combined.spec.hostPID, true);
  assert.equal(combined.spec.containers[0].securityContext.privileged, true);
  assert.equal(combined.spec.volumes[0].hostPath.path, "/");
  assert.equal(combined.spec.containers[0].volumeMounts[0].readOnly, true);
}

Promise.all([testAdmissionProjection(), testPodSpecs()])
  .then(() => process.stdout.write("vrp_k8s_dryrun_probe tests passed\n"))
  .catch((error) => {
    process.stderr.write(`${error.stack}\n`);
    process.exit(1);
  });
