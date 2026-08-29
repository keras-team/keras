"use strict";

const assert = require("assert");
const {
  gcpResourcePermissions,
  goalWritePermissions,
  permissionResult,
} = require("./vrp_gcp_resource_permissions_probe.js");

function response(value, status = 200) {
  return { ok: status >= 200 && status < 300, status, value };
}

async function testResourcePermissionProjection() {
  const calls = [];
  const metadata = async (options) => {
    calls.push({ kind: "metadata", options });
    return response({ access_token: "SENSITIVE-GCP-ACCESS-TOKEN" });
  };
  const api = async (url, token, method = "GET", body = null) => {
    calls.push({ kind: "api", url, token, method, body });
    if (url.includes("oauth2.googleapis.com/tokeninfo")) {
      return response({
        email: "test.svc.id.goog",
        scope: "https://www.googleapis.com/auth/cloud-platform",
        expires_in: 1234,
      });
    }
    if (
      url.includes("artifactregistry.googleapis.com") &&
      url.endsWith(":testIamPermissions")
    ) {
      return response({
        permissions: token
          ? [
              "artifactregistry.repositories.downloadArtifacts",
              "artifactregistry.repositories.uploadArtifacts",
              "artifactregistry.packages.update",
            ]
          : ["artifactregistry.repositories.downloadArtifacts"],
      });
    }
    if (url.includes("storage.googleapis.com") && url.includes("testPermissions")) {
      return response({
        permissions: ["storage.objects.get", "storage.objects.create"],
      });
    }
    if (url.includes("storage.googleapis.com") && url.includes("/o?")) {
      return token
        ? response({ items: [{ name: "PRIVATE-NAME-NOT-RETURNED" }] })
        : response({ message: "Anonymous caller denied" }, 403);
    }
    if (url.includes("secretmanager.googleapis.com")) {
      if (url.endsWith(":testIamPermissions")) {
        return response({ permissions: [] });
      }
      return response({ message: "not found" }, 404);
    }
    if (url.includes("pubsub.googleapis.com")) {
      if (url.endsWith(":testIamPermissions")) {
        return response({
          permissions: token ? ["pubsub.topics.get", "pubsub.topics.publish"] : [],
        });
      }
      return token
        ? response({ name: "projects/test/topics/topic" })
        : response({ message: "denied" }, 403);
    }
    if (url.endsWith("/entries:list")) {
      return response({ entries: [] });
    }
    if (url.includes("logging.googleapis.com") && url.includes("/logs?")) {
      return response({ logNames: ["PRIVATE-LOG-NAME-NOT-RETURNED"] });
    }
    if (url.includes("artifactregistry.googleapis.com")) {
      if (url.includes("/packages?")) {
        return response({ packages: [{ name: "PACKAGE-NAME-NOT-RETURNED" }] });
      }
      return response({
        name: "projects/test/locations/us/repositories/repo",
        format: "DOCKER",
        mode: "STANDARD_REPOSITORY",
        description: "test",
        registryUri: "us-docker.pkg.dev/test/repo",
      });
    }
    throw new Error(`unexpected URL: ${url}`);
  };

  const result = await gcpResourcePermissions(api, metadata);
  assert.equal(result.token_obtained_in_process, true);
  assert.deepEqual(
    result.artifact_registry.test_iam_permissions.goal_write_permissions,
    [
      "artifactregistry.packages.update",
      "artifactregistry.repositories.uploadArtifacts",
    ],
  );
  assert.equal(
    result.artifact_registry.anonymous_test_iam_permissions
      .goal_write_permissions.length,
    0,
  );
  assert.deepEqual(
    result.cloud_storage[0].test_iam_permissions.goal_write_permissions,
    ["storage.objects.create"],
  );
  assert.deepEqual(
    result.pubsub.test_iam_permissions.goal_write_permissions,
    ["pubsub.topics.publish"],
  );
  assert(
    result.pubsub.test_iam_permissions.allowed_permissions.includes(
      "pubsub.topics.publish",
    ),
  );
  assert(
    calls.some(
      (call) =>
        call.kind === "api" &&
        call.method === "POST" &&
        call.url.endsWith(":testIamPermissions"),
    ),
  );
  assert(
    calls.every(
      (call) =>
        call.kind !== "api" ||
        call.method === "GET" ||
        call.url.endsWith(":testIamPermissions") ||
        call.url.includes("/iam/testPermissions") ||
        call.url.includes("pubsub.googleapis.com") ||
        call.url.endsWith("/entries:list"),
    ),
  );
  assert(!JSON.stringify(result).includes("SENSITIVE-GCP-ACCESS-TOKEN"));
  assert(!JSON.stringify(result).includes("PRIVATE-NAME-NOT-RETURNED"));
  assert(!JSON.stringify(result).includes("PRIVATE-LOG-NAME-NOT-RETURNED"));
  assert(!JSON.stringify(result).includes("PACKAGE-NAME-NOT-RETURNED"));
}

function testPermissionHelpers() {
  assert.deepEqual(
    goalWritePermissions([
      "artifactregistry.files.download",
      "artifactregistry.files.upload",
      "artifactregistry.tags.create",
    ]),
    ["artifactregistry.files.upload", "artifactregistry.tags.create"],
  );
  const result = permissionResult(
    response({ permissions: ["storage.objects.get", "storage.objects.delete"] }),
  );
  assert.deepEqual(result.goal_write_permissions, ["storage.objects.delete"]);
}

Promise.all([testResourcePermissionProjection(), testPermissionHelpers()])
  .then(() => process.stdout.write("vrp_gcp_resource_permissions_probe tests passed\n"))
  .catch((error) => {
    process.stderr.write(`${error.stack}\n`);
    process.exit(1);
  });
