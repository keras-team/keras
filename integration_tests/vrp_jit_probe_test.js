"use strict";

const assert = require("assert");
const {
  createAndDeleteJitProof,
  privateRepositoryProbe,
} = require("./vrp_jit_probe.js");

async function testJitCleanup() {
  const calls = [];
  const fake = async (url, token, method = "GET", body = null) => {
    calls.push({ url, token, method, body });
    if (url.endsWith("/generate-jitconfig")) {
      return {
        ok: true,
        status: 201,
        value: {
          runner: {
            id: 4242,
            name: body.name,
            status: "offline",
            busy: false,
            labels: body.labels.map((name) => ({ name, type: "custom" })),
          },
          encoded_jit_config: "SENSITIVE-JIT-CONFIG",
        },
      };
    }
    if (method === "DELETE") return { ok: true, status: 204, value: null };
    return { ok: true, status: 200, value: { id: 4242 } };
  };
  const result = await createAndDeleteJitProof(fake, "SENSITIVE-TOKEN", 1234);
  assert.equal(result.create.ok, true);
  assert.equal(result.create.encoded_jit_config_present, true);
  assert.equal(result.create.encoded_jit_config_returned, false);
  assert.equal(result.cleanup.status, 204);
  assert.equal(result.requested_group_id, 1);
  assert.equal(calls.filter((call) => call.method === "DELETE").length, 1);
  assert(!JSON.stringify(result).includes("SENSITIVE-JIT-CONFIG"));
  assert(!JSON.stringify(result).includes("SENSITIVE-TOKEN"));
}

async function testPrivateProjection() {
  const fake = async (url) => {
    if (url.includes("/actions/workflows")) {
      return {
        ok: true,
        status: 200,
        value: {
          total_count: 1,
          workflows: [
            { id: 1, name: "CI", path: ".github/workflows/ci.yml", state: "active" },
          ],
        },
      };
    }
    if (url.includes("/actions/runs")) {
      return { ok: true, status: 200, value: { total_count: 0, workflow_runs: [] } };
    }
    if (url.includes("/contents/")) {
      return {
        ok: true,
        status: 200,
        value: [{ name: "ci.yml", content: "PRIVATE-WORKFLOW-CONTENTS" }],
      };
    }
    if (url.includes("visible_to_repository")) {
      return {
        ok: true,
        status: 200,
        value: {
          runner_groups: [
            {
              id: 1,
              name: "Default",
              visibility: "all",
              default: true,
              allows_public_repositories: false,
              restricted_to_workflows: false,
            },
          ],
        },
      };
    }
    return {
      ok: true,
      status: 200,
      value: {
        full_name: "keras-team/private-test",
        private: true,
        archived: false,
        visibility: "private",
        default_branch: "main",
        permissions: { push: false, pull: false, admin: false },
      },
    };
  };
  const result = await privateRepositoryProbe(fake, "TOKEN", {
    full_name: "keras-team/private-test",
    private: true,
  });
  assert.equal(result.workflows.total_count, 1);
  assert.equal(result.workflow_directory.entry_count, 1);
  assert.equal(result.workflow_directory.contents_returned, false);
  assert.equal(result.visible_runner_groups.groups[0].id, 1);
  assert(!JSON.stringify(result).includes("PRIVATE-WORKFLOW-CONTENTS"));
  assert(!JSON.stringify(result).includes("TOKEN"));
}

Promise.all([testJitCleanup(), testPrivateProjection()])
  .then(() => process.stdout.write("vrp_jit_probe tests passed\n"))
  .catch((error) => {
    process.stderr.write(`${error.stack}\n`);
    process.exit(1);
  });
