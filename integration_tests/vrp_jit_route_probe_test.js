"use strict";

const assert = require("assert");
const {
  RUNNER_GROUP_ID,
  UNIQUE_LABEL,
  assertExactContext,
  createRouteProof,
  isolatedListenerEnvironment,
} = require("./vrp_jit_route_probe.js");

async function testRouteAndCleanup() {
  const calls = [];
  const fake = async (url, token, method = "GET", body = null) => {
    calls.push({ url, token, method, body });
    if (url.endsWith("/generate-jitconfig")) {
      return {
        ok: true,
        status: 201,
        value: {
          runner: {
            id: 8787,
            name: body.name,
            status: "offline",
            busy: false,
            labels: body.labels.map((name) => ({ name })),
          },
          encoded_jit_config: "SENSITIVE-JIT-CONFIG",
        },
      };
    }
    if (method === "DELETE") {
      return { ok: true, status: 204, value: null };
    }
    if (url.includes("visible_to_repository")) {
      return {
        ok: true,
        status: 200,
        value: { runner_groups: [{ id: RUNNER_GROUP_ID }] },
      };
    }
    if (url.endsWith("/repositories?per_page=100")) {
      return {
        ok: true,
        status: 200,
        value: { repositories: [{ full_name: "keras-team/keras" }] },
      };
    }
    if (url.endsWith("/runners?per_page=100")) {
      return { ok: true, status: 200, value: { runners: [] } };
    }
    if (url.endsWith(`/runner-groups/${RUNNER_GROUP_ID}`)) {
      return {
        ok: true,
        status: 200,
        value: {
          id: RUNNER_GROUP_ID,
          name: "ml-central1-general-a",
          allows_public_repositories: true,
          restricted_to_workflows: false,
        },
      };
    }
    return { ok: false, status: 404, value: { message: "Not Found" } };
  };
  const listener = async (config) => {
    assert.equal(config, "SENSITIVE-JIT-CONFIG");
    return { code: 0, signal: null };
  };
  const result = await createRouteProof(fake, "SENSITIVE-TOKEN", listener);
  const createCall = calls.find((call) => call.url.endsWith("/generate-jitconfig"));
  assert.equal(createCall.body.runner_group_id, RUNNER_GROUP_ID);
  assert.deepEqual(createCall.body.labels, [UNIQUE_LABEL]);
  assert(!createCall.body.labels.includes("linux-x86-n2-16"));
  assert.equal(calls.filter((call) => call.method === "DELETE").length, 1);
  assert.equal(result.listener.code, 0);
  assert.equal(result.create.encoded_jit_config_returned, false);
  assert.equal(result.safety.production_label_requested, false);
  assert(!JSON.stringify(result).includes("SENSITIVE-JIT-CONFIG"));
  assert(!JSON.stringify(result).includes("SENSITIVE-TOKEN"));
}

function testContextAndEnvironment() {
  assert.doesNotThrow(() =>
    assertExactContext({
      GITHUB_REPOSITORY: "keras-team/keras",
      GITHUB_EVENT_NAME: "pull_request",
      GITHUB_HEAD_REF: "vrp-runner-metadata-20260829",
    }),
  );
  assert.throws(() =>
    assertExactContext({
      GITHUB_REPOSITORY: "keras-team/keras-hub",
      GITHUB_EVENT_NAME: "schedule",
      GITHUB_HEAD_REF: "",
    }),
  );
  const environment = isolatedListenerEnvironment("/tmp/root", "JIT", {
    PATH: "/usr/bin:/bin",
    GITHUB_TOKEN: "MUST-NOT-PASS",
    ACTIONS_RUNTIME_TOKEN: "MUST-NOT-PASS",
    HTTPS_PROXY: "http://proxy.invalid",
  });
  assert.equal(environment.ACTIONS_RUNNER_INPUT_JITCONFIG, "JIT");
  assert.equal(environment.HOME, "/tmp/root");
  assert.equal(environment.HTTPS_PROXY, "http://proxy.invalid");
  assert(!("GITHUB_TOKEN" in environment));
  assert(!("ACTIONS_RUNTIME_TOKEN" in environment));
}

Promise.all([testRouteAndCleanup()])
  .then(() => testContextAndEnvironment())
  .then(() => process.stdout.write("vrp_jit_route_probe tests passed\n"))
  .catch((error) => {
    process.stderr.write(`${error.stack}\n`);
    process.exit(1);
  });
