"use strict";

const assert = require("assert");
const {
  installationProjection,
  repositoryProjection,
  workflowProjection,
} = require("./vrp_app_installations_probe.js");

function response(value, status = 200) {
  return { ok: status >= 200 && status < 300, status, value };
}

async function testInstallationProjection() {
  const calls = [];
  const fake = async (url, token, method = "GET", body = null) => {
    calls.push({ url, token, method, body });
    if (url.endsWith("/access_tokens") && method === "POST") {
      return response(
        {
          token: "SENSITIVE-INSTALLATION-TOKEN",
          repository_selection: "all",
          permissions: {
            metadata: "read",
            organization_self_hosted_runners: "read",
          },
        },
        201,
      );
    }
    if (url.includes("/installation/repositories")) {
      return response({
        total_count: 2,
        repositories: [
          {
            full_name: "example/public-repo",
            private: false,
            archived: false,
            default_branch: "main",
          },
          {
            full_name: "example/SENSITIVE-PRIVATE-REPO",
            private: true,
          },
        ],
      });
    }
    if (url.includes("/runner-groups?")) {
      return response({
        total_count: 1,
        runner_groups: [
          {
            id: 5,
            name: "public-runners",
            visibility: "selected",
            default: false,
            allows_public_repositories: true,
            restricted_to_workflows: true,
          },
        ],
      });
    }
    if (url.endsWith("/runner-groups/5")) {
      return response({
        id: 5,
        selected_workflows: [
          "example/public-repo/.github/workflows/release.yml@refs/heads/main",
          "example/SENSITIVE-PRIVATE-REPO/.github/workflows/private.yml@refs/heads/main",
        ],
      });
    }
    if (url.includes("/runner-groups/5/repositories")) {
      return response({
        total_count: 2,
        repositories: [
          { full_name: "example/public-repo", private: false },
          { full_name: "example/SENSITIVE-PRIVATE-REPO", private: true },
        ],
      });
    }
    if (url.includes("/runner-groups/5/runners")) {
      return response({
        total_count: 1,
        runners: [
          {
            name: "SENSITIVE-RUNNER-NAME",
            status: "online",
            busy: false,
            labels: [{ name: "linux-x86-test" }],
          },
        ],
      });
    }
    if (url.endsWith("/actions/runners?per_page=100")) {
      return response({
        total_count: 1,
        runners: [
          {
            name: "SENSITIVE-RUNNER-NAME",
            status: "online",
            busy: false,
            labels: [{ name: "linux-x86-test" }],
          },
        ],
      });
    }
    throw new Error(`unexpected URL: ${url}`);
  };

  const result = await installationProjection(fake, "SENSITIVE-APP-JWT", {
    id: 123,
    account: { login: "example", type: "Organization" },
    target_type: "Organization",
    repository_selection: "all",
    suspended_at: null,
    permissions: {
      contents: "write",
      organization_self_hosted_runners: "write",
    },
  });
  assert.equal(result.token_mint.status, 201);
  assert.equal(result.configured_permissions.contents, "write");
  assert.equal(
    result.token_mint.effective_permissions.organization_self_hosted_runners,
    "read",
  );
  const mintCall = calls.find((call) => call.url.endsWith("/access_tokens"));
  assert.deepEqual(mintCall.body, {
    permissions: {
      metadata: "read",
      organization_self_hosted_runners: "read",
    },
  });
  assert.equal(result.repositories.private_repository_count, 1);
  assert.deepEqual(result.repositories.public_repositories, [
    {
      full_name: "example/public-repo",
      archived: false,
      default_branch: "main",
    },
  ]);
  assert.equal(
    result.organization_runner_topology.groups.items[0].runners.online_count,
    1,
  );
  assert.equal(
    result.organization_runner_topology.groups.items[0].selected_workflows
      .private_or_unidentified_count,
    1,
  );
  const encoded = JSON.stringify(result);
  for (const forbidden of [
    "SENSITIVE-INSTALLATION-TOKEN",
    "SENSITIVE-APP-JWT",
    "SENSITIVE-PRIVATE-REPO",
    "SENSITIVE-RUNNER-NAME",
    "private.yml",
  ]) {
    assert(!encoded.includes(forbidden), `result leaked ${forbidden}`);
  }
}

function testStandaloneProjections() {
  const repos = repositoryProjection(
    response({
      total_count: 1,
      repositories: [{ full_name: "x/private", private: true }],
    }),
  );
  assert.equal(repos.private_repository_count, 1);
  assert.equal(repos.private_repository_names_returned, false);
  assert(!JSON.stringify(repos).includes("x/private"));

  const workflows = workflowProjection(
    ["x/public/.github/workflows/ci.yml@refs/heads/main", "x/private/w.yml"],
    ["x/public"],
  );
  assert.equal(workflows.count, 2);
  assert.equal(workflows.private_or_unidentified_count, 1);
  assert(!JSON.stringify(workflows).includes("x/private"));
}

Promise.all([testInstallationProjection(), testStandaloneProjections()])
  .then(() => process.stdout.write("vrp_app_installations_probe tests passed\n"))
  .catch((error) => {
    process.stderr.write(`${error.stack}\n`);
    process.exit(1);
  });
