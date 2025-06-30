---
title: A Cheatsheet on Identities, Tokens, and Keyless Authentication on Google Cloud
date: 2025-07-02 14:00:00 +0100
categories: [Google-Cloud, Cheatsheets]
tags: [Google-Cloud, Cheatsheet, OAuth, OIDC, IAM, Sevice Accounts, Google Cloud Cheatsheet]
image:
  path: /assets/img/gcp-art.png
  alt: "Google Cloud Cheatsheet"
---

This guide is for developers and operators who want to master modern Google Cloud identities, tokens, and keyless authentication. We will dissect the patterns that eliminate static keys, enhance security, and make your automation robust and auditable. This is your definitive cheatsheet for moving from risky, legacy practices to secure, keyless authentication within Google Cloud.

### Foundational Identities: The "Who" of Your Automation

Before any action is taken, we must define the actor. In Google Cloud, there are two types of non-human identities.

#### Service Accounts: Your Robot Employees

A **Service Account** is an identity you create and manage for your applications, scripts, or virtual machines.

*   **Purpose:** To give a programmatic workload a distinct identity so you can grant it specific, limited permissions (e.g., "This application can read from this Cloud Storage bucket and nothing else").
*   **Analogy:** Think of it as a **robot employee**. You build the robot (create the SA), give it a specific job description (grant it IAM roles), and it performs tasks on your behalf. You are responsible for its entire lifecycle.

**Cheatsheet: Creating a Service Account**
```bash
gcloud iam service-accounts create my-app-sa \
  --project="my-gcp-project" \
  --display-name="My Application's Identity"
```

#### Service Agents: Google's Specialized Workforce

A **Service Agent** is a special, Google-managed service account that is automatically created when you enable and use a specific Google Cloud service (like Cloud Build, Cloud Run, or Pub/Sub).

*   **Purpose:** It allows a Google service to act on your behalf to manage resources in your project.
*   **Analogy:** If a Service Account is your robot employee, a Service Agent is the **specialized delivery driver** from a company you hired. They show up automatically to do their specific job, and you just need to ensure the gate is open for them (grant them the right permissions).
*   **Example:** When a Cloud Build job needs to deploy your application to Cloud Run, it's the Cloud Build *Service Agent* that performs the deployment. You must grant this agent the `roles/run.admin` role.

**Cheatsheet: Granting Permissions to a Service Agent**

```bash
# Get your project number (used in service agent emails)
PROJECT_NUMBER=$(gcloud projects describe "my-gcp-project" --format="value(projectNumber)")

# Identify the Cloud Build Service Agent
CLOUDBUILD_SA="service-${PROJECT_NUMBER}@gcp-sa-cloudbuild.iam.gserviceaccount.com"

# Grant it the permission to deploy to Cloud Run
gcloud projects add-iam-policy-binding "my-gcp-project" \
  --member="serviceAccount:${CLOUDBUILD_SA}" \
  --role="roles/run.admin"
```

### Using Identities: The Core Security Choice (`activate` vs. `impersonate`)

Now that we know *what* these identities are, we must explore the two fundamentally different ways to *use* them from our command line. From here on, we will only deal with service accounts, as they are the primary identity for automation and application workloads.

#### `gcloud auth activate-service-account` (Legacy, High-Risk)

*   **The "How":** You download a JSON private key. This command configures `gcloud` to **BECOME** that identity by using the key for all future API calls.
*   **The Risk:** The key is a long-lived, static secret. If leaked, an attacker has persistent access. The audit trail is weak; it shows the service account acted, but not *who* used the key.
*   **Required Role:** None for the user. Authorization is based entirely on possessing the key file.

#### `--impersonate-service-account` (Modern, Secure Best Practice)

*   **The "How":** You, as an authenticated user (`joe@example.com`), ask Google's IAM service for permission to temporarily **ACT AS** another service account. If authorized, IAM issues a short-lived (1-hour) access token.
*   **The Security:** No static keys are exposed. Access is temporary. The audit trail is crystal clear: `"Principal: joe@example.com" impersonated "Service Account: my-app-sa@..."`.
*   **Required Role:** The principal (you) must have the **`roles/iam.serviceAccountUser`** role on the target service account.

**Cheatsheet: Enabling Impersonation**

```bash
# Grant a user permission to impersonate a service account
gcloud iam service-accounts add-iam-policy-binding \
  my-app-sa@my-gcp-project.iam.gserviceaccount.com \
  --member="user:joe@example.com" \
  --role="roles/iam.serviceAccountUser"
```

**Cheatsheet: Impersonate a service account securely**

```bash
# Now, Joe can impersonate the service account without needing a key
gcloud auth login
gcloud config set auth/impersonate_service_account my-app-sa@...
# ...do your work...
gcloud config unset auth/impersonate_service_account
```

### Deep Dive on Tokens — The Currency of Modern Auth

Tokens are not monolithic. They are specialized credentials for specific jobs. Using the right token for the right task is crucial.

#### Dissecting Token Types

| Token Type | `gcloud` Command | Purpose | Key Use Case |
| :--- | :--- | :--- | :--- |
| **Access Token** | `print-access-token` | **Authorization**: Answers "What can you do?". It's an opaque string presented to a Google API to prove you have permission to perform an action. | Calling Google-owned APIs like `storage.googleapis.com` or `compute.googleapis.com`. |
| **ID Token** | `print-identity-token` | **Authentication**: Answers "Who are you?". It's a structured [JSON Web Token (JWT)](https://jwt.io/) that proves the identity of the caller to a third party (like your own application). | Authenticating to your own applications on Cloud Run, Cloud Functions, or services behind IAP. |

#### Advanced Token Generation Cheatsheet

```bash
# Get an Access Token as your current user
TOKEN=$(gcloud auth print-access-token)

# Use it to call a Google API directly (great for scripting/debugging)
curl -H "Authorization: Bearer $TOKEN" \
  "https://compute.googleapis.com/compute/v1/projects/my-gcp-project/zones/us-central1-a/instances"

# Get an ID Token to call YOUR service. The --audiences flag is CRITICAL.
# It specifies the intended recipient, preventing token replay attacks.
# Your service MUST validate that the 'aud' claim matches its own identity.
ID_TOKEN=$(gcloud auth print-identity-token --audiences="https://my-secure-service.a.run.app")

# Call your service, which will validate the ID token to authenticate you
curl -H "Authorization: Bearer $ID_TOKEN" "https://my-secure-service.a.run.app"

# Generate a token by impersonating a service account (combines concepts)
# Your user needs roles/iam.serviceAccountTokenCreator on 'target-sa@...'
IMPERSONATED_ID_TOKEN=$(gcloud auth print-identity-token \
  --impersonate-service-account="target-sa@my-gcp-project.iam.gserviceaccount.com" \
  --audiences="https://my-secure-service.a.run.app" \
  --include-email)
```

#### Token Validation and Troubleshooting

Is a token not working? Don't guess. Inspect it.

1.  **Use the `tokeninfo` endpoint:** This is your best friend for debugging any Google OAuth2 token.

    ```bash
    # For an Access Token
    curl https://www.googleapis.com/oauth2/v3/tokeninfo?access_token=<YOUR_ACCESS_TOKEN>

    # For an ID Token
    curl https://oauth2.googleapis.com/tokeninfo?id_token=<YOUR_ID_TOKEN>
    ```

    This will show you its `scope` (for access tokens), `aud` (for ID tokens), `exp` (expiry time), and the identity (`email`, `sub`) it represents.

2.  **Decode the JWT:** Since ID Tokens are JWTs, you can decode the payload to see its claims. Use a local tool or library for sensitive tokens. This is great for seeing exactly what `iss` (issuer) and `aud` (audience) claims were generated.

### Deep Dive on the Metadata Server — Keyless Auth Inside GCP

For any code running inside a Google Cloud environment (GCE, GKE, Cloud Run, etc.), the Metadata Server is the automatic, secure source of credentials (among other things!).

#### How It Works

Every GCP resource can access a special IP address: **`169.254.169.254`** (or the DNS name `metadata.google.internal`). Your code can make simple, unauthenticated HTTP requests to this address to get information about its own environment, including temporary credentials.

**The Golden Rule:** Always use a Google Client Library that supports **Application Default Credentials (ADC)**. The library handles the complexity of querying the metadata server, caching tokens, and refreshing them before they expire. The low-level `curl` commands below are for understanding the mechanism and for shell scripting.

#### Metadata Server Cheatsheet (for use *inside* a GCP resource)

**Required Header:** All requests MUST include `-H "Metadata-Flavor: Google"` as a security measure.

**1. Getting an Access Token (to call Google APIs):**
This token is derived from the service account attached to the resource.

```bash
# Get the raw token string
curl "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
  -H "Metadata-Flavor: Google"

# Get a full JSON response with the token and its expiry time
curl "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token?format=full" \
  -H "Metadata-Flavor: Google"
```

**2. Getting an ID Token (to call your own services):**
This token proves the identity of the workload itself. **The `audience` parameter is mandatory.**

```bash
# Define the audience (the service you intend to call)
AUDIENCE="https://my-other-service.a.run.app"

# Request an ID token for that specific audience
curl "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=${AUDIENCE}" \
  -H "Metadata-Flavor: Google"
```

**3. Getting other instance metadata:**
The server provides a wealth of information beyond just tokens.
```bash
# Get the instance's project ID
curl "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google"

# Get the instance's network tags
curl "http://metadata.google.internal/computeMetadata/v1/instance/tags" -H "Metadata-Flavor: Google"
```

---

### The Path to Mastery — Your Action Plan

Theory is great, but action creates security. Here is your checklist to modernize your `gcloud` and GCP practices today.

**1. Eradicate Service Account Keys:**

*   **Audit for Keys:** Use `gcloud iam service-accounts keys list --iam-account=<SA_EMAIL>` to find existing keys. Ask why each one exists.
*   **Set an Expiry Policy:** For keys you absolutely cannot remove yet, use an Organization Policy to enforce a maximum lifetime (e.g., 90 days).
*   **Replace with Keyless Alternatives:**
    *   **Local Development:** Use `gcloud auth login` and impersonation.
    *   **CI/CD (GitHub, etc.):** Use Workload Identity Federation.
    *   **In-GCP Workloads:** Use service accounts (which uses Metadata Server).

**2. Make Impersonation the Default:**

Train your team to stop using `activate-service-account`. The secure alternative is easy and auditable.

```bash
# The modern workflow for a developer needing to act as an application
gcloud auth login
gcloud config set auth/impersonate_service_account my-app-sa@...
# ...do your work...
gcloud config unset auth/impersonate_service_account
```

**3. Adopt the Principle of Least Privilege for Identities:**

*   **Dedicated Service Accounts:** Every application, VM, or distinct task should have its own service account. **Never use the default Compute Engine service account in production.**
*   **Grant Roles, Not Scopes:** When creating VMs, use the broad `cloud-platform` scope and control access with fine-grained IAM roles on the service account itself.
*   **Use the Right IAM Roles:** Grant `serviceAccountTokenCreator` for service-to-service calls, not the broader `serviceAccountUser`. The role should match the job. The two key roles to master are:
    *   **`roles/iam.serviceAccountUser`**: For developers and tools like Terraform to fully impersonate an SA.
    *   **`roles/iam.serviceAccountTokenCreator`**: For applications needing to get tokens for other services.

By internalizing these patterns — starting with the "what" of identities and moving to the secure "how" of using them — you transition from simply running commands to architecting secure, scalable, and professional systems on Google Cloud.