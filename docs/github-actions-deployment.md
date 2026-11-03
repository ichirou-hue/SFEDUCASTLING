# GigaChess production CI/CD

Production releases may only come from `main`. CI runs on pushes and pull requests targeting `main`; it applies Alembic migrations to a clean PostgreSQL 16 database, runs Ruff and the complete backend test suite, and builds the frontend. A commit-pinned artifact is assembled only after a successful push build on `main`. The archive explicitly excludes local datasets, databases, credentials, virtual environments, and frontend dependencies.

The production workflow accepts only a successful CI run caused by a push to `main` in this repository. Manual dispatch is also restricted to `main` and requires an exact SHA with a successful CI artifact. Deployments are serialized and use strict SSH host-key checking. The root-owned server helper creates a full PostgreSQL backup before migrations, stages an immutable release, restarts `gigachess.service`, and checks `/api/health`; code rolls back if restart or health checks fail. Re-running a deployment for an already staged SHA reuses the release. Database schema rollback is not automatic.

## Server readiness

The production service uses `/srv/SFEDUCASTLING/venv/bin/python` (Python 3.14.10), with `/srv/SFEDUCASTLING/current` as its working directory and deploy as its user. The deploy helper reuses this shared venv and does not install application dependencies; `requirements-ci.txt` intentionally omits ML runtime packages. Maia3 therefore needs one-time provisioning in that venv after a fresh server setup or venv rebuild. Run `bash /srv/SFEDUCASTLING/current/scripts/provision-maia3-runtime.sh` as deploy: it installs CPU PyTorch, NumPy and Hugging Face Hub, then caches the 5M Maia3 checkpoint under the service account. `backend/api_gateway/state.py` launches the checked-out Maia3 submodule from its source directory, so the editable package install is not needed. Keep the `.env` private (mode `0600`), and preserve the shared venv across releases.

Configure Nginx to serve `/srv/SFEDUCASTLING/current/frontend/dist`; the `releases` directory and active release root must be searchable by Nginx (mode `0755`). Confirm systemd, health checks, and a backup restore before enabling deployment.

Install `scripts/gigachess-deploy.sh` as root-owned `/usr/local/sbin/gigachess-deploy` (`root:root`, mode `0750`). The deploy account must have only this sudo permission:

```text
deploy ALL=(root) NOPASSWD: /usr/local/sbin/gigachess-deploy
```

Validate it with `visudo -cf /etc/sudoers.d/gigachess-deploy`. The helper uses `runuser -u postgres -- pg_dumpall`, stores mode-0600 backups under `/var/backups/gigachess`, and retains them for 30 days.

## GitHub configuration

Create the `production` Environment and restrict it to the `main` branch. Store `PRODUCTION_DEPLOY_ENABLED` as a **repository variable** and keep it `false` until the server runtime, backup restore, deploy helper, SSH access, and exact-SHA deployment are verified. The job checks this flag before GitHub loads Environment-scoped values. Keep `SERVER_HOST`, `SERVER_USER`, `SERVER_SSH_KEY`, and `SERVER_KNOWN_HOSTS` as secrets in the `production` Environment. The manual deployment path explicitly selects this repository when looking up the successful CI artifact, so it works without checking out the source repository. The current deploy workflow uses the fixed server paths and service configured in the root-owned helper. Never put credentials in the repository or deployment artifact.
