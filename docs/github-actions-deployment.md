# GigaChess production CI/CD

Production releases may only come from `main`. CI runs on pushes and pull requests targeting `main`; it applies Alembic migrations to a clean PostgreSQL 16 database, runs Ruff and the complete backend test suite, and builds the frontend. A commit-pinned artifact is assembled only after a successful push build on `main`. The archive explicitly excludes local datasets, databases, credentials, virtual environments, and frontend dependencies.

The production workflow accepts only a successful CI run caused by a push to `main` in this repository. Manual dispatch is also restricted to `main` and requires an exact SHA with a successful CI artifact. Deployments are serialized and use strict SSH host-key checking. The root-owned server helper creates a full PostgreSQL backup before migrations, stages an immutable release, restarts `gigachess.service`, and checks `/api/health`; code rolls back if restart or health checks fail. Re-running a deployment for an already staged SHA reuses the release. Database schema rollback is not automatic.

## Server readiness

The latest repository record identifies the production host as Ubuntu `gigachess`, app path `/srv/SFEDUCASTLING`, service `gigachess.service`, loopback port `8005`, and deploy account `deploy`. Preserve the server-side `.env` and virtual environment. The latest recorded setup says Python 3.14.10 is installed at `/srv/SFEDUCASTLING/venv314`, while systemd still uses the old Python 3.12.3 environment. Confirm systemd is switched to the validated runtime, health checks pass, and a backup restore has been tested before enabling deployment.

Install `scripts/gigachess-deploy.sh` as root-owned `/usr/local/sbin/gigachess-deploy` (`root:root`, mode `0750`). The deploy account must have only this sudo permission:

```text
deploy ALL=(root) NOPASSWD: /usr/local/sbin/gigachess-deploy
```

Validate it with `visudo -cf /etc/sudoers.d/gigachess-deploy`. The helper uses `runuser -u postgres -- pg_dumpall`, stores mode-0600 backups under `/var/backups/gigachess`, and retains them for 30 days.

## GitHub configuration

Create the `production` Environment and restrict it to the `main` branch. Keep `PRODUCTION_DEPLOY_ENABLED=false` until the server runtime, backup restore, deploy helper, SSH access, and exact-SHA dry run are verified. Configure Environment secrets `SERVER_HOST`, `SERVER_USER`, `SERVER_SSH_KEY`, and `SERVER_KNOWN_HOSTS` with independently checked values. The current deploy workflow uses the fixed server paths and service configured in the root-owned helper. Never put credentials in the repository or deployment artifact.
