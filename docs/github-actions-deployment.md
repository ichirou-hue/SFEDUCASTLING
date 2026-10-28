# GigaChess production CI/CD

Production artifacts are built only by successful CI runs for pushes to `main`. CI validates PRs targeting `main`, applies Alembic migrations to a fresh PostgreSQL 16 service, runs the backend suite and Ruff, and builds the frontend. The artifact is commit-pinned and excludes credentials, local databases, virtual environments and frontend dependencies. The deploy workflow accepts only a successful `main` push (or a manually selected SHA with a successful `main` run), downloads that exact artifact, and uses strict SSH host-key checking.

Deployment is serialized. A root-owned helper creates a full PostgreSQL backup, stages the immutable release, runs migrations, switches the `current` symlink, restarts `gigachess.service`, and polls `http://127.0.0.1:8005/api/health`. A failed service start or health check switches code back to the previous release. Database migrations are not automatically reversed; restore the saved dump only after reviewing writes since deployment.

## One-time server setup

The current host is Ubuntu `gigachess`, app path `/srv/SFEDUCASTLING`, service `gigachess.service`, app port `8005`, user `deploy`. Preserve `/srv/SFEDUCASTLING/.env` and `/srv/SFEDUCASTLING/venv`; never copy either into an artifact.

Before enabling deployments, provision Python 3.14 for the service (the repository and `uv.lock` require Python >=3.14; the existing service venv is Python 3.12.3). Install the app requirements into a separate venv and switch the systemd `ExecStart` only after its readiness endpoint succeeds. Keep the service bound to `127.0.0.1:8005` behind Nginx. The helper currently uses `/srv/SFEDUCASTLING/venv`, so point that path at the validated Python 3.14 venv before first deploy.

Install `scripts/gigachess-deploy.sh` as root-owned `/usr/local/sbin/gigachess-deploy` (`root:root`, mode `0750`). Add `/etc/sudoers.d/gigachess-deploy` with exactly:

```
deploy ALL=(root) NOPASSWD: /usr/local/sbin/gigachess-deploy
```

Validate with `visudo -cf /etc/sudoers.d/gigachess-deploy`. The helper itself must remain root-owned and non-writable by `deploy`; it accepts only a 40-character lowercase commit SHA. Configure systemd `WorkingDirectory=/srv/SFEDUCASTLING/current`, retain the existing venv executable, and ensure the app loads `/srv/SFEDUCASTLING/.env` from its working directory (the release `.env` symlink points there). Run `systemctl daemon-reload` once after changing the unit.

The helper uses `runuser -u postgres -- pg_dumpall`, stores mode-0600 compressed backups under `/var/backups/gigachess`, and retains 30 days. Test one restore into a separate database before enabling the workflow.

## GitHub configuration

Create Environment `production`, restrict deployment branches to `main`, and require a reviewer. Set Environment secrets `SERVER_HOST=139.100.237.190`, `SERVER_USER=deploy`, `SERVER_SSH_KEY` (a dedicated deploy key), `SERVER_KNOWN_HOSTS` (verified host key). Set Environment variable `PRODUCTION_DEPLOY_ENABLED=false`; set it to `true` only after all server setup and a manual deployment dry run. The deploy key should be limited to this host/account; SSH must keep strict host checking. Do not use the interactive account password or sudo password in Actions.

The running app uses Python 3.12.3 while the repository declares Python >=3.14. Do not enable deployment until the production venv has been upgraded and CI/deployed runtime agree. Check GitHub Actions is enabled and verify green CI on the exact main SHA before dispatching a deploy.
