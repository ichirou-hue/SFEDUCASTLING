# GitHub Actions deployment

The CI workflow validates pull requests targeting `main` and pushes to any branch. It runs backend checks against a fresh PostgreSQL database, applies Alembic migrations, runs pytest and Ruff, builds the frontend, and stores a commit-named bundle only for successful `main` runs. Production deployment is triggered only by a successful CI run caused by a push to `main`.

Production deployment is intentionally disabled until the server is provisioned and the GitHub production environment is protected. The deploy workflow checks `PRODUCTION_DEPLOY_ENABLED == true`, then downloads the artifact from the successful CI run for the exact commit. It serializes production deployments, uploads the archive over SSH with strict host key checking, applies migrations, switches the `current` symlink, restarts systemd, and checks the configured health URL. On restart or health failure it switches the application symlink back to the previous release. Database migrations are not automatically rolled back.

Before approving a production deployment, the reviewer must verify that a recent restorable database backup exists and that the release migrations are backward-compatible. The workflow does not create a database backup because the production database and approved backup mechanism have not been identified yet.

## Required GitHub configuration before enabling deployment

Create the `production` environment and restrict it to the `main` branch. Configure required reviewers if supported by the repository visibility and plan. Add these environment secrets:

- `SERVER_HOST`
- `SERVER_USER`
- `SERVER_SSH_KEY`
- `SERVER_KNOWN_HOSTS` with the independently verified host key, never a key collected with an unverified `ssh-keyscan`

Add these environment variables:

- `PRODUCTION_DEPLOY_ENABLED`, initially `false`, then `true` only after a deployment dry run
- `SERVER_DEPLOY_PATH`, for example `/srv/sfeducasling`
- `SERVER_SERVICE_NAME`, the preconfigured systemd service name
- `SERVER_HEALTH_URL`, the externally or locally reachable `/api/health` URL

The SSH account must be restricted to this application host and deployment directory, and must have only the required permission to restart the named systemd service. The server must be provisioned with `$SERVER_DEPLOY_PATH/venv/bin/alembic`, a shared `.env` at `$SERVER_DEPLOY_PATH/shared/.env`, the runtime dependencies, Stockfish if required, and a systemd unit whose working directory is `$SERVER_DEPLOY_PATH/current`. Configure database connection in the server-side `.env`; do not put production values in GitHub workflow files or the bundle.

Before the first deployment, confirm the server OS, current service command, Python/runtime dependency strategy, Maia3 availability, Stockfish binary, database backup and migration compatibility. The workflow does not install ML weights or roll back database schema automatically.

The manual `workflow_dispatch` path is restricted to `main` and requires a commit SHA that already has a successful CI artifact.
