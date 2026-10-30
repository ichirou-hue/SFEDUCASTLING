#!/usr/bin/env bash
set -Eeuo pipefail

# Install root-owned as /usr/local/sbin/gigachess-deploy.
readonly root=/srv/SFEDUCASTLING
readonly service=gigachess.service
readonly health_url=http://127.0.0.1:8005/api/health
readonly sha="${1:-}"
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid commit SHA" >&2; exit 2; }
readonly archive="/tmp/gigachess-${sha}.tar.gz"
readonly releases="$root/releases"
readonly release="$releases/$sha"
readonly backup_dir=/var/backups/gigachess
readonly timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
staging=""
cleanup_staging() {
  if [[ -n "$staging" && -d "$staging" ]]; then rm -rf -- "$staging"; fi
}
trap cleanup_staging EXIT
readonly previous="$(readlink -f "$root/current" || true)"
test -s "$archive"
install -d -o deploy -g deploy -m 0750 "$releases"
install -d -o root -g root -m 0700 "$backup_dir"

# Back up every PostgreSQL database before migrations; never print credentials.
runuser -u postgres -- pg_dumpall --clean --if-exists | gzip -9 > "$backup_dir/postgres-$timestamp.sql.gz"
chmod 0600 "$backup_dir/postgres-$timestamp.sql.gz"

if [[ ! -d "$release" ]]; then
  staging="$releases/.${sha}.staging.$$"
  install -d -o deploy -g deploy -m 0750 "$staging"
  tar -xzf "$archive" -C "$staging" --no-same-owner
  ln -s "$root/.env" "$staging/.env"
  ln -s "$root/venv" "$staging/venv"
  mv "$staging" "$release"
  staging=""
else
  test -f "$release/backend/app.py"
  test -L "$release/.env"
fi

if ! (cd "$release" && "$root/venv/bin/alembic" upgrade head); then
  echo "Migration failed; code was not switched. Backup: $backup_dir/postgres-$timestamp.sql.gz" >&2
  exit 1
fi
ln -s "$release" "$root/current.next"
mv -Tf "$root/current.next" "$root/current"
if ! systemctl restart "$service"; then
  if [[ -n "$previous" ]]; then
    ln -s "$previous" "$root/current.rollback"
    mv -Tf "$root/current.rollback" "$root/current"
    systemctl restart "$service" || true
  fi
  exit 1
fi
for _ in $(seq 1 30); do
  if curl --fail --silent --show-error "$health_url" >/dev/null; then
    rm -f "$archive"
    find "$backup_dir" -type f -name 'postgres-*.sql.gz' -mtime +30 -delete
    exit 0
  fi
  sleep 2
done
if [[ -n "$previous" ]]; then
  ln -s "$previous" "$root/current.rollback"
  mv -Tf "$root/current.rollback" "$root/current"
  systemctl restart "$service" || true
fi
echo "Health check failed; code rolled back. Review migration and backup $backup_dir/postgres-$timestamp.sql.gz." >&2
exit 1
