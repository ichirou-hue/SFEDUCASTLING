#!/usr/bin/env bash
# Совместимая обёртка: основной скрипт запуска — ./run.sh
exec "$(cd "$(dirname "$0")" && pwd)/run.sh" "$@"
