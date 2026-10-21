#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
command -v python3.12 >/dev/null || { echo 'Нужен Python 3.12.'; exit 1; }
command -v docker >/dev/null || { echo 'Нужен Docker Engine с Compose plugin.'; exit 1; }
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    docker_cmd=(sudo docker)
fi
"${docker_cmd[@]}" info >/dev/null
"${docker_cmd[@]}" compose version
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
"${docker_cmd[@]}" compose up -d --wait --wait-timeout 90 postgres
.venv/bin/python -m scripts.check_runtime
HUNTERILKA_REQUIRE_DB=1 .venv/bin/python -m unittest discover -s tests -v
