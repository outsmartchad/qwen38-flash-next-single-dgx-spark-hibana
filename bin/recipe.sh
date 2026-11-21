#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache="${SPARK_HIBANA_CACHE:-$root/.cache}"
if [[ "$cache" != /* ]]; then
    printf '%s\n' 'SPARK_HIBANA_CACHE must be an absolute path.' >&2
    exit 2
fi
upstream="$cache/upstream"
revision=d474b74d5a892b2cd1cc8fc50cb17946e2a539bb
image=spark-bench-hibana:d474b74
port=18904

usage() {
    printf 'Usage: %s build | fetch {exact|fast} | serve {exact|fast}\n' "$0" >&2
    exit 2
}

snapshot_path() {
    python3 - "$cache/$1.json" "$2" <<'PY'
import json
import sys
from pathlib import PurePosixPath

with open(sys.argv[1]) as source:
    entry = json.load(source)
path = PurePosixPath(entry["snapshot"])
if entry["pin"] != sys.argv[2] or path.is_absolute() or ".." in path.parts:
    raise SystemExit("Unexpected checkpoint marker. Run the fetch command again.")
print(path)
PY
}

case "${1:-}" in
    build)
        [[ $# == 1 ]] || usage
        if [[ ! -d "$upstream/.git" ]]; then
            mkdir -p "$upstream"
            git -C "$upstream" init -q
        fi
        if ! git -C "$upstream" cat-file -e "$revision^{commit}" 2>/dev/null; then
            git -C "$upstream" fetch --depth=1 https://github.com/sharar-project/hibana.git "$revision"
        fi
        git -C "$upstream" checkout --detach "$revision"
        if [[ -n "$(git -C "$upstream" status --porcelain)" ]]; then
            printf '%s\n' 'The cached Hibana checkout has local changes. Use a clean cache directory.' >&2
            exit 1
        fi
        docker build --platform linux/arm64 -f "$root/Dockerfile" -t "$image" "$upstream"
        ;;
    fetch)
        [[ $# == 2 && ( "$2" == exact || "$2" == fast ) ]] || usage
        mkdir -p "$cache/hf"
        docker run --rm --cap-drop ALL --security-opt no-new-privileges \
            --user "$(id -u):$(id -g)" --env HOME=/cache --env HF_HOME=/cache/hf \
            --env HF_HUB_DISABLE_IMPLICIT_TOKEN=1 --env HF_HUB_DOWNLOAD_TIMEOUT=90 \
            --env HF_XET_NUM_CONCURRENT_RANGE_GETS=8 \
            --volume "$cache:/cache" --volume "$root/bin/fetch.py:/recipe/fetch.py:ro" \
            --entrypoint /opt/hibana/.venv/bin/python "$image" -u /recipe/fetch.py "$2"
        ;;
    serve)
        [[ $# == 2 && ( "$2" == exact || "$2" == fast ) ]] || usage
        profile="$2"
        target="$(snapshot_path target 7b719225242aacd3dbd3f9407468c2ee9a9d2594)"
        test -f "$cache/hf/$target/config.json"
        profile_env=(-e HIBANA_SSD=0)
        if [[ "$profile" == fast ]]; then
            w3="$(snapshot_path w3 5c4c62a62853caebea8cedda7fbed952efa1effe)"
            test -f "$cache/hf/$w3/manifest.json"
            profile_env+=(-e "HIBANA_W3_CHECKPOINT=/hf/$w3")
        fi
        state="$cache/state/$profile"
        mkdir -p "$state" "$cache/kernels"
        touch "$state/empty.conf"
        if [[ -s "$state/empty.conf" ]]; then
            printf '%s\n' 'The benchmark config file must be empty.' >&2
            exit 1
        fi
        exec docker run --rm --name qwen38-hibana-flash-next --gpus all --network host --shm-size=2g \
            --cap-drop ALL --security-opt no-new-privileges --user "$(id -u):$(id -g)" \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -e HOME=/state -e USER=benchmark -e LOGNAME=benchmark -e XDG_CACHE_HOME=/kernels \
            -e TORCH_EXTENSIONS_DIR=/kernels/torch -e TRITON_CACHE_DIR=/kernels/triton \
            -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
            -e HIBANA_HF_CACHE=/hf -e HIBANA_FLASHNEXT_CACHE=/hf -e HIBANA_RESULTS=/state \
            -e HIBANA_CONFIG_FILE=/state/empty.conf -e HIBANA_HOST_FLOOR_GIB=12 \
            "${profile_env[@]}" -v "$cache/hf:/hf:ro" -v "$cache/kernels:/kernels:rw" \
            -v "$state:/state:rw" "$image" hibana serve flashnext "--$profile" --no-fetch \
            --host 127.0.0.1 --port "$port" --max-batch 5 --context-length 262144
        ;;
    *) usage ;;
esac
