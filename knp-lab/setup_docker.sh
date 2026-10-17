#!/usr/bin/env bash
# Собирает два контейнера с Kaspersky Neuromorphic Platform (KNP), которые нужны нейроморфной части:
#   knp-run — KNP v2.0.0 (gcc 14), CPU-бэкенд + нейрон AltAILIF       -> бэкенд "cpu-v2"
#   knp-lab — KNP v1.0.0 (Ubuntu 22.04) + эмулятор чипа «Алтай» 1.0.0 -> бэкенд "altai-v1"
# Обе машины монтируют эту папку (knp-lab/) как /work.
#
# Запуск из корня репозитория:  bash knp-lab/setup_docker.sh
# Повторный запуск безопасен: готовые клоны, контейнеры и сборки пропускаются.
# Первая сборка на Apple Silicon идёт через эмуляцию amd64 и занимает ~1 час.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
KNP_URL=https://github.com/KasperskyLab/knp
RELEASES="$KNP_URL/releases/download"

# --- исходники KNP: клон тега + наши патчи + наши примеры -----------------------------------------
prepare_repo() {  # <папка> <тег> <патч> <примеры...>
    local dir=$1 tag=$2 patch=$3; shift 3
    if [ ! -d "$dir/.git" ]; then
        git clone --branch "$tag" --depth 1 "$KNP_URL.git" "$dir"
        git -C "$dir" apply --ignore-whitespace "$HERE/patches/$patch"
    fi
    for ex in "$@"; do
        rsync -a "examples/$ex/" "$dir/examples/$ex/"
    done
}
prepare_repo repo    v2.0.0 knp-v2.0.0.patch altai-fault-classifier altai-simple-network matlab-bridge
prepare_repo repo-v1 v1.0.0 knp-v1.0.0.patch altai-fault-classifier

# --- бинарные пакеты бэкенда «Алтай» из релизов KNP ----------------------------------------------
fetch() { [ -f "$1/$2" ] || curl -fL --create-dirs -o "$1/$2" "$RELEASES/$3/$2"; }
fetch pkgs    knp-altai-backend_1.0.2_amd64.deb     v2.0.0
fetch pkgs    knp-altai-backend-dev_1.0.2_amd64.deb v2.0.0
fetch pkgs-v1 knp-altai-backend_1.0.0_amd64.deb     v1.0.0

# --- контейнеры ---------------------------------------------------------------------------------
start() {  # <имя> <образ>
    if ! docker container inspect "$1" >/dev/null 2>&1; then
        docker run -d --name "$1" --platform linux/amd64 -v "$HERE:/work" -w /work "$2" sleep infinity
    else
        docker start "$1" >/dev/null
    fi
}
start knp-run gcc:14.1.0
start knp-lab ubuntu:22.04

# KNP v2.0.0 (CMake из pip: KNP требует >= 3.25, в образах версии старее)
docker exec knp-run bash -ec '
    if [ ! -x repo/build-gcc14/bin/altai-fault-classifier ]; then
        apt-get update && apt-get install -y libboost-all-dev python3-pip
        pip3 install --break-system-packages cmake
        dpkg -i pkgs/knp-altai-backend_1.0.2_amd64.deb pkgs/knp-altai-backend-dev_1.0.2_amd64.deb || apt-get install -fy
        cmake -S repo -B repo/build-gcc14 -DCMAKE_BUILD_TYPE=Release -DKNP_BUILD_TESTS=OFF -DKNP_BUILD_EXAMPLES=ON
        cmake --build repo/build-gcc14 -j"$(nproc)"
    fi'

# KNP v1.0.0 + golden model «Алтая» 1.0.0 (пакет распаковывается в /opt/altai-v1, не ставится)
docker exec knp-lab bash -ec '
    if [ ! -x repo-v1/build/bin/altai-fault-classifier ]; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update && apt-get install -y build-essential git libboost-all-dev python3-pip
        pip3 install cmake
        mkdir -p /opt/altai-v1 && dpkg -x pkgs-v1/knp-altai-backend_1.0.0_amd64.deb /opt/altai-v1
        cmake -S repo-v1 -B repo-v1/build -DCMAKE_BUILD_TYPE=Release -DKNP_BUILD_TESTS=OFF -DKNP_BUILD_EXAMPLES=ON
        cmake --build repo-v1/build -j"$(nproc)"
    fi'

echo "Готово. Проверка: cd analysis/synth && python chip_eval.py cpu-v2"
