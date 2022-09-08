#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

python "${SCRIPT_DIR}/rewrite_configuration.py" "${1}"

CONFIG_PATH="${PWD}/config.yaml.tmp"
DATA_ROOT=$(cat data_directory.tmp)
OUTPUT_ROOT="${PWD}/output/"

[ -d "${OUTPUT_ROOT}". ] || mkdir "${OUTPUT_ROOT}"

if [ -z "${2}" ]; then
  IMAGE_NAME="anovos-worker"
else
  if [ -z "${3}" ]; then
    IMAGE_NAME="anovos-worker-${2}"
  else
    IMAGE_NAME="anovos-worker-${2}:${3}"
  fi
fi

docker run \
  --mount type=bind,source="${CONFIG_PATH}",target=/config.yaml \
  --mount type=bind,source="${DATA_ROOT}",target=/data \
  --mount type=bind,source="${OUTPUT_ROOT}",target=/output \
  "${IMAGE_NAME}"
