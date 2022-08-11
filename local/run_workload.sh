#!/usr/bin/env bash

python rewrite_configuration.py "${1}"

CONFIG_PATH="${PWD}/config.yaml.tmp"
DATA_ROOT=$(cat data_directory.tmp)
OUTPUT_ROOT="${PWD}/output/"

[ -d "${OUTPUT_ROOT}". ] || mkdir "${OUTPUT_ROOT}"

docker run anovos-worker \
       --mount type=bind,source="${CONFIG_PATH}",target=/config.yaml \
       --mount type=bind,source="${DATA_ROOT}",target=/data \
       --mount type=bind,source="${OUTPUT_ROOT}",target=/output
