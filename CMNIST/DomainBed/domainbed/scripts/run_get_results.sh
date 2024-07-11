#!/usr/bin/env bash

. sweep_config.sh

set -x

echo $PYTHONPATH

#python -m domainbed.scripts.collect_results \
#    --input_dir="${RESULTS_DIR}" --latex || exit

python -m domainbed.scripts.collect_results \
    --input_dir="${RESULTS_DIR}" || exit
