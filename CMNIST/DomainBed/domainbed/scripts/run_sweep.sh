#!/bin/bash

. sweep_config.sh

ALGORITHMS=("ERM" "IRM" "VREx" "ERM_WRI")

set -x
mkdir -p "${RESULTS_DIR}"

# download in 1 thread to avoid file system issues
python -m domainbed.scripts.download_mnist "${DATADIR}"

python -m domainbed.scripts.sweep delete_incomplete \
       --data_dir="${DATADIR}" \
       --output_dir="${RESULTS_DIR}" \
       --command_launcher "multi_gpu" \
       --algorithms "${ALGORITHMS[@]}" \
       --datasets "CMNISTHetero25" "CMNISTHetero25_CovShift65" \
       --n_hparams ${N_HPARAMS} \
       --n_trials ${N_TRIALS} \
       --skip_confirmation \
       --single_test_envs || exit

python -m domainbed.scripts.sweep launch \
       --data_dir="${DATADIR}" \
       --output_dir="${RESULTS_DIR}" \
       --command_launcher "multi_gpu" \
       --algorithms "${ALGORITHMS[@]}" \
       --datasets "CMNISTHetero25" "CMNISTHetero25_CovShift65" \
       --n_hparams ${N_HPARAMS} \
       --n_trials ${N_TRIALS} \
       --single_test_envs || exit

