#!/bin/bash

. sweep_config.sh

DATASETS=("VLCS" "PACS" "OfficeHome" "TerraIncognita" "DomainNet")
ALGORITHMS=("ERM" "IRM" "GroupDRO" "Mixup" "MLDG" "CORAL" "VREx" "WRI")

N_HPARAMS=10
N_TRIALS=5

mkdir -p "${RESULTS_DIR}"

python -m domainbed.scripts.sweep delete_incomplete \
       --data_dir="${DATADIR}" \
       --output_dir="${RESULTS_DIR}" \
       --command_launcher "multi_gpu" \
       --algorithms "${ALGORITHMS[@]}" \
       --datasets "${DATASETS[@]}" \
       --n_hparams ${N_HPARAMS} \
       --n_trials ${N_TRIALS} \
       --single_test_envs || exit

python -m domainbed.scripts.sweep launch \
    --data_dir="${DATADIR}" \
    --output_dir="${RESULTS_DIR}" \
    --command_launcher "multi_gpu" \
    --algorithms "${ALGORITHMS[@]}" \
    --datasets "${DATASETS[@]}" \
    --n_hparams ${N_HPARAMS} \
    --n_trials ${N_TRIALS} \
    --single_test_envs \
    --skip_confirmation

