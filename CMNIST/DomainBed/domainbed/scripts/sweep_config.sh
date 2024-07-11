#!/usr/bin/env bash

ulimit -n 8192

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wri_tmlr_cmnist

DATADIR="$SCRIPT_DIR/../../domainbed_data"
RESULTS_DIR="$SCRIPT_DIR/../../../results"

export PYTHONPATH=$SCRIPT_DIR/../..
cd $SCRIPT_DIR/../..

N_HPARAMS=10
N_TRIALS=5


