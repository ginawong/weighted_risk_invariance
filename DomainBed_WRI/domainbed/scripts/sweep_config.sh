#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DATADIR="$SCRIPT_DIR/../../domainbed_data"
RESULTS_DIR="$SCRIPT_DIR/../../results"

. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dbed

export PYTHONPATH=$SCRIPT_DIR/../..
cd $SCRIPT_DIR/../..

N_HPARAMS=10
N_TRIALS=5

