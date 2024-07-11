#!/bin/bash

set -e

. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wri_tmlr_cmnist

pushd DomainBed/domainbed/scripts
./run_sweep.sh
./run_get_results.sh
popd


