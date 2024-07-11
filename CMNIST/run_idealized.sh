#!/bin/bash

set -e

. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wri_tmlr_cmnist

pushd DomainBed
python -m domainbed.scripts.idealized_cmnist_exp
popd

