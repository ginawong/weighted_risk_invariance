### Step 1. Setup environment

This will create a conda environment `wri_cmnist` with the packages in `requirements.txt`.

```
bash setup_env.sh
```

### Step 2. Download and unpack pretrained featurizers (autoencoders)

Download `CMNIST_features.tar.gz` from the [Github release](https://github.com/ginawong/weighted_risk_invariance/releases) assets and save it to `CMNIST/DomainBed`. Untar the file with

```
cd CMNIST/DomainBed
tar -xzvf CMNIST_features.tar.gz
```

### Step 3. Run experiments

To generate the results in Table 1, run

```
bash run_experiment.sh
```

To generate the results in Table 2, run

```
bash run_idealized.sh
```

