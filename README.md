# Weighted Risk Invariance: Domain Generalization under Invariant Feature Shift

Code repository for the paper [Weighted Risk Invariance: Domain Generalization under Invariant Feature Shift](https://openreview.net/forum?id=WyPKLWPYsr).

<p align="center">
  <img src="https://github.com/ginawong/weighted_risk_invariance/blob/main/images/WRI_2d_figure.png?raw=true" width="800"/>
</p>

Learning features that are conditionally invariant across environments is a popular approach for domain generalization, but this can be complicated when the invariant features are not aligned between environments. With weighted risk invariance (WRI), we impose risk invariance on training examples that are reweighted to account for a shift in the invariant features. With appropriate reweighting, WRI allows us to perform invariant learning even under shift in the invariant features.

## Repository structure
This repository contains the source code for recreating the experiments in our paper. Each set of experiments corresponds to a subdirectory. Specifically,
* Results on simulated data can be recreated using `multidim_simulation` and `twodim_simulation`
* HCMNIST and HCMNIST-CS results can be recreated from `CMNIST`
* DomainBed results can be recreated from `DomainBed_WRI`

Each subdirectory contains its own README, with instructions on how to set up and run the code.

## Support
For questions or comments, please file a Github issue or tag [@ginawong](https://github.com/ginawong)
