from typing import Optional, Container
from argparse import Namespace
import numpy as np


def get_hyperparams(args: Namespace, vary_only: Optional[Container[str]] = None):
    hparams = dict()
    rng = np.random.RandomState(args.hparams_seed)

    def add_random_hparam(name, default_val, random_val_fn):
        assert name not in hparams
        use_default = (args.hparams_seed == 0) or (vary_only is None) or (name not in vary_only)
        hparams[name] = default_val if use_default else random_val_fn(rng)

    # dataset info
    hparams['loader_workers'] = 0

    # algorithm params
    if args.algorithm == 'erm':
        pass
    elif args.algorithm == 'wri':
        add_random_hparam('wri_lambda', 10, lambda r: 10 ** float(r.uniform(-1, 2)))
    elif args.algorithm == 'vrex':
        add_random_hparam('vrex_lambda', 10, lambda r: 10 ** float(r.uniform(-1, 5)))
    elif args.algorithm == "irm":
        add_random_hparam('irm_lambda', 10, lambda r: 10 ** float(r.uniform(-1, 5)))
    else:
        raise NotImplementedError

    # optimization params
    hparams['optimizer'] = 'adam'
    if hparams['optimizer'] == 'adam':
        pass
    elif hparams['optimizer'] == 'sgd':
        pass
    add_random_hparam('batch_size', 2048, lambda r: int(round(2 ** r.uniform(6, 12))))
    add_random_hparam('grad_clip', 10, lambda r: float(10 ** r.uniform(-1, 1)))
    add_random_hparam('lr', 0.04, lambda r: float(10 ** r.uniform(-2, -1)))
    add_random_hparam('weight_decay', 1e-5, lambda r: float(10 ** r.uniform(-6, -1)))
    hparams['steps'] = 100 # max(50, 50 * int(round(2048 / hparams["batch_size"])))
    hparams['steps_per_validation'] = 1
    add_random_hparam(
        'steps_per_density_update', 1, # max(3, int(round(3 * 2048 / hparams["batch_size"]))),
        lambda r: int(round(max(1, r.uniform(0, 4) * 2048 / hparams["batch_size"])))
    )

    # density model params
    # hparams['density_model'] = 'ground_truth'
    # hparams['density_model'] = 'gmm'
    hparams['density_model'] = 'model_based'
    if hparams['density_model'] == 'ground_truth':
        pass
    elif hparams['density_model'] == 'gmm':
        add_random_hparam('gmm_n_components', 3, lambda r: int(r.choice(range(1, 11))))
        hparams['gmm_covariance_type'] = 'full'
        add_random_hparam('gmm_reg_covar', 1e-1, lambda r: float(10 ** r.uniform(0, -5)))
        hparams['gmm_max_iter'] = 2000
        hparams['gmm_singularity_threshold'] = 1e10
        hparams['gmm_num_workers'] = 12
    elif hparams['density_model'] == 'model_based':
        add_random_hparam("mbd_lr", 0.04, lambda r: float(10 ** r.uniform(-2, -1)))
        hparams["mbd_optimizer"] = hparams["optimizer"]
        add_random_hparam('mbd_weight_decay', 1e-5, lambda r: float(10 ** r.uniform(-6, -1)))
        add_random_hparam('mbd_grad_clip', 10, lambda r: float(10 ** r.uniform(-1, 1)))
        add_random_hparam('mbd_batch_size', 64, lambda r: int(2 ** r.choice(range(5, 12))))
        add_random_hparam('mbd_wri_lambda', 50, lambda r: float(10 ** r.uniform(-1, 2)))
        add_random_hparam('mbd_ent_lambda', 0.1, lambda r: float(10 ** r.uniform(-2, 2)))
        add_random_hparam('mbd_fit_steps', 2, lambda r: int(r.choice([1, 2, 4, 8, 16, 32])))
        add_random_hparam('mbd_min_density', 0.1, lambda r: float(10 ** r.uniform(-1, 0.5)))
        add_random_hparam('mbd_max_density', 1.0, lambda r: max(2 * hparams["mbd_min_density"], float(10 ** r.uniform(-1, 2))))
    else:
        raise NotImplementedError

    return hparams
