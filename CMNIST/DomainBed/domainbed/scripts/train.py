# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Tuple, Optional, Any

if __name__ == "__main__":
    import os
    os.environ["OMP_NUM_THREADS"] = "8"

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from tqdm import tqdm

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import torch.nn as nn

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed import autoencoders
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from domainbed.model_based_density.multidensity_model import OptimizedMultiDensityModel
from domainbed.model_based_density.density_models import NNDensityModel


def print_subheading(subhead):
    print(f'\n---------- {subhead} ----------')


def train_featurizer(train_splits, input_shape, feature_dims, device):
    assert all(weights is None for _, weights in train_splits)

    if input_shape[1:] == (28, 28):
        featurizer, arg_str = autoencoders.train.train_mnist(train_splits, input_shape[0], feature_dims, device, args=args)
    else:
        raise ValueError("Dataset input_shape is not recognized as valid.")

    return featurizer, arg_str


def apply_featurizer(featurizer, splits, device):
    # so the application doesn't need to know about that.
    feature_splits = []
    for img_dataset, weights in splits:
        loader = DataLoader(img_dataset, batch_size=256, shuffle=False, num_workers=2)
        features = []
        labels = []
        for data in tqdm(loader, ncols=0, total=len(loader), desc='Applying featurizer'):
            with torch.no_grad():
                features.append(featurizer(data[0].to(device)).cpu())
            labels.append(data[1])
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        feature_dataset = TensorDataset(features, labels)
        feature_splits.append((feature_dataset, weights))
    return feature_splits


def init_density_models(train_in_splits: List[Tuple[TensorDataset, Optional[np.ndarray]]], input_shape: Tuple[int, ...],
                        min_density, max_density, lr, optimizer, weight_decay, grad_clip, batch_size, wri_lambda,
                        ent_lambda, fit_steps) -> OptimizedMultiDensityModel:
    assert len(input_shape) == 1, "Input should be 1-D features"
    num_dimensions = input_shape[0]
    num_envs = len(train_in_splits)
    density_models = dict()
    for env in range(num_envs):
        density_models[env] = NNDensityModel(num_dimensions, min_density, max_density)
    multidensity_model = OptimizedMultiDensityModel(
        density_models, lr, optimizer,
        weight_decay, grad_clip, batch_size,
        wri_lambda, ent_lambda, fit_steps,
    )
    return multidensity_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--density_method', type=str, default="ModelBased")
    parser.add_argument('--pretrained_featurizer', action='store_true')
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=500,
                        help='Number of steps.')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Checkpoint every N steps.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--featurizer_dims_override', type=int, default=None)
    parser.add_argument('--autoencoder_pretrained', action='store_true')
    parser.add_argument('--autoencoder_arch', type=str, default='resnet18')
    parser.add_argument('--autoencoder_disable_finetune', action='store_true')
    parser.add_argument('--autoencoder_batch_size', type=int, default=64)
    parser.add_argument('--autoencoder_use_ssim', action='store_true')
    parser.add_argument('--autoencoder_use_latent_img', action='store_true')
    parser.add_argument('--save_logits', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    assert args.density_method == "ModelBased"

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.density_method,
                                                   args.pretrained_featurizer, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.density_method,
                                                  args.pretrained_featurizer, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    if args.featurizer_dims_override is not None:
        hparams['pretrained_featurizer_dims'] = args.featurizer_dims_override

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    input_shape = dataset.input_shape

    print('-----------------------------------------------------------------------\n')
    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selection method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discarded at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
                                      int(len(env)*args.holdout_fraction),
                                      misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                                          int(len(in_)*args.uda_holdout_fraction),
                                          misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    if args.pretrained_featurizer:
        original_input_shape = input_shape  # input shape of autoencoder
        input_shape = (hparams['pretrained_featurizer_dims'],)  # feature shape of autoencoder, input shape of model
        hparams['lr'] = hparams['featurized_lr']

        train_in_splits = [(env, env_weights) for i, (env, env_weights) in enumerate(in_splits) if
                           i not in args.test_envs]

        featurizer, arg_str = train_featurizer(train_in_splits, original_input_shape,
                                               hparams['pretrained_featurizer_dims'], device)

        # Load featurizer
        assert len(uda_splits) == 0
        new_in_splits, new_out_splits = [], []
        for (in_, in_weights), (out, out_weights) in zip(in_splits, out_splits):
            is_test = isinstance(in_.underlying_dataset, misc._SplitDataset)
            if is_test:
                underlying_dataset = in_.underlying_dataset.underlying_dataset
            else:
                underlying_dataset = in_.underlying_dataset
            dataset_name = underlying_dataset.name

            cached_name = f'dataset_feats_{arg_str}_{dataset_name}.pt'
            if os.path.exists(cached_name):
                features, labels = torch.load(cached_name)
                feature_dataset = torch.utils.data.TensorDataset(*torch.load(cached_name, map_location='cpu'))
            else:
                (feature_dataset, _), = apply_featurizer(featurizer, [(underlying_dataset, None)], device)
                os.makedirs(os.path.dirname(os.path.abspath(cached_name)), exist_ok=True)
                torch.save(feature_dataset.tensors, cached_name)

            if is_test:
                in_idx = np.array(in_.underlying_dataset.keys)[in_.keys]
            else:
                in_idx = in_.keys
            out_idx = out.keys

            in_feats, in_labels = feature_dataset.tensors[0][in_idx], feature_dataset.tensors[1][in_idx]
            out_feats, out_labels = feature_dataset.tensors[0][out_idx], feature_dataset.tensors[1][out_idx]
            new_in_splits.append((torch.utils.data.TensorDataset(in_feats, in_labels), in_weights))
            new_out_splits.append((torch.utils.data.TensorDataset(out_feats, out_labels), out_weights))
        in_splits = new_in_splits
        out_splits = new_out_splits

    DENSITY_DEPENDENT_ALGS = ["ERM_WRI"]

    density_models: Optional[OptimizedMultiDensityModel] = None
    train_in_splits: Optional[List[Tuple[TensorDataset, Any]]] = None
    if args.algorithm in DENSITY_DEPENDENT_ALGS:
        density_params = {
            'lr': hparams['mbd_lr'],
            'optimizer': hparams['mbd_optimizer'],
            'weight_decay': hparams['mbd_weight_decay'],
            'grad_clip': hparams['mbd_grad_clip'],
            'batch_size': hparams['mbd_batch_size'],
            'wri_lambda': hparams['mbd_wri_lambda'],
            'ent_lambda': hparams['mbd_ent_lambda'],
            'fit_steps': hparams['mbd_fit_steps'],
            'min_density': hparams['mbd_min_density'],
            'max_density': hparams['mbd_max_density'],
        }

        train_in_splits = [(env, env_weights) for i, (env, env_weights) in enumerate(in_splits) if i not in args.test_envs]
        test_in_splits = [(env, env_weights) for i, (env, env_weights) in enumerate(in_splits) if i in args.test_envs]

        print_subheading("Initializing density estimator")
        density_models = init_density_models(train_in_splits, input_shape, **density_params)
        density_models.to(device)

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=0)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=0)
        for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=0,
        shuffle=False)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
                         for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
                          for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
                          for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict(),
        }
        if density_models is not None:
            save_dict["density_state"] = density_models.state_dict()
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None
    print_subheading("Training algorithm")
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        def collate(v):
            if isinstance(v, list) or isinstance(v, tuple):
                return torch.cat([x.reshape(-1, 1) for x in v], dim=1)
            return v
        minibatches_device = [[collate(t).to(device) for t in items]
                              for items in next(train_minibatches_iterator)]
        step_vals = algorithm.update(minibatches_device, density_models)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        if density_models is not None:
            rel_step = step - hparams["wri_penalty_anneal_iters"]
            if rel_step >= 0 and rel_step % hparams["steps_per_density_update"] == 0:
                density_models.refit(train_in_splits, algorithm, device)
                if hasattr(algorithm, "reset_density_stats"):
                    algorithm.reset_density_stats()

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            compute_ece = False
            compute_entropy = True
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                logits_path = os.path.join(args.output_dir, f'logits_step{step}_{name}.pt') if args.save_logits else ''
                metrics = misc.accuracy(algorithm, loader, weights, device, compute_ece, compute_entropy, logits_path)
                acc = metrics.pop(0) if isinstance(metrics, list) else metrics
                results[name + '_acc'] = acc
                if compute_ece:
                    results[name + '_ece'] = metrics.pop(0)

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
