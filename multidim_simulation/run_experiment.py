import random
from argparse import Namespace
from typing import List, Tuple, Dict, Any, Iterator, TypeVar, Optional, Callable, Container
from pathlib import Path
import json
from hashlib import md5
from collections import Counter

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from data_model import Environment, Invariant, DataModel, random_covariance, MVNormal, Normal
from density_models import GroundTruthDensityModel, GMMDensityModel, NNDensityModel
from multidensity_model import MultiDensityModel, OptimizedMultiDensityModel
from classifier import Classifier, TrueInvariantClassifier
from algorithms import Algorithm, ERM, WRI, VREx, IRM
from hparams import get_hyperparams
from dataloader import FixedLengthDataLoader, LoaderDictIter
from metrics import Metrics
from logger import init_logging
import logging
from command_line_args import parse_command_line, NO_HASH_ARGS


T = TypeVar('T')


def init_datasets(args: Namespace) -> (List[DataModel], List[Tuple[TensorDataset, TensorDataset]]):
    num_envs = args.num_envs
    r_dims = args.r_dims
    s_dims = args.s_dims
    num_classes = args.num_classes
    samples_per_env = args.samples_per_env
    train_percentage = args.train_percentage
    seed = args.data_seed

    assert 0 <= train_percentage <= 1
    assert samples_per_env > 0

    rng = np.random.RandomState(seed)

    if args.data_parity_swap:
        env_parity = np.ones(num_envs)
        env_parity[args.test_env] = -1

        assert 0 <= args.parity_cone_angle <= 90
        cone_angle = np.radians(args.parity_cone_angle)

        parity_normals = rng.normal(loc=0, scale=1, size=(num_classes, s_dims))
        parity_normals = parity_normals / np.linalg.norm(parity_normals, axis=1, keepdims=True)
        mean_s_cond_y = rng.normal(scale=args.spurious_mean_std, size=(num_envs, num_classes, s_dims,))
        current_parity = np.sign(np.sum(mean_s_cond_y * parity_normals, axis=2))
        # flip any negative parity points so all are positive
        mean_s_cond_y = current_parity[..., None] * mean_s_cond_y
        # rotate means towards cone axis to ensure all are within the cone
        for env in range(num_envs):
            for cls in range(num_classes):
                current = mean_s_cond_y[env][cls]
                ref = parity_normals[cls]
                proj = ref * np.dot(ref, current)
                ang = np.arccos(np.dot(ref, current / np.linalg.norm(current)))
                assert ang <= np.pi / 2
                orth = current - proj
                new_ang = (ang / (np.pi / 2)) * cone_angle
                new_vec = np.linalg.norm(current) * (np.cos(new_ang) * (proj / np.linalg.norm(proj)) + np.sin(new_ang) * (orth / np.linalg.norm(orth)))
                mean_s_cond_y[env][cls] = new_vec
        # apply desired parity
        mean_s_cond_y = mean_s_cond_y * env_parity[:, None, None]

        environments = [
            Environment(
                dist_r=MVNormal(rng.normal(scale=args.invariant_mean_std, size=(r_dims,)),
                                random_covariance(r_dims, 1.0 - args.invariant_std_delta, 1.0 + args.invariant_std_delta, rng)),
                dist_s_cond_y=[
                    MVNormal(mean_s_cond_y[env, cls], random_covariance(s_dims, 0.5, 1.5, rng))
                    for cls in range(args.num_classes)
                ])
            for env in range(num_envs)]
    else:
        environments = [
            Environment(
                dist_r=MVNormal(rng.normal(scale=args.invariant_mean_std, size=(r_dims,)),
                                random_covariance(r_dims, 1.0 - args.invariant_std_delta, 1.0 + args.invariant_std_delta, rng)),
                dist_s_cond_y=[
                    MVNormal(rng.normal(scale=args.spurious_mean_std, size=(s_dims,)), random_covariance(s_dims, 0.5, 1.5, rng))
                    for _ in range(args.num_classes)
                ])
            for _ in range(num_envs)]

    invariant = Invariant(rng.standard_normal((num_classes, r_dims,)), Normal(0.0, args.invariant_noise_std))
    data_model = DataModel(environments, invariant, rng=rng)


    datasets = []
    for env in range(data_model.num_envs):
        x, y = data_model.sample(env, samples_per_env)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_percentage, random_state=rng)
        train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
        test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
        datasets.append((train_dataset, test_dataset))

    return data_model, datasets


def init_ground_truth_densities(data_model: DataModel, test_env: int) -> MultiDensityModel:
    assert test_env in range(data_model.num_envs)
    density_models = dict()
    for env_idx, environment in enumerate(data_model.environments):
        if env_idx != test_env:
            density_models[env_idx] = GroundTruthDensityModel(environment)
    return MultiDensityModel(density_models)


def init_gmm_densities(num_envs: int, test_env: int, hparams: Dict[str, Any])\
    -> MultiDensityModel:
    density_models = dict()
    for env in range(num_envs):
        if env == test_env:
            continue
        density_models[env] = GMMDensityModel(
            n_components=hparams['gmm_n_components'],
            covariance_type=hparams['gmm_covariance_type'],
            reg_covar=hparams['gmm_reg_covar'],
            max_iter=hparams['gmm_max_iter'],
            singularity_threshold=hparams['gmm_singularity_threshold'],
            num_workers=hparams['gmm_num_workers']
        )
    return MultiDensityModel(density_models)


def init_model_based_densities(num_envs: int, test_env: int, num_dimensions: int, hparams: Dict[str, Any])\
    -> MultiDensityModel:
    density_models = dict()
    for env in range(num_envs):
        if env == test_env:
            continue
        density_models[env] = NNDensityModel(
            num_dimensions,
            min_density=hparams['mbd_min_density'],
            max_density=hparams['mbd_max_density'],
        )
    multidensity_model = OptimizedMultiDensityModel(
        density_models,
        lr=hparams["mbd_lr"],
        optimizer=hparams["mbd_optimizer"],
        weight_decay=hparams["mbd_weight_decay"],
        grad_clip=hparams["mbd_grad_clip"],
        batch_size=hparams["mbd_batch_size"],
        wri_lambda=hparams["mbd_wri_lambda"],
        ent_lambda=hparams["mbd_ent_lambda"],
        fit_steps=hparams["mbd_fit_steps"],
    )
    return multidensity_model


def init_density_model(data_model: DataModel, args: Namespace, hparams: Dict[str, Any])\
        -> MultiDensityModel:
    if hparams['density_model'] == 'ground_truth':
        density_models = init_ground_truth_densities(data_model, args.test_env)
    elif hparams['density_model'] == 'gmm':
        density_models = init_gmm_densities(args.num_envs, args.test_env, hparams)
    elif hparams['density_model'] == 'model_based':
        density_models = init_model_based_densities(args.num_envs, args.test_env, args.r_dims + args.s_dims, hparams)
    else:
        raise Exception()

    return density_models


def init_algorithm(classifier: Classifier, args: Namespace, hparams: Dict[str, Any]) -> Algorithm:
    if args.algorithm == 'erm':
        model = ERM(classifier, args.num_envs - 1, hparams)
    elif args.algorithm == 'wri':
        model = WRI(classifier, args.num_envs - 1, hparams)
    elif args.algorithm == 'vrex':
        model = VREx(classifier, args.num_envs - 1, hparams)
    elif args.algorithm == 'irm':
        model = IRM(classifier, args.num_envs - 1, hparams)
    else:
        raise NotImplementedError
    return model


def init_classifier(args: Namespace, hparams: Dict[str, Any]) -> Classifier:
    if args.true_invariant_classifier:
        classifier = TrueInvariantClassifier(args.r_dims + args.s_dims, args.num_classes, args.r_dims)
    else:
        classifier = Classifier(args.r_dims + args.s_dims, args.num_classes)
    return classifier


def init_optimizer(algorithm: Algorithm, hparams: Dict[str, Any]):
    params = algorithm.parameters()
    if hparams['optimizer'] == 'adam':
        optimizer = optim.Adam(
            params,
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay'])
    elif hparams['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay'])
    else:
        raise RuntimeError

    return optimizer


def init_dataloaders(datasets: List[Tuple[Dataset, Dataset]], args: Namespace, hparams: Dict[str, Any]) \
        -> (Dict[int, DataLoader], Dict[str, DataLoader]):
    assert args.test_env in range(len(datasets))

    train_loaders = dict()
    train_datasets = dict()
    eval_loaders = dict()
    for env_idx, (left_dataset, right_dataset) in enumerate(datasets):
        if env_idx != args.test_env:
            left_name = f"train_{env_idx}"
            right_name = f"val_{env_idx}"
            train_datasets[env_idx] = left_dataset
            train_loaders[env_idx] = FixedLengthDataLoader(
                left_dataset,
                hparams['steps'],
                hparams['batch_size'],
                num_workers=args.loader_workers,
            )
        else:
            left_name = "test"
            right_name = "val"

        eval_loaders[left_name] = DataLoader(
            left_dataset,
            shuffle=False,
            batch_size=hparams['batch_size'],
            num_workers=args.loader_workers,
        )
        eval_loaders[right_name] = DataLoader(
            right_dataset,
            shuffle=False,
            batch_size=hparams['batch_size'],
            num_workers=args.loader_workers,
        )

    return train_loaders, train_datasets, eval_loaders


def init_train_iterators(train_loaders: Dict[int, DataLoader]) -> Dict[int, Iterator[List[Any]]]:
    return {env: iter(loader) for env, loader in train_loaders.items()}


def init_metrics(args: Namespace, hparams: Dict[str, Any]) -> Metrics:
    metrics_out_path = Path(args.results_dir) / "metrics.json"
    return Metrics(hparams["steps_per_validation"], hparams["steps"], metrics_out_path, logger=logging.getLogger())


def refit_density_models(density_model: MultiDensityModel, datasets: Dict[int, TensorDataset], algorithm: Algorithm):
    if not algorithm.uses_density:
        return
    density_model.refit(datasets, algorithm)


def mark_complete(args: Namespace):
    with open(Path(args.results_dir) / "__complete__", 'w') as f:
        f.write("experiment complete")


def run_experiment(data_model: DataModel, datasets: List[Tuple[Dataset, Dataset]], args: Namespace,
                   hparams: Dict[str, Any]):
    seed_rng(args.seed)
    classifier = init_classifier(args, hparams)
    algorithm = init_algorithm(classifier, args, hparams)

    # create density model
    density_models = init_density_model(data_model, args, hparams)
    optimizer = init_optimizer(algorithm, hparams)
    train_loaders, train_datasets, eval_loaders = init_dataloaders(datasets, args, hparams)
    metrics = init_metrics(args, hparams)

    for step, train_batches in enumerate(LoaderDictIter(train_loaders)):
        train_loss = algorithm.compute_loss(train_batches, density_models)

        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(algorithm.parameters(), hparams['grad_clip'])
        optimizer.step()

        metrics.step(algorithm, eval_loaders, density_models, step)
        metrics.log_last_step()

        if step > 0 and step % hparams['steps_per_density_update'] == 0:
            refit_density_models(density_models, train_datasets, algorithm)
            algorithm.reset_density_stats()

    mark_complete(args)


def seed_rng(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def sort_dict_by_keys(d: Dict[T, Any], key: Optional[Callable[[T], bool]] = None) -> Dict[T, Any]:
    return {k: d[k] for k in sorted(d.keys(), key=key)}


def get_parameters_hash(args: Namespace, hparams: Dict[str, Any], extra_no_hash_args: Optional[Container]=()) -> str:
    hash_dict = dict(
        args=sort_dict_by_keys({k: v for k, v in vars(args).items() if k not in NO_HASH_ARGS and k not in extra_no_hash_args}),
        hparams=sort_dict_by_keys(hparams),
    )
    return md5(json.dumps(hash_dict).encode("utf-8")).hexdigest()


def init_results_dir(args: Namespace, hparams: Dict[str, Any]):
    results_dir = Path(args.results_dir)
    if args.use_hash_subdir:
        hash = get_parameters_hash(args, hparams)
        results_dir = results_dir / hash
    results_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir = str(results_dir)
    return results_dir


def save_parameters(args: Namespace, hparams: Dict[str, Any]):
    results_dir = Path(args.results_dir)
    parameters = dict(
        args=sort_dict_by_keys(vars(args)),
        hparams=sort_dict_by_keys(hparams))
    with open(results_dir / "parameters.json", 'w') as fout:
        json.dump(parameters, fout, indent=4)

    logging.info("Program Parameters")
    for line in json.dumps(parameters, indent=4).split('\n'):
        logging.info(line)


def log_datasets(data_model: DataModel, datasets: List[Tuple[TensorDataset, ...]]):
    logging.info(f"Data Model")
    logging.info(f"    num envs: {data_model.num_envs}")
    logging.info(f"    num classes: {data_model.num_classes}")
    logging.info(f"    data dims: {data_model.num_data_dims} ({data_model.num_invariant_dims} invariant + {data_model.num_spurious_dims} spurious")
    for env, splits in enumerate(datasets):
        logging.info(f"    env {env}: {':'.join(str(len(dataset)) for dataset in splits)}")
        for split_idx, dataset in enumerate(splits):
            num_samples = len(dataset)
            class_counts = Counter(dataset.tensors[1].tolist())
            class_counts = [class_counts.get(c, 0) for c in range(data_model.num_classes)]
            class_count_str = " / ".join(f"{n}({n/num_samples * 100:0.1f}%)" for n in class_counts)
            logging.info(f"          split {split_idx} samples/class = {class_count_str}")


def max_class_ratio(data_model: DataModel, datasets: List[Tuple[TensorDataset, ...]]) -> float:
    max_ratio = 0
    for env, splits in enumerate(datasets):
        class_counts = dict()
        num_samples = 0
        for dataset in splits:
            num_samples += len(dataset)
            class_counts.update(
                {c: class_counts.get(c, 0) + n for c, n in Counter(dataset.tensors[1].tolist()).items()})
        class_ratios = [class_counts.get(c, 0) / num_samples for c in range(data_model.num_classes)]
        max_ratio = max(max_ratio, max(class_ratios))
    return max_ratio


def main():
    args = parse_command_line()
    hparams = get_hyperparams(args)
    init_results_dir(args, hparams)
    init_logging(args.results_dir, "MDS")

    data_model, datasets = init_datasets(args)

    save_parameters(args, hparams)
    log_datasets(data_model, datasets)

    run_experiment(data_model, datasets, args, hparams)


if __name__ == "__main__":
    main()