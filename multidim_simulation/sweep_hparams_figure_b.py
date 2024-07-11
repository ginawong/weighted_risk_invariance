import itertools
from functools import partial
from os import PathLike
import copy
import json
import argparse
from argparse import Namespace
import numpy as np
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from run_experiment import get_hyperparams, init_results_dir, init_logging, init_datasets, run_experiment,\
    save_parameters, log_datasets, get_parameters_hash, max_class_ratio
import logging
from typing import Optional, List, Dict, Any, Union, Tuple
from command_line_args import add_main_args
import matplotlib.pyplot as plt


def random_data_args(data_seed, invariant_std_delta: float):
    return dict(
        data_seed=data_seed,
        invariant_mean_std=0.0,
        invariant_std_delta=invariant_std_delta,
        invariant_noise_std=0.5,
        spurious_mean_std=1.0,
    )


def locate_data_seeds(args: Namespace, num_seeds: int, invariant_std_delta: float) -> Tuple[List[int], Dict[str, Any]]:
    assert num_seeds > 0
    args = copy.deepcopy(args)

    args.hparams_seed = 0

    variations = {
        'erm': dict(algorithm='erm', true_invariant_classifier=False),
        'erm_cheat': dict(algorithm='erm', true_invariant_classifier=True),
    }

    base_results_dir = Path(args.results_dir) / "dataset_search"

    class StopSweep(BaseException):
        def __init__(self, msg):
            self.msg = msg

    sweep_results = defaultdict(dict)
    data_seeds = []
    data_seed = -1
    while len(data_seeds) < num_seeds:
        data_seed += 1
        seed_results_dir = base_results_dir / f"delta={invariant_std_delta}_ds={data_seed}"
        for k, v in random_data_args(data_seed, invariant_std_delta).items():
            setattr(args, k, v)

        rejected_path = seed_results_dir / "__rejected__"
        try:
            if rejected_path.exists():
                with open(rejected_path, "r") as f:
                    raise StopSweep(f.read())

            agg_results = dict()
            for variation_name, variation_args in variations.items():
                for k, v in variation_args.items():
                    setattr(args, k, v)

                results_dir = seed_results_dir / f"{variation_name}"
                if not (results_dir / "__complete__").exists():
                    args.results_dir = str(results_dir)

                    hparams = get_hyperparams(args)
                    init_results_dir(args, hparams)
                    init_logging(args.results_dir, "MDS")
                    logging.getLogger().setLevel(logging.ERROR)

                    data_model, datasets = init_datasets(args)

                    save_parameters(args, hparams)
                    log_datasets(data_model, datasets)

                    # ensure the data is balanced
                    max_ratio = max_class_ratio(data_model, datasets)
                    if max_ratio > 1.5 / args.num_classes:
                        raise StopSweep(f"max_ratio({max_ratio}) > {1.5 / args.num_classes}")

                    run_experiment(data_model, datasets, args, hparams)
                with open(results_dir / "metrics.json", "r") as f:
                    results = json.load(f)

                max_acc = max(results["test_acc"].values())
                max_step = int(max(results["test_acc"], key=results["test_acc"].__getitem__))

                # heuristics for finding good data
                if max_acc > 0.9:
                    # too easy
                    raise StopSweep(f"max_acc({max_acc}) > 0.9")
                elif max_acc < 1.5 / args.num_classes:
                    # too hard
                    raise StopSweep(f"max_acc({max_acc}) < {1.5 / args.num_classes}")
                elif max_step < 3:
                    # too weird
                    raise StopSweep(f"max_step({max_step}) < 3")

                agg_results[variation_name] = dict(acc=max_acc, step=max_step)

            erm_acc = agg_results["erm"]["acc"]
            cheat_acc = agg_results["erm_cheat"]["acc"]
            if cheat_acc < erm_acc:
                # really weird if this happens
                raise StopSweep(f"cheat_acc({cheat_acc}) < erm_acc({erm_acc})")
            elif (cheat_acc - erm_acc) / cheat_acc < 0.1:
                raise StopSweep(f"Margin for improvement too low (erm_acc={erm_acc}, cheat_acc={cheat_acc})")

        except StopSweep as err:
            if not rejected_path.exists():
                rejected_path.parent.mkdir(parents=True, exist_ok=True)
                with open(rejected_path, "w") as f:
                    f.write(err.msg)
            print(f"data seed {data_seed} rejected ({err.msg})")
            continue

        # if we made it here then data seed is good. Save it
        print(f"!!!data seed {data_seed} KEPT - {agg_results}")
        sweep_results["erm"][f"data_seed_{data_seed}"] = agg_results["erm"]
        sweep_results["erm_cheat"][f"data_seed_{data_seed}"] = agg_results["erm_cheat"]
        data_seeds.append(data_seed)

    return data_seeds, sweep_results


def run(args, vary_keys: Optional[List[str]] = None, hparams_override: Optional[Dict[str, Any]] = None)\
        -> (Dict[str, Dict[str, Any]], Dict[str, Any]):
    args = copy.deepcopy(args)

    if not (Path(args.results_dir) / "__complete__").exists():
        hparams = get_hyperparams(args, vary_only=vary_keys)
        if hparams_override is not None:
            hparams.update(hparams_override)

        init_results_dir(args, hparams)
        init_logging(args.results_dir, "MDS")
        logging.getLogger().setLevel(logging.ERROR)

        data_model, datasets = init_datasets(args)

        save_parameters(args, hparams)
        log_datasets(data_model, datasets)

        run_experiment(data_model, datasets, args, hparams)
    else:
        with open(Path(args.results_dir) / "parameters.json") as f:
            hparams = json.load(f)["hparams"]

    with open(Path(args.results_dir) / 'metrics.json', 'r') as f:
        results = json.load(f)

    return results, hparams


def get_metrics(results: Dict[Any, Any],
                hparams: Dict[str, Any], hparams_baseline: Dict[str, Any],
                args: Namespace, args_baseline: Namespace)\
        -> Dict[str, Dict[str, Any]]:

    metrics = dict()
    hparams_delta = {k: v for k, v in hparams.items() if k not in hparams_baseline or v != hparams_baseline[k]}
    args, args_baseline = vars(args), vars(args_baseline)
    args_delta = {k: v for k, v in args.items() if k not in args or v != args_baseline[k]}
    metrics["hparams_values"] = hparams_delta
    metrics["args_values"] = args_delta
    best_step = max(results["test_acc"], key=lambda k: results["val_acc"][k])
    metrics["acc"] = results["test_acc"][best_step]
    metrics["step"] = int(best_step)
    return metrics


def process_results(metrics, output_dir: Union[str, PathLike]):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    args_baseline = metrics["args"]
    hparams_baseline = metrics["hparams"]

    get_arg = lambda r, k: r["args_values"].get(k, args_baseline[k])
    get_hparam = lambda r, k: r["hparams_values"].get(k, hparams_baseline[k])

    ROW_ORDER = ["Oracle", "ERM", "VREx", "IRM", "WRI-gt", "WRI"]

    entries = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for exp in metrics["experiments"]:
        results = metrics["experiments"][exp]
        alg = get_arg(results, "algorithm")
        spread = get_arg(results, 'invariant_std_delta')
        seed = get_arg(results, "seed")
        data_seed = get_arg(results, "data_seed")
        hparams_seed = get_arg(results, "hparams_seed")

        gt_density = get_hparam(results, "density_model") == "ground_truth"
        oracle = get_arg(results, "true_invariant_classifier")

        if oracle:
            assert alg == 'erm'
            alg = "Oracle"
        elif gt_density:
            assert alg == 'wri'
            alg = "WRI-gt"
        elif alg == "wri":
            alg = "WRI"
        elif alg == "erm":
            alg = "ERM"
        elif alg == "vrex":
            alg = "VREx"
        elif alg == "irm":
            alg = "IRM"
        else:
            raise NotImplementedError

        entries[(alg, spread)][seed][data_seed][hparams_seed] = results["acc"]

    algs, spreads = zip(*entries.keys())
    # sort rows by ROW_ORDER, values not in ROW_ORDER come after
    algs = sorted(list(set(algs)), key=lambda a: (0, ROW_ORDER.index(a)) if a in ROW_ORDER else (1, a))
    spreads = list(map(str, sorted(list(set(spreads)))))

    df = pd.DataFrame(columns=["Alg"] + spreads)
    df.set_index("Alg")

    table = {alg: {spread: "" for spread in spreads} for alg in algs}

    # process entries
    for alg, spread in entries:
        k = (alg, spread)
        for seed in entries[k]:
            for data_seed in entries[k][seed]:
                entries[k][seed][data_seed] = max(entries[k][seed][data_seed].values())
            entries[k][seed] = float(np.mean(list(entries[k][seed].values())))
        mean = float(np.mean(list(entries[k].values())))
        std = float(np.std(list(entries[k].values())))
        table[alg][str(spread)] = f"{mean * 100:.1f} ± {std * 100:.1f}"

    df = df.from_dict(table, orient='index')
    pd.options.display.width = 0
    print(df)
    print(f"Saving table to {output_dir / 'results.txt'}")
    with open(output_dir / "results.txt", "w") as f:
        f.write(repr(df))
    print(df.to_latex())

def worker(worker_args, args_baseline: Namespace, vary_hparam_keys: List[str]):
    key, experiment = worker_args
    args_override = experiment["args_override"]
    hparams_override = experiment["hparams_override"]

    args = copy.deepcopy(args_baseline)
    for k, v in args_override.items():
        setattr(args, k, v)
    results, hparams = run(args, vary_hparam_keys, hparams_override)

    return dict(results=results, hparams=hparams, args=args, key=key)


def main():
    parser = argparse.ArgumentParser("Sweep hyperparameters")
    sweep_group = parser.add_argument_group("sweep")
    sweep_group.add_argument("--vary_hparam_keys", type=str, nargs='+', default=[
        'mbd_batch_size', 'mbd_wri_lambda', 'mbd_ent_lambda', 'mbd_fit_steps',
    ])
    sweep_group.add_argument("--num_hparams_seeds", type=int, default=0)
    sweep_group.add_argument("--num_data_seeds", type=int, default=5, help="for each set of hparams test on this many datasets")
    sweep_group.add_argument("--num_seeds", type=int, default=5)
    sweep_group.add_argument("--invariant_deltas", nargs="+", type=float,
                             default=[0.0, 0.2, 0.4, 0.6, 0.8])
    sweep_group.add_argument("--reprocess_only", action="store_true")
    experiment_group = parser.add_argument_group("Experiment args")
    add_main_args(experiment_group)

    parser.set_defaults(
        algorithm="wri",
        results_dir="results/sweep_hparams_figure_b",
        samples_per_env=1000)

    args = parser.parse_args()

    hparams_baseline = get_hyperparams(args)
    sweep_hash = get_parameters_hash(args, hparams_baseline, extra_no_hash_args=[a.dest for a in sweep_group._group_actions])
    args.results_dir = str(Path(args.results_dir) / sweep_hash)
    args_baseline = copy.deepcopy(args)

    plot_dir = Path(args_baseline.results_dir) / "plots"

    experiments = dict()
    for invariant_delta in args.invariant_deltas:
        data_seeds, erm_metrics = locate_data_seeds(args, args.num_data_seeds, invariant_delta)

        # seed, hparam_seed
        for seed in range(args.num_seeds):
            for data_seed in data_seeds:
                for hparams_seed in range(args.num_hparams_seeds + 1):
                    args_suffix = f"delta={invariant_delta}_ds={data_seed}_hps={hparams_seed}_seed={seed}"

                    args_override = {"seed": seed, "hparams_seed": hparams_seed}
                    args_override.update(random_data_args(data_seed, invariant_delta))

                    key = f"erm_{args_suffix}"
                    args_override["results_dir"] = str(Path(args.results_dir) / key)
                    args_override["algorithm"] = "erm"
                    args_override["true_invariant_classifier"] = False
                    experiments[key] = {"args_override": copy.deepcopy(args_override), "hparams_override": dict()}

                    key = f"wri_{args_suffix}"
                    args_override["results_dir"] = str(Path(args.results_dir) / key)
                    args_override["algorithm"] = "wri"
                    args_override["true_invariant_classifier"] = False
                    experiments[key] = {"args_override": copy.deepcopy(args_override), "hparams_override": dict()}

                    key = f"vrex_{args_suffix}"
                    args_override["results_dir"] = str(Path(args.results_dir) / key)
                    args_override["algorithm"] = "vrex"
                    args_override["true_invariant_classifier"] = False
                    experiments[key] = {"args_override": copy.deepcopy(args_override), "hparams_override": dict()}

                    key = f"irm_{args_suffix}"
                    args_override["results_dir"] = str(Path(args.results_dir) / key)
                    args_override["algorithm"] = "irm"
                    args_override["true_invariant_classifier"] = False
                    experiments[key] = {"args_override": copy.deepcopy(args_override), "hparams_override": dict()}

    from multiprocessing import Pool
    if args.reprocess_only:
        with open(Path(args_baseline.results_dir) / "results.json", 'r') as f:
            metrics = json.load(f)
    else:
        metrics = dict()
        metrics['args'] = vars(args_baseline)
        metrics['hparams'] = hparams_baseline
        metrics['experiments'] = dict()

        worker_partial = partial(worker, args_baseline=args_baseline, vary_hparam_keys=args.vary_hparam_keys)

        with Pool(processes=3) as pool:
            for output in tqdm(pool.imap_unordered(worker_partial, experiments.items()), desc="Sweeping", total=len(experiments)):
                results = output['results']
                hparams = output['hparams']
                args = output['args']
                key = output['key']
                metrics['experiments'][key] = get_metrics(results, hparams, hparams_baseline, args, args_baseline)
                with open(Path(args_baseline.results_dir) / "results.json", 'w') as f:
                    json.dump(metrics, f, indent=1)

    process_results(metrics, plot_dir)


if __name__ == "__main__":
    main()
