import json
import os
from os import PathLike

import seaborn as sns
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from typing import Union, List, Any, Optional
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np


sns.set_theme(style='whitegrid', font_scale=1.4)
plt.rc('text', usetex=True)


def process_results(metrics, ax: plt.Axes, sweep_key: str, x_label: str, x_order=List[Any]):
    sns.set_theme(style='whitegrid')

    args_baseline = metrics["args"]
    hparams_baseline = metrics["hparams"]

    get_arg = lambda r, k: r["args_values"].get(k, args_baseline[k])
    get_hparam = lambda r, k: r["hparams_values"].get(k, hparams_baseline[k])

    ROW_ORDER = ["Oracle", "ERM", "VREx", "IRM", "WRI-gt", "WRI"]

    entries = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for exp in metrics["experiments"]:
        results = metrics["experiments"][exp]
        alg = get_arg(results, "algorithm")
        try:
            sweep_val = get_arg(results, sweep_key)
        except KeyError:
            sweep_val = get_hparam(results, sweep_key)
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

        entries[(alg, sweep_val)][seed][data_seed][hparams_seed] = results["acc"]

    algs, spreads = zip(*entries.keys())
    # sort rows by ROW_ORDER, values not in ROW_ORDER come after
    algs = sorted(list(set(algs)), key=lambda a: (0, ROW_ORDER.index(a)) if a in ROW_ORDER else (1, a))
    spreads = list(map(str, sorted(list(set(spreads)))))

    df = pd.DataFrame(columns=["Alg"] + spreads)
    df.set_index("Alg")

    table = {alg: {spread: "" for spread in spreads} for alg in algs}

    PLOT_ALGS = ["ERM", "IRM", "VREx", "WRI"]
    plot_df: pd.DataFrame = pd.DataFrame(columns=["Algorithm", "Test Acc", x_label])
    # process entries
    for alg, sweep_val in entries:
        k = (alg, sweep_val)
        for seed in entries[k]:
            for data_seed in entries[k][seed]:
                entries[k][seed][data_seed] = max(entries[k][seed][data_seed].values())
            entries[k][seed] = float(np.mean(list(entries[k][seed].values())))
        mean = float(np.mean(list(entries[k].values())))
        std = float(np.std(list(entries[k].values())))
        table[alg][str(sweep_val)] = f"{mean * 100:.1f} ± {std * 100:.1f}"

        if alg in PLOT_ALGS and (x_order is None or sweep_val in x_order):
            for value in entries[k].values():
                plot_df = pd.concat([
                        plot_df,
                        pd.DataFrame([{"Algorithm": alg, "Test Acc": value, x_label: sweep_val}])
                    ], ignore_index=True)

    b: plt.Axes = sns.barplot(
        data=plot_df,
        x=x_label,
        y="Test Acc",
        hue="Algorithm",
        order=x_order,
        hue_order=PLOT_ALGS,
        errorbar="ci",
        alpha=0.8,
        errwidth=1.2,
        ax=ax)
    entries = list(b.legend_.get_patches())
    labels = [t.get_text() for t in b.legend_.get_texts()]
    b.legend_.remove()
    b.set_ylim(0, 0.6)
    b.set_yticks([0, 0.2, 0.4, 0.6])
    b.set_xticklabels([f"${v.get_text()}$" for v in b.get_xticklabels()])
    b.set_ylabel("Test Acc", fontname="Times New Roman", size=18) #fontdict=dict(name='Times New Roman', family='serif', weight='normal', size=20))

    df = df.from_dict(table, orient='index')
    pd.options.display.width = 0
    print(df)

    return entries, labels


def plot_results(ax: plt.Axes, results_dir: Path, sweep_key: str, x_label: str, x_order: Optional[List[Any]]=None):
    results_path = results_dir / "results.json"
    if not results_path.exists():
        for d in os.listdir(results_dir):
            results_path = results_dir / d / "results.json"
            if results_path.exists():
                break
    assert results_path.exists()

    print(f"Loading {results_path}")
    with open(results_path, "r") as f:
        metrics = json.load(f)

    return process_results(metrics, ax, sweep_key, x_label, x_order)

def main():
    results_root = Path("results")
    experiments = [
        dict(results_dir=results_root / "sweep_hparams_figure_a",
             sweep_key="invariant_mean_std",
             x_label="$\sigma_{inv}$",
             x_order=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25]),
        dict(results_dir=results_root / "sweep_hparams_figure_b",
             sweep_key="invariant_std_delta",
             x_label="$\Delta_{inv}$",
             x_order=[0.0, 0.2, 0.4, 0.6, 0.8]),
        dict(results_dir=results_root / "sweep_hparams_figure_c",
             sweep_key="parity_cone_angle",
             x_label="$\\theta_{parity}$",
             x_order=[90, 87, 85, 80, 70])
    ]
    width = 4 * len(experiments)
    height = 4.2
    fig: plt.Figure = plt.figure(figsize=(width, height))
    axs = fig.subplots(1, len(experiments), squeeze=False)
    for ax, exp in zip(axs[0], experiments):
        entries, labels = plot_results(ax, **exp)


    from matplotlib import font_manager
    font = font_manager.FontProperties(family='Times New Roman',
                                       weight='normal',
                                       style='normal', size=18)

    fig.legend(entries, labels, loc='lower center', ncol=1000, prop=font)

    axs[0][0].set_title("(a)", fontdict=dict(fontname="Times New Roman", size=18, weight="normal"))
    axs[0][1].set_title("(b)", fontdict=dict(fontname="Times New Roman", size=18, weight="normal"))
    axs[0][2].set_title("(c)", fontdict=dict(fontname="Times New Roman", size=18, weight="normal"))

    plt.tight_layout(rect=[-0.01, 0.1, 1.01, 1.025])

    print("Saving to results/multidim_sim.pdf")
    fig.savefig("results/multidim_sim.pdf", backend="pgf")


if __name__ == "__main__":
    main()
