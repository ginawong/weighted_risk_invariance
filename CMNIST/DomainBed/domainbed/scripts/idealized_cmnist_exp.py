import itertools
import torch.nn.functional
from torch.utils.data import TensorDataset
import numpy as np
from tqdm import tqdm
from domainbed import datasets


def all_combinations(any_list):
    return itertools.chain.from_iterable(itertools.combinations(any_list, i + 1) for i in range(len(any_list)))


def digit_predictor(x, y, verify_expected=True):
    digit, color = x[:, 0], x[:, 1]
    expected_mapping = torch.zeros((10, 2), dtype=torch.long)
    expected_mapping[:5, :] = 1

    if verify_expected:
        # verify the best digit-based empirical predictor is what we expect
        best_correct, best_mapping = -1, None
        for d1 in all_combinations(range(10)):
            mapping = torch.zeros((10, 2), dtype=torch.long)
            mapping[list(d1), :] = 1
            f = mapping[digit, color]
            num_correct = torch.sum(f == y).item()
            if num_correct > best_correct:
                best_correct = num_correct
                best_mapping = mapping
        assert torch.all(expected_mapping == best_mapping)

    return expected_mapping


def color_predictor(x, y):
    digit, color = x[:, 0], x[:, 1]
    best_correct, best_mapping = -1, None
    # find the color-based predictor that empirically gets the most correct
    for c1 in all_combinations(range(2)):
        mapping = torch.zeros((10, 2), dtype=torch.long)
        mapping[:, list(c1)] = 1
        f = mapping[digit, color]
        num_correct = torch.sum(f == y).item()
        if num_correct > best_correct:
            best_correct = num_correct
            best_mapping = mapping
    return best_mapping


def erm_predictor(x, y):
    digit, color = x[:, 0], x[:, 1]
    best_correct, best_mapping = -1, None
    for dc1 in tqdm(all_combinations(range(20)), total=2**20):
        mapping = torch.zeros((20), dtype=torch.long)
        mapping[list(dc1)] = 1
        mapping = mapping.reshape(10, 2)
        f = mapping[digit, color]
        num_correct = torch.sum(f == y).item()
        if num_correct > best_correct:
            best_correct = num_correct
            best_mapping = mapping

    return best_mapping


def main():
    DATASETS = [
        "CMNISTHetero25",
        "CMNISTHetero25_CovShift65",
    ]

    data_cache = {}
    for dataset_name in DATASETS:
        print("Processing dataset: ", dataset_name)
        dataset = getattr(datasets, dataset_name)("domainbed_data", None, None, idealized=True)

        # create additional "unified" training dataset
        x0, y0 = dataset[0].tensors
        x1, y1 = dataset[1].tensors
        dataset.datasets.append(TensorDataset(
            torch.cat([x0, x1], dim=0),
            torch.cat([y0, y1], dim=0)
        ))

        data_cache[dataset_name] = {}
        for env_name, env in zip(("env_0", "env_1", "env_01"), (0, 1, 3)):
            x, y = dataset[env].tensors
            # ideal shape-predictive model
            f_inv = digit_predictor(x, y, verify_expected=True)
            y_hat_inv = f_inv[x[:, 0], x[:, 1]]
            # ideal color-predictive model
            f_spu = color_predictor(x, y)
            y_hat_spu = f_spu[x[:, 0], x[:, 1]]
            # predictor that uses both color and shape
            f_erm = erm_predictor(x, y)
            y_hat_erm = f_erm[x[:, 0], x[:, 1]]

            # empirical density (mass) of digit
            p_x_inv = torch.tensor(np.histogram(x[:, 0].numpy(), bins=np.linspace(-0.5, 9.5, 11), density=True)[0])
            data_cache[dataset_name][env_name] = {
                "digit": x[:, 0],
                "color": x[:, 1],
                "label": y,
                "prob_digit": p_x_inv,
                "pred_erm": y_hat_erm,
                "pred_digit_based": y_hat_inv,
                "pred_color_based": y_hat_spu,
                "map_erm": f_erm,
                "map_inv": f_inv,
                "map_spu": f_spu,
            }
        test_x, test_y = dataset[2].tensors
        data_cache[dataset_name]["env_2"] = {
            "digit": test_x[:, 0],
            "color": test_x[:, 1],
            "label": test_y
        }

    dwidth = max(map(len, DATASETS))
    sep = "+-" + "-" * dwidth + "-+-------------+----------+--------------+--------------+----------+"
    header = f'| {{:{dwidth}}} | Predictor   | ERM Loss | VREx Penalty | WRI Penalty  | Test Acc |'.format(
        "Dataset")
    entry_str = f"| {{:{dwidth}}} | {{:11}} | {{:8}} | {{:12}} | {{:12}} | {{:8}} |"

    print(sep)
    print(header)
    print(sep)

    for dataset_name in data_cache:
        data = data_cache[dataset_name]

        x_inv0 = data["env_0"]["digit"]
        y0 = data["env_0"]["label"]
        p_x_inv0 = data["env_0"]["prob_digit"]
        y_hat_erm0 = data["env_0"]["pred_erm"]
        y_hat_inv0 = data["env_0"]["pred_digit_based"]
        y_hat_spu0 = data["env_0"]["pred_color_based"]

        x_inv1 = data["env_1"]["digit"]
        y1 = data["env_1"]["label"]
        p_x_inv1 = data["env_1"]["prob_digit"]
        y_hat_erm1 = data["env_1"]["pred_erm"]
        y_hat_inv1 = data["env_1"]["pred_digit_based"]
        y_hat_spu1 = data["env_1"]["pred_color_based"]

        x_inv2, x_spu2 = data["env_2"]["digit"], data["env_2"]["color"]
        y2 = data["env_2"]["label"]
        map_erm01 = data["env_01"]["map_erm"]
        map_inv01 = data["env_01"]["map_inv"]
        map_spu01 = data["env_01"]["map_spu"]
        y2_hat_erm01 = map_erm01[x_inv2, x_spu2]
        y2_hat_inv01 = map_inv01[x_inv2, x_spu2]
        y2_hat_spu01 = map_spu01[x_inv2, x_spu2]
        test_acc_erm = (y2_hat_erm01 == y2).sum() / len(y2)
        test_acc_inv = (y2_hat_inv01 == y2).sum() / len(y2)
        test_acc_spu = (y2_hat_spu01 == y2).sum() / len(y2)

        erm_erm_loss0 = (y0 != y_hat_erm0).float()
        erm_inv_loss0 = (y0 != y_hat_inv0).float()
        erm_spu_loss0 = (y0 != y_hat_spu0).float()

        erm_erm_loss1 = (y1 != y_hat_erm1).float()
        erm_inv_loss1 = (y1 != y_hat_inv1).float()
        erm_spu_loss1 = (y1 != y_hat_spu1).float()

        mean_mean_erm_loss = 0.5 * (erm_erm_loss0.mean() + erm_erm_loss1.mean())
        mean_mean_inv_loss = 0.5 * (erm_inv_loss0.mean() + erm_inv_loss1.mean())
        mean_mean_spu_loss = 0.5 * (erm_spu_loss0.mean() + erm_spu_loss1.mean())

        wri_erm_penalty = (((erm_erm_loss0 * p_x_inv1[x_inv0]).mean() - (erm_erm_loss1 * p_x_inv0[x_inv1]).mean()) / mean_mean_erm_loss) ** 2
        wri_inv_penalty = (((erm_inv_loss0 * p_x_inv1[x_inv0]).mean() - (erm_inv_loss1 * p_x_inv0[x_inv1]).mean()) / mean_mean_inv_loss) ** 2
        wri_spu_penalty = (((erm_spu_loss0 * p_x_inv1[x_inv0]).mean() - (erm_spu_loss1 * p_x_inv0[x_inv1]).mean()) / mean_mean_spu_loss) ** 2

        vrex_erm_penalty = (erm_erm_loss0.mean() - mean_mean_erm_loss) ** 2 + (erm_erm_loss1.mean() - mean_mean_erm_loss) ** 2
        vrex_inv_penalty = (erm_inv_loss0.mean() - mean_mean_inv_loss) ** 2 + (erm_inv_loss1.mean() - mean_mean_inv_loss) ** 2
        vrex_spu_penalty = (erm_spu_loss0.mean() - mean_mean_spu_loss) ** 2 + (erm_spu_loss1.mean() - mean_mean_spu_loss) ** 2

        print(entry_str.format(
            "",
            "digit only",
            f"{mean_mean_inv_loss.item(): 0.4f}",
            f"{vrex_inv_penalty.item(): 0.6f}",
            f"{wri_inv_penalty.item(): 0.6f}",
            f"{test_acc_inv.item(): 0.5f}",
        ))
        print(entry_str.format(
            dataset_name,
            "color only",
            f"{mean_mean_spu_loss.item(): 0.4f}",
            f"{vrex_spu_penalty.item(): 0.6f}",
            f"{wri_spu_penalty.item(): 0.6f}",
            f"{test_acc_spu.item(): 0.5f}",
            )
        )
        print(entry_str.format(
            "",
            "digit+color",
            f"{mean_mean_erm_loss.item(): 0.4f}",
            f"{vrex_erm_penalty.item(): 0.6f}",
            f"{wri_erm_penalty.item(): 0.6f}",
            f"{test_acc_erm.item(): 0.5f}",
        ))
        print(sep)


if __name__ == "__main__":
    main()
