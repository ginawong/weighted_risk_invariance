import argparse


NO_HASH_ARGS = ['results_dir']

def add_main_args(parser: argparse.ArgumentParser):
    parser.add_argument("--algorithm", type=str, default="wri")

    parser.add_argument("--seed", default=0)
    parser.add_argument("--hparams_seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--use_hash_subdir", action="store_true",
                        help="Store results in subdirectory of --results_dir named using hash of args and hparams")
    parser.add_argument("--test_env", type=int, default=0)

    parser.add_argument("--data_seed", type=int, default=0)  # 8
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--r_dims", type=int, default=10)
    parser.add_argument("--s_dims", type=int, default=10)
    parser.add_argument("--samples_per_env", type=int, default=2000)
    parser.add_argument("--train_percentage", type=float, default=0.8)
    parser.add_argument("--loader_workers", type=int, default=0)
    parser.add_argument("--invariant_mean_std", type=float, default=0.307505816296629,
                        help="Defines how much P(R) differs between environments. Higher values imply larger "
                             " difference.")
    parser.add_argument("--invariant_std_delta", type=float, default=0.45798053462914223,
                        help="Between 0 and 1. Defines how much covariance of P(R) differs between environments. "
                             "Higher values imply larger difference.")
    parser.add_argument("--invariant_noise_std", type=float, default=0.7099788415072562,
                        help="Defines how much label noise there is in P(Y|R). Larger values make invariant features"
                             " less predictive of label.")
    parser.add_argument("--spurious_mean_std", type=float, default=1.7255012339485913,
                        help="Defines how much variation there is between P(S|Y=y) for different values of y. Larger values"
                             " here make spurious features more predictive of label.")
    parser.add_argument("--data_parity_swap", action="store_true",
                        help="Specifies that test environment should have opposite correlation with spurrious data")
    parser.add_argument("--parity_cone_angle", type=float, default=90,
                        help="Specifies the angle of the parity cone that distributions are constrained to. Only"
                             " applies if --data_parity_swap is provided")
    parser.add_argument("--true_invariant_classifier", action="store_true")


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Simulate multi-environment domain generalization")
    add_main_args(parser)
    args = parser.parse_args()
    return args