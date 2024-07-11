import logging
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from sklearn.mixture import BayesianGaussianMixture
from multiprocessing import Pool
import sys


class DensityModel(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def needs_refit(self):
        raise NotImplementedError

    @property
    def feature_domain(self):
        raise NotImplementedError

    def is_initialized(self):
        raise NotImplementedError

    def refit(self, samples: Tensor):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class GMMDensityModel(DensityModel):
    def __init__(self, n_components: int, covariance_type: str = 'full',
                 reg_covar: float = 1e-6, max_iter: int = 2000, singularity_threshold: float=1e30,
                 num_workers: int = 0):
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.singularity_threshold = singularity_threshold
        self.num_workers = num_workers
        self.gm: Optional[BayesianGaussianMixture] = None

    @property
    def needs_refit(self):
        return True

    @property
    def feature_domain(self):
        return True

    def is_initialized(self):
        return self.gm is not None

    def refit(self, samples: Tensor):
        samples = samples.detach().cpu().numpy()
        reg_covar = self.reg_covar
        self.gm = None
        while self.gm is None:
            self.gm = BayesianGaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                verbose=0,
                tol=1e-3,
                reg_covar=reg_covar,
            ).fit(samples)
            sample_density = self._apply_gmm(samples)
            max_density = np.max(sample_density)
            if max_density > self.singularity_threshold:
                reg_covar *= 10
                self.gm = None
                logging.info(
                    f"GMM estimate too high ({max_density} > {self.ingularity_threshold}),"
                    f" reg_covar increased to {reg_covar}")

    def _apply_gmm(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2, f"GMM called on {len(x.shape)}-dim data. " \
                                  f"Flatten or use featurizer first (recommended)."

        # ------------ parallelize the GMM sampling for speed ------------
        num_workers_local = min(max(1, self.num_workers), len(x))
        if num_workers_local > 1:
            # partition data (preserving order)
            x_parts = []
            part_len = math.ceil(len(x) / num_workers_local)
            for i in range(self.num_workers):
                x_parts.append((x[i * part_len:(i + 1) * part_len], self.gm))
                if len(x_parts[-1][0]) == 0:
                    x_parts.pop(-1)
                    num_workers_local -= 1
            assert len(x_parts) == num_workers_local
            assert sum(len(x_part[0]) for x_part in x_parts) == len(x)

            # create multiprocessing pool, use a GMM to compute log probabilities
            pool = Pool(processes=num_workers_local)
            log_probs_parts = []
            for log_probs_part in pool.imap(self._apply_density_worker, x_parts):
                log_probs_parts.append(log_probs_part)
            pool.close()
            pool.join()
            log_probs = np.concatenate(log_probs_parts, axis=0)
        else:
            log_probs = self._apply_density_worker((x, self.gm))

        assert len(log_probs) == len(x)

        return np.exp(log_probs)[:, None, ...]

    @staticmethod
    def _apply_density_worker(args: Tuple[np.ndarray, BayesianGaussianMixture]):
        try:
            x_part, skl_density_model = args
            log_probs = skl_density_model.score_samples(x_part)
            return log_probs
        except Exception as e:
            print(e, file=sys.stderr)
            raise

    def forward(self, x: Tensor):
        x_np = x.detach().cpu().numpy()
        density_np = self._apply_gmm(x_np)
        density = torch.from_numpy(density_np).to(device=x.device, dtype=x.dtype)
        return density


class NNDensityModel(DensityModel):
    def __init__(self, dimensions, min_density, max_density):
        super().__init__()
        self.dimensions = dimensions
        self.model = nn.Sequential(
            nn.Linear(dimensions, max(1, dimensions // 2)),
            nn.ReLU(),
            nn.Linear(max(1, dimensions // 2), 1),
            nn.Sigmoid()
        )
        assert min_density > 1e-8
        self.min_density = min_density
        self.max_density = max_density
        self._initialized = False

    def refit(self, samples: Tensor):
        pass

    @property
    def needs_refit(self) -> bool:
        return True

    @property
    def feature_domain(self) -> bool:
        return False

    def is_initialized(self) -> bool:
        return self._initialized

    def set_initialized(self):
        self._initialized = True

    def forward(self, x: Tensor) -> Tensor:
        return self.log_density(x).exp()

    def log_density(self, x: Tensor) -> Tensor:
        mn = math.log(self.min_density)
        mx = math.log(self.max_density)
        return self.model(x) * (mx - mn) + mn
