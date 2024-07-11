from typing import List, Optional, Union, Tuple
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from numpy.random import RandomState


class Normal:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def sample(self, size: Union[int, Tuple[int, ...]], rng: Optional[RandomState] = None) -> np.ndarray:
        if rng is None: rng = np.random
        return rng.normal(loc=self.mean, scale=self.std, size=size)

    def logpdf(self, x: np.ndarray):
        return norm.logpdf(x, loc=self.mean, scale=self.std)

    def logcdf(self, x: np.ndarray):
        return norm.logcdf(x, loc=self.mean, scale=self.std)


class MVNormal:
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        assert mean.shape == (mean.shape[0],)
        assert cov.shape == (mean.shape[0], mean.shape[0])
        self.mean = mean.copy()
        self.cov = cov.copy()

    def dims(self):
        return self.mean.shape[0]

    def sample(self, num_samples: int, rng: Optional[RandomState] = None) -> np.ndarray:
        if rng is None: rng = np.random
        return rng.multivariate_normal(self.mean, self.cov, num_samples)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)



class Environment:
    """
    Environment class encapsulates distributional parameters that change between environments
    """
    def __init__(self, dist_r: MVNormal, dist_s_cond_y: List[MVNormal]):
        self.num_classes = len(dist_s_cond_y)
        assert self.num_classes > 1
        assert all(s.dims() == dist_s_cond_y[0].dims() for s in dist_s_cond_y)

        self.dist_r = dist_r
        self.dist_s_cond_y = dist_s_cond_y
        self.dims_r = dist_r.dims()
        self.dims_s = dist_s_cond_y[0].dims()

    def sample_r(self, num_samples: int, rng: Optional[RandomState] = None) -> np.ndarray:
        return self.dist_r.sample(num_samples, rng)

    def sample_s_cond_y(self, y: np.ndarray, rng: Optional[RandomState] = None) -> np.ndarray:
        num_samples = len(y)
        s_cond_y = []
        for dist_s_cond_yi in self.dist_s_cond_y:
            s_cond_y.append(dist_s_cond_yi.sample(num_samples, rng))
        s_cond_y = np.array(s_cond_y)
        return s_cond_y[y, np.arange(num_samples)]

    def logpdf_r(self, r: np.ndarray) -> np.ndarray:
        return self.dist_r.logpdf(r)

    def logpdf_s_cond_y(self, s: np.ndarray, y: np.ndarray) -> np.ndarray:
        log_p_s_cond_y = []
        for dist_s_cond_yi in self.dist_s_cond_y:
            log_p_s_cond_y.append(dist_s_cond_yi.logpdf(s))
        log_p_s_cond_y = np.array(log_p_s_cond_y)
        return log_p_s_cond_y[y, np.arange(len(y))]


class Invariant:
    """
    Invariant class encapsulates distributional parameters that do not change between environments
    """
    def __init__(self, class_dirs_r: np.ndarray, dist_noise: Normal):
        self.class_dirs_r = class_dirs_r / np.linalg.norm(class_dirs_r, axis=1, keepdims=True)
        self.dist_noise = dist_noise
        self.num_classes = len(self.class_dirs_r)
        assert self.num_classes > 1

    def sample_y_cond_r(self, r: np.ndarray, rng: Optional[RandomState] = None) -> np.ndarray:
        if rng is None: rng = np.random
        num_samples = r.shape[0]
        num_classes = self.class_dirs_r.shape[0]
        eps = self.dist_noise.sample((num_samples, num_classes), rng)
        y = np.argmax(r @ self.class_dirs_r.T + eps, axis=1)
        return y

    def logpmf_y_cond_r(self, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        # compute in chunks to avoid running out of memory
        if len(y) > 1000:
            return np.concatenate([self.logpmf_y_cond_r(y[i:i+1000], r[i:i+1000]) for i in range(0, len(y), 1000)])

        # code here explicitly assumes noise is gaussian
        mean_proj = r @ self.class_dirs_r.T + self.dist_noise.mean
        samples = np.linspace(-6 * self.dist_noise.std, 6 * self.dist_noise.std, 121).reshape(1, -1)
        pdf_mask = np.zeros(mean_proj.shape, dtype=bool)
        num_classes = self.class_dirs_r.shape[0]
        pdf_mask[np.arange(len(y)), y] = True
        pdf_mean = mean_proj[pdf_mask].reshape(-1, 1)
        cdf_mean = mean_proj[~pdf_mask].reshape(-1, num_classes - 1)

        x = samples + pdf_mean
        log_integrand = norm.logpdf(x, loc=pdf_mean, scale=self.dist_noise.std)
        for cdf_col in range(num_classes - 1):
            log_integrand += norm.logcdf(x, loc=cdf_mean[:, cdf_col:cdf_col+1], scale=self.dist_noise.std)
        integrand = np.exp(log_integrand)
        pmf_y_cond_r = np.trapz(integrand, dx=samples[0, 1]-samples[0, 0])
        # clamp to (0, 1) range
        dtype = pmf_y_cond_r.dtype
        logpmf_y_cond_r = np.log(np.clip(pmf_y_cond_r, np.finfo(dtype).tiny, 1 - np.finfo(dtype).epsneg))
        return logpmf_y_cond_r


class DataModel:
    """
    DataModel provides mechanism to sample data from each environment and provides true pdfs and pmfs
    """
    def __init__(self, environments: List[Environment], invariant: Invariant, rng: Optional[RandomState] = None):
        self.environments = environments
        self.invariant = invariant

        assert all(e.dims_r == self.environments[0].dims_r for e in self.environments)
        assert all(e.dims_s == self.environments[0].dims_s for e in self.environments)

        self._rng = rng
        self._r_dims = self.environments[0].dims_r
        self._s_dims = self.environments[0].dims_s

    @property
    def num_envs(self) -> int:
        return len(self.environments)

    @property
    def num_classes(self) -> int:
        return self.invariant.num_classes

    @property
    def num_data_dims(self) -> int:
        return self.num_invariant_dims + self.num_spurious_dims

    @property
    def num_invariant_dims(self) -> int:
        return self._r_dims

    @property
    def num_spurious_dims(self) -> int:
        return self._s_dims

    def sample(self, e: int, n: int) -> (np.ndarray, np.ndarray, np.ndarray):
        env = self.environments[e]
        r = env.sample_r(n, rng=self._rng)
        y = self.invariant.sample_y_cond_r(r, rng=self._rng)
        s = env.sample_s_cond_y(y, rng=self._rng)
        x = self.get_x_from_r_s(r, s)
        return x, y

    def pdf_x_y(self, e: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r, s = self.get_r_s_from_x(x)
        env = self.environments[e]
        logpdf_r = env.logpdf_r(r)
        logpmf_y_cond_r = self.invariant.logpmf_y_cond_r(y, r)
        logpdf_s_cond_y = env.logpdf_s_cond_y(s, y)
        # S is conditionally independent of R given Y
        return np.exp(logpdf_r + logpmf_y_cond_r + logpdf_s_cond_y)

    def pdf_x(self, e: int, x: np.ndarray) -> np.ndarray:
        p_x_y = sum(self.pdf_x_y(e, x, np.full(x.shape[0], y, dtype=int)) for y in range(self.num_classes))
        return p_x_y

    def pmf_y(self, e: int, y: np.ndarray) -> np.ndarray:
        # cant compute this efficiently in multi-class case
        raise NotImplementedError

    def get_r_s_from_x(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        assert x.shape[1] == self._r_dims + self._s_dims
        r, s = x[:, :self._r_dims], x[:, self._r_dims:]
        return r, s

    @staticmethod
    def get_x_from_r_s(r: np.ndarray, s: np.ndarray) -> np.ndarray:
        x = np.concatenate([r, s], axis=1)
        return x

    def __repr__(self):
        s = f"{self.__class__.__name__}(environments=[\n"
        for e in self.environments:
            s += f"        {e.__class__.__name__}(\n"
            s += f"            dist_r={repr(e.dist_r)},\n"
            s += f"            dist_s_cond_y={repr(e.dist_s_cond_y)},\n"
        s += "    ],\n"
        s += f"    invariant={self.invariant.__class__.__name__}(\n"
        s += f"        class_dirs_r={repr(self.invariant.class_dirs_r)},\n"
        s += f"        dist_noise={repr(self.invariant.dist_noise)}))\n"
        return s


def random_covariance(num_dims: int, min_std: float, max_std: float, rng: Optional[RandomState] = None) -> np.ndarray:
    if rng is None: rng = np.random
    sigma = rng.uniform(min_std, max_std, num_dims)
    q, _ = np.linalg.qr(rng.standard_normal((num_dims, num_dims)))
    cov_sqrt = (np.diag(sigma) @ q)
    return cov_sqrt @ cov_sqrt.T


def test_random_covariance():
    cov = random_covariance(5, 1, 3)
    var, _ = np.linalg.eig(cov)
    assert cov.shape == (5, 5)
    assert np.all(var > 0)
    assert np.all(np.logical_or(1 <= np.sqrt(var), np.sqrt(var) <= 3))
    assert np.sum(np.abs(cov - cov.T)) < 1e-7
    assert np.linalg.det(cov) > 0


def test_environment():
    N, C, DR, DS = 10000, 4, 2, 2
    env = Environment(
        dist_r=MVNormal(np.zeros(DR), np.eye(DR)),
        dist_s_cond_y=[
            MVNormal(np.ones(DS), np.eye(DS)),
            MVNormal(-np.ones(DS), np.eye(DS)),
            MVNormal(np.concatenate([np.ones(DS//2), -np.ones(DS-DS//2)]), np.eye(DS)),
            MVNormal(np.concatenate([-np.ones(DS // 2), np.ones(DS - DS // 2)]), np.eye(DS)),
        ])

    class_directions = np.array([
        np.concatenate([np.full(DR // 2, 1), np.full(DR - DR // 2, 0)]),
        np.concatenate([np.full(DR // 2, 0), np.full(DR - DR // 2, 1)]),
        np.concatenate([np.full(DR // 2, -1), np.full(DR - DR // 2, 0)]),
        np.concatenate([np.full(DR // 2, 0), np.full(DR - DR // 2, -1)]),
    ])
    inv = Invariant(class_directions, Normal(0, 1))

    r = env.sample_r(N)
    y = inv.sample_y_cond_r(r)
    s = env.sample_s_cond_y(y)
    p_r = np.exp(env.logpdf_r(r))
    p_s_cond_y = np.exp(env.logpdf_s_cond_y(s, y))

    p_y0_cond_r = np.exp(inv.logpmf_y_cond_r(np.full(N, 0, dtype=int), r))
    p_y1_cond_r = np.exp(inv.logpmf_y_cond_r(np.full(N, 1, dtype=int), r))
    p_y2_cond_r = np.exp(inv.logpmf_y_cond_r(np.full(N, 2, dtype=int), r))
    p_y3_cond_r = np.exp(inv.logpmf_y_cond_r(np.full(N, 3, dtype=int), r))
    assert np.all(np.abs(p_y0_cond_r + p_y1_cond_r + p_y2_cond_r + p_y3_cond_r - 1) < 1e-6)

    # statistically high probability
    assert np.mean(y[p_y0_cond_r > 0.7] == 0) > 0.4
    assert np.mean(y[p_y1_cond_r > 0.7] == 1) > 0.4
    assert np.mean(y[p_y2_cond_r > 0.7] == 2) > 0.4
    assert np.mean(y[p_y3_cond_r > 0.7] == 3) > 0.4

    assert r.shape == (N, DR)
    assert s.shape == (N, DS)
    assert y.shape == (N,)
    assert p_y0_cond_r.shape == (N,)
    assert p_r.shape == (N,)
    assert p_s_cond_y.shape == (N,)


def test_data_model():
    N, E, C, DR, DS = 10000, 3, 3, 10, 10
    environments = [
        Environment(
            dist_r=MVNormal(np.random.randn(DR), random_covariance(DR, 0.5, 1.5)),
            dist_s_cond_y=[MVNormal(np.random.randn(DS), random_covariance(DS, 0.5, 1.5)) for _ in range(C)])
        for _ in range(E)]
    invariant = Invariant(np.random.randn(C, DR), Normal(0.0, 1.0))
    data_model = DataModel(environments, invariant)

    assert data_model.num_envs == E

    for env in range(E):
        x, y = data_model.sample(env, N)
        assert data_model.pdf_x_y(env, x, y).shape == (N,)
