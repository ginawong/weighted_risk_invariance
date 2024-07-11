from typing import Dict, Callable, Any, Tuple, List, Optional
from .density_models import DensityModel, NNDensityModel
import torch
from torch.types import Device
from torch import Tensor
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as nnf
import itertools
from .dataloader import FixedLengthDataLoader, LoaderDictIter
from collections import defaultdict
from .average_meter import AverageMeter


class MultiDensityModel(nn.Module):
    def __init__(self, density_models: Dict[int, DensityModel]):
        super().__init__()
        self.density_models = nn.ModuleDict({str(k): v for k, v in density_models.items()})

    def compute_density(self, env: int, x: torch.Tensor, algorithm) -> Tensor:
        if self.density_models[str(env)].feature_domain:
            x = algorithm.features(x)
        return self.density_models[str(env)](x)

    def is_initialized(self, env: int):
        return self.density_models[str(env)].is_initialized()

    def refit(self, datasets: Dict[int, TensorDataset], algorithm):
        for env, density_model in self.density_models.items():
            env = int(env)
            self._refit_single(env, datasets[env], algorithm.features)

    def _refit_single(self, env: int, dataset: TensorDataset, featurizer: Callable[[Tensor], Tensor]):
        density_model = self.density_models[str(env)]
        if not density_model.needs_refit:
            return
        with torch.no_grad():
            if density_model.feature_domain:
                samples = featurizer(dataset.tensors[0])
            else:
                samples = dataset.tensors[0]
        density_model.refit(samples)

    def compute_extra_metrics(self, eval_loaders: Dict[str, DataLoader]) -> Dict[str, float]:
        return dict()


class OptimizedMultiDensityModel(MultiDensityModel):
    def __init__(self, density_models: Dict[int, NNDensityModel], lr: float, optimizer: str, weight_decay: float,
                 grad_clip: float, batch_size: int, wri_lambda: float, ent_lambda: float, fit_steps: int):
        super().__init__(density_models)

        params = itertools.chain(*[density_model.parameters() for density_model in density_models.values()])
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError
        self.lr = lr
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.wri_lambda = wri_lambda
        self.ent_lambda = ent_lambda
        self.fit_steps = fit_steps

        self.wri_penalty_meter = AverageMeter(tau=1000)
        self.ent_penalty_meter = AverageMeter(tau=1000)

    def refit(self, datasets: List[Tuple[TensorDataset, Any]], algorithm, device: Optional[Device] = None):
        train_loaders = self._init_dataloaders(datasets)

        for step, train_batches in enumerate(LoaderDictIter(train_loaders)):
            train_loss = self.compute_loss(train_batches, algorithm, device)

            self.optimizer.zero_grad()
            train_loss.backward()
            model_params = itertools.chain(*[density_model.parameters() for density_model in self.density_models.values()])
            torch.nn.utils.clip_grad_norm_(model_params, self.grad_clip)
            self.optimizer.step()

        for density_model in self.density_models.values():
            density_model.set_initialized()

    def compute_loss(self, minibatches: Dict[int, Tensor], algorithm, device: Optional[Device] = None) -> Tensor:
        batch_size = 0

        average_densities = defaultdict(dict)
        losses = dict()
        log_densities = dict()
        scaled_losses = defaultdict(dict)
        for env_idx, batch in minibatches.items():
            x, y = batch
            x, y = x.to(device), y.to(device)
            nll = nnf.cross_entropy(algorithm.predict(x), y, reduction='none')

            log_densities[env_idx] = self.density_models[str(env_idx)].log_density(x)

            for density_env_idx in minibatches:
                if density_env_idx != env_idx:
                    density = self.compute_density(density_env_idx, x, algorithm.penultimate)
                    average_densities[env_idx][density_env_idx] = density.mean()
                    scaled_losses[env_idx][density_env_idx] = (nll * density).mean()

            batch_size += len(nll)
            losses[env_idx] = nll.mean()

        wri_penalties = []
        all_envs = list(minibatches.keys())
        for i, env1_idx in enumerate(all_envs):
            for env2_idx in all_envs[i+1:]:
                if env2_idx in average_densities[env1_idx] and env1_idx in average_densities[env2_idx]:
                    density_scale = 0.5 * (average_densities[env1_idx][env2_idx] + average_densities[env2_idx][env1_idx])
                    nll_scale = 0.5 * (losses[env1_idx] + losses[env2_idx])
                    total_scale = density_scale * nll_scale
                    wri_penalties.append(((scaled_losses[env1_idx][env2_idx] - scaled_losses[env2_idx][env1_idx]) / total_scale)**2)

        entropy_penalties = []
        for env_idx in all_envs:
            entropy = -log_densities[env_idx].mean()
            entropy_penalties.append(entropy)

        wri_penalty = sum(wri_penalties) / len(wri_penalties)
        ent_penalty = sum(entropy_penalties) / len(entropy_penalties)

        self.wri_penalty_meter.update(wri_penalty.item(), batch_size)
        self.ent_penalty_meter.update(ent_penalty.item(), batch_size)

        return self.wri_lambda * wri_penalty + self.ent_lambda * ent_penalty

    def _init_dataloaders(self, datasets: List[Tuple[TensorDataset, Any]]) -> Dict[int, DataLoader]:
        train_loaders = dict()
        for env_idx, (dataset, _) in enumerate(datasets):
            train_loaders[env_idx] = FixedLengthDataLoader(dataset, self.fit_steps, self.batch_size)
        return train_loaders

    def compute_extra_metrics(self, eval_loaders: Dict[str, DataLoader]) -> Dict[str, float]:
        return dict(
            mds_wri_penalty=self.wri_penalty_meter.average,
            mds_ent_penalty=self.ent_penalty_meter.average,
        )