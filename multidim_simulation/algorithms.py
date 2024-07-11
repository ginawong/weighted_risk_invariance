from typing import Dict, Any, Mapping, Optional, Tuple
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader
from torch import Tensor, autograd
from multidensity_model import MultiDensityModel
from classifier import Classifier
from average_meter import AverageMeter


class Algorithm(nn.Module):
    def __init__(self, classifier: Classifier, num_domains: int, hparams: Dict[str, Any]):
        super(Algorithm, self).__init__()
        self.classifier = classifier
        self.num_domains = num_domains
        self.hparams = hparams

    def compute_loss(self, minibatches: Dict[int, Tuple[Tensor, Tensor]], density_models: Optional[MultiDensityModel] = None)\
            -> Tensor:
        raise NotImplementedError

    def features(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def predict(self, x: Tensor, density_models: Optional[MultiDensityModel] = None)\
            -> Tensor:
        raise NotImplementedError

    @property
    def uses_density(self):
        raise NotImplementedError

    def reset_density_stats(self):
        pass

    def compute_extra_metrics(self, eval_loaders: Dict[str, DataLoader], density_models: Optional[MultiDensityModel] = None)\
            -> Dict[str, float]:
        return dict()

class ERM(Algorithm):
    def __init__(self, classifier: Classifier, num_domains: int, hparams: Dict[str, Any]):
        super().__init__(classifier, num_domains, hparams)
        self.average_loss = AverageMeter(tau=1000)

    @property
    def uses_density(self):
        return False

    def reset_density_stats(self):
        pass

    def compute_loss(self, minibatches: Dict[int, Tuple[Tensor, Tensor]], density_models: Optional[MultiDensityModel] = None) \
            -> Tensor:
        all_x = torch.cat([batch[0] for batch in minibatches.values()])
        all_y = torch.cat([batch[1] for batch in minibatches.values()])
        loss = nnf.cross_entropy(self.predict(all_x, density_models), all_y)
        self.average_loss.update(loss.item(), sum(len(batch) for batch in minibatches.values()))
        return loss

    def features(self, x: Tensor) -> Tensor:
        return self.classifier.features(x)

    def predict(self, x: Tensor, density_models: Optional[MultiDensityModel] = None) \
            -> Tensor:
        return self.classifier(x)

    def compute_extra_metrics(self, eval_loaders: Dict[str, DataLoader], density_models: Optional[MultiDensityModel] = None) \
            -> Dict[str, float]:
        metrics = dict(
            train_loss=self.average_loss.average
        )
        return metrics


class WRI(ERM):
    def __init__(self, classifier: Classifier, num_domains: int, hparams: Dict[str, Any]):
        super().__init__(classifier, num_domains, hparams)
        self.penalty_weight = hparams["wri_lambda"]
        self.average_densities = None
        self.reset_density_stats()

        self.average_loss = AverageMeter(tau=1000)
        self.average_nll = AverageMeter(tau=1000)
        self.average_penalty = AverageMeter(tau=1000)

    @property
    def uses_density(self):
        return True

    def reset_density_stats(self):
        self.average_densities = defaultdict(lambda: defaultdict(lambda: AverageMeter(tau=5000)))

    def compute_loss(self, minibatches: Dict[int, Tuple[Tensor, Tensor]], density_models: Optional[MultiDensityModel] = None) \
            -> Tensor:
        assert density_models is not None

        batch_size = 0

        losses = dict()
        scaled_losses = defaultdict(dict)
        for env_idx, batch in minibatches.items():
            x, y = batch
            nll = nnf.cross_entropy(self.predict(x, density_models), y, reduction='none')

            for density_env_idx in minibatches:
                with torch.no_grad():
                    if density_env_idx == env_idx:
                        continue
                    if not density_models.is_initialized(density_env_idx):
                        break
                    density = density_models.compute_density(density_env_idx, x, self)
                    self.average_densities[env_idx][density_env_idx].update(density.mean().item(), len(density))
                scaled_losses[env_idx][density_env_idx] = (nll * density).mean()

            batch_size += len(nll)
            losses[env_idx] = nll.mean()

        nll = torch.stack(list(losses.values())).mean()

        penalties = []
        all_envs = list(minibatches.keys())
        for i, env1_idx in enumerate(all_envs):
            for env2_idx in all_envs[i+1:]:
                if self.average_densities[env1_idx][env2_idx].count and self.average_densities[env2_idx][env1_idx].count:
                    density_scale = 0.5 * (self.average_densities[env1_idx][env2_idx].average + self.average_densities[env2_idx][env1_idx].average)
                    nll_scale = 0.5 * (losses[env1_idx] + losses[env2_idx])
                    total_scale = density_scale * nll_scale
                    penalties.append(((scaled_losses[env1_idx][env2_idx] - scaled_losses[env2_idx][env1_idx]) / total_scale)**2)

        penalty = sum(penalties) / len(penalties) if penalties else torch.tensor(0.0)
        penalty = penalty
        loss = nll + self.penalty_weight * penalty

        self.average_nll.update(nll.item(), batch_size)
        self.average_penalty.update(penalty.item(), batch_size)
        self.average_loss.update(loss.item(), batch_size)

        return loss

    def compute_extra_metrics(self, eval_loaders: Dict[str, DataLoader], density_models: Optional[MultiDensityModel] = None) \
            -> Dict[str, float]:
        return dict(
            train_nll=self.average_nll.average,
            train_penalty=self.average_penalty.average,
            train_loss=self.average_loss.average,
        )


class VREx(ERM):
    """Adapted from DomainBed"""
    def __init__(self, classifier: Classifier, num_domains: int, hparams: Dict[str, Any]):
        super().__init__(classifier, num_domains, hparams)
        self.penalty_weight = hparams['vrex_lambda']

        self.average_loss = AverageMeter(tau=1000)
        self.average_nll = AverageMeter(tau=1000)
        self.average_penalty = AverageMeter(tau=1000)

    def compute_loss(self, minibatches: Dict[int, Tuple[Tensor, Tensor]], density_models: Optional[MultiDensityModel] = None) \
            -> Tensor:
        assert density_models is not None

        batch_size = 0
        logits = dict()
        losses = dict()
        for env_idx, batch in minibatches.items():
            x, y = batch
            batch_size += len(x)
            logits[env_idx] = self.predict(x)
            losses[env_idx] = nnf.cross_entropy(logits[env_idx], y)

        mean = sum(losses.values()) / len(losses)
        penalty = sum((loss - mean) ** 2 for loss in losses.values()) / len(losses)
        loss = mean + self.penalty_weight * penalty

        self.average_nll.update(mean.item(), batch_size)
        self.average_penalty.update(penalty.item(), batch_size)
        self.average_loss.update(loss.item(), batch_size)

        return loss

    def compute_extra_metrics(self, eval_loaders: Dict[str, DataLoader], density_models: Optional[MultiDensityModel] = None) -> Dict[str, float]:
        return dict(
            train_nll=self.average_nll.average,
            train_penalty=self.average_penalty.average,
            train_loss=self.average_loss.average,
        )


class IRM(ERM):
    """Adapted from DomainBed"""
    def __init__(self, classifier: Classifier, num_domains: int, hparams: Dict[str, Any]):
        super().__init__(classifier, num_domains, hparams)
        self.penalty_weight = hparams['irm_lambda']

        self.average_loss = AverageMeter(tau=1000)
        self.average_nll = AverageMeter(tau=1000)
        self.average_penalty = AverageMeter(tau=1000)

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = nnf.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = nnf.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def compute_loss(self, minibatches: Dict[int, Tuple[Tensor, Tensor]], density_models: Optional[MultiDensityModel] = None) \
            -> Tensor:
        assert density_models is not None

        nll = 0
        penalty = 0
        batch_size = 0
        for env_idx, batch in minibatches.items():
            x, y = batch
            batch_size += len(x)
            logits = self.predict(x)
            nll = nll + nnf.cross_entropy(logits, y)
            penalty = penalty + self._irm_penalty(logits, y)

        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + self.penalty_weight * penalty

        self.average_nll.update(nll.item(), batch_size)
        self.average_penalty.update(penalty.item(), batch_size)
        self.average_loss.update(loss.item(), batch_size)

        return loss

    def compute_extra_metrics(self, eval_loaders: Dict[str, DataLoader], density_models: Optional[MultiDensityModel] = None) -> Dict[str, float]:
        return dict(
            train_nll=self.average_nll.average,
            train_penalty=self.average_penalty.average,
            train_loss=self.average_loss.average,
        )
