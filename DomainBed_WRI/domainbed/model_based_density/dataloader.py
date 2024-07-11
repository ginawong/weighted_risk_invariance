from typing import Optional, List, Callable, Any, Dict, Iterable, Iterator
from torch.utils.data import DataLoader, Sampler, BatchSampler, Dataset, RandomSampler


class InfiniteSampler(Sampler):
    def __init__(self, sampler: Sampler):
        super().__init__(None)
        self.sampler = sampler

    def __iter__(self):
        while True:
            for sample in self.sampler:
                yield sample

    def __len__(self):
        raise RuntimeError


class FixedCountSampler(Sampler):
    def __init__(self, sampler: Sampler, num_samples: int):
        super().__init__(None)
        self.sampler = sampler
        self.num_steps = num_samples

    def __iter__(self):
        step = 0
        while True:
            for sample in self.sampler:
                yield sample
                step += 1
                if step == self.num_steps:
                    return

    def __len__(self):
        return self.num_steps


class FixedLengthDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, num_batches: int, batch_size: int, num_workers: int = 0,
                 collate_fn: Optional[Callable[[List], Any]] = None, pin_memory: bool = False):
        batch_sampler = FixedCountSampler(
            BatchSampler(InfiniteSampler(RandomSampler(dataset, replacement=False)),
                batch_size,
                drop_last=False
            ),
            num_batches
        )

        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory)


class LoaderDictIter:
    def __init__(self, loaders: Dict[Any, DataLoader]):
        self.loaders = loaders
        self.num_samples = len(next(iter(self.loaders.values())))
        assert all(len(loader) == self.num_samples for loader in self.loaders.values())

    def __iter__(self) -> Iterator[Dict[Any, Any]]:
        return iter(self._gen_samples())

    def __len__(self) -> int:
        return self.num_samples

    def _gen_samples(self) -> Iterable[Dict[Any, Any]]:
        iters = {env: iter(loader) for env, loader in self.loaders.items()}
        while True:
            samples = dict()
            try:
                for env in self.loaders:
                    samples[env] = next(iters[env])
            except StopIteration:
                break
            yield samples


def test_data_loader():
    from torch.utils.data import TensorDataset
    import torch
    dataset = TensorDataset(torch.arange(300))
    loader = FixedLengthDataLoader(dataset, num_batches=4, batch_size=160)
    assert len(loader) == 4
    loader_iter = iter(loader)
    x0, = next(loader_iter)
    x1, = next(loader_iter)
    x2, = next(loader_iter)
    x3, = next(loader_iter)
    try:
        next(loader_iter)
        raise RuntimeError
    except StopIteration:
        pass
    except:
        raise
    a = torch.cat([x0, x1[:140]])
    b = torch.cat([x1[140:], x2, x3[:120]])
    c = x3[120:]

    assert len(torch.unique(a)) == 300
    assert len(torch.unique(b)) == 300
    assert len(torch.unique(c)) == 40

