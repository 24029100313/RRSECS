import math, torch, bisect
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def lr_scheduler_builder(optimizer, lr_scheduler_cfg, len_loader, epochs):
    type, kwargs = lr_scheduler_cfg['type'], lr_scheduler_cfg['kwargs']

    total_iters = len_loader * epochs
    update_type = kwargs.get('update_type', 'iter')
    assert update_type in ['iter', 'epoch']

    if type == 'poly':
        lr_func = lambda x: (1 - x / total_iters if update_type == 'iter' else epochs) ** kwargs.get('lr_power', 0.9)
        lr_scheduler = LambdaLR(optimizer, lr_func)

    elif type == 'cosine':
        lr_func = lambda x: 0.5 * (1. + math.cos(math.pi * x / total_iters if update_type == 'iter' else epochs))
        lr_scheduler = LambdaLR(optimizer, lr_func)

    elif type == 'step':
        assert isinstance(kwargs.get('milestones'), int)
        lr_scheduler = StepLR(optimizer, kwargs.get('milestones', int(epochs * 2 / 3)))

    elif type == 'multistep':
        assert isinstance(kwargs.get('milestones'), list)
        default_dict = {12: [8, 11],
                        24: [16, 22],
                        36: [27, 33],
                        50: [36, 45],
                        100: [72, 90]}
        lr_scheduler = MultiStepLR(optimizer, kwargs.get('milestones', default_dict[epochs]))

    else:
        raise NotImplementedError('%s lr_scheduler is not implemented' % type)

    return lr_scheduler


def build_scheduler(optimizer, cfg, n_iter_per_epoch):
    num_steps = int(cfg.get('epochs', 300) * n_iter_per_epoch)
    warmup_steps = int(cfg.get('warmup_epochs', 20) * n_iter_per_epoch)
    decay_steps = int(cfg.get('decay_epochs', 30) * n_iter_per_epoch)
    multi_steps = [i * n_iter_per_epoch for i in cfg.get('multi_steps', [])]

    warmup_lr = cfg.get('warmup_lr', 1e-6)
    min_lr = cfg.get('min_lr', 1e-5)

    type, kwargs = cfg['lr_scheduler']['type'], cfg['lr_scheduler']['kwargs']
    
    if type == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps - warmup_steps,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=True,
        )
    elif type == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif type == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif type == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            milestones=multi_steps,
            gamma=0.1,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    else:
        raise NotImplementedError

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


class MultiStepLRScheduler(Scheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, milestones, gamma=0.1, warmup_t=0, warmup_lr_init=0.,
                 t_in_epochs=True) -> None:
        super().__init__(optimizer, param_group_field="lr")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

        assert self.warmup_t <= min(self.milestones)

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [v * (self.gamma ** bisect.bisect_right(self.milestones, t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None