"""Scale warmup initialization like SWIFT's per-group max/min learning rates.

Enabled explicitly by the launcher. SWIFT folds vit_lr/aligner_lr multipliers
into max_lr/min_lr and resets lr_mult to one, while Megatron's default warmup
uses a shared init_lr. Apply the same peak ratio to that starting value.
"""

from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


def _install():
    original = OptimizerParamScheduler.get_lr
    if getattr(original, "_group_warmup_scaled", False):
        return

    def get_lr(self, param_group):
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            max_lr = param_group.get("max_lr", self.max_lr)
            init_lr = self.init_lr * max_lr / self.max_lr if self.max_lr else 0.0
            progress = float(self.num_steps) / float(self.lr_warmup_steps)
            return init_lr + (max_lr - init_lr) * progress
        return original(self, param_group)

    get_lr._group_warmup_scaled = True
    OptimizerParamScheduler.get_lr = get_lr


_install()
