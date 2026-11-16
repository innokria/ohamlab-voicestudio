"""The training loss of the flow head: Lagrangian Self-Distillation (LSD, arXiv 2505.18825).

Two terms on each frame: (1) instantaneous flow matching at s == t, (2) a self-distillation term that makes the (s -> t) jump match
the instantaneous flow at its own endpoint. A tiny net w(s, t) learns how much to trust each term (uncertainty weighting). That
net is used in training only: it is not part of the model you ship.

Adapted from MIT-licensed code (see LICENSE-THIRD-PARTY)."""
import math
import torch
from torch import nn


class MLP(nn.Sequential):
    def __init__(self, in_channels, hidden_channels):
        layers, dim = [], in_channels
        for h in hidden_channels[:-1]:
            layers += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        layers.append(nn.Linear(dim, hidden_channels[-1]))
        super().__init__(*layers)

    def forward(self, input, *rest):
        return super().forward(torch.cat((input, *rest), dim=-1))


def zero_init(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class _InputGradOnly(torch.autograd.Function):
    """y = f(x) whose backward flows only into x, never into f's parameters."""

    @staticmethod
    def forward(ctx, x, f):
        with torch.enable_grad():
            y = f(x)
        ctx.save_for_backward(x, y)
        return y.detach()

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_output, retain_graph=False)[0], None


def grad_x_only(f, x):
    return _InputGradOnly.apply(x, f)


class LSD(nn.Module):
    num_time_conds = 2

    def __init__(self, p_equal=0.75, lognorm_mean=0.4, lognorm_std=1.0, w_t_dims=32, w_t_depth=3, distill_prob=0.25):
        super().__init__()
        self.p_equal, self.lognorm_mean, self.lognorm_std, self.distill_prob = p_equal, lognorm_mean, lognorm_std, distill_prob
        self._skip_rng = torch.Generator().manual_seed(0)
        self.w_s_t = MLP(2, [w_t_dims] * w_t_depth + [1])
        self.w_s_t.apply(zero_init)

    def sample_t(self, x):
        return torch.sigmoid(torch.randn_like(x[..., :1]) * self.lognorm_std + self.lognorm_mean)

    def sample_s_t(self, x):
        logits = torch.randn_like(x[..., :2]) * self.lognorm_std + self.lognorm_mean
        return torch.sigmoid(logits.min(-1)[0])[..., None], torch.sigmoid(logits.max(-1)[0])[..., None]

    def loss(self, v_t, x_0, x_1):
        x, e = x_1, x_0
        metrics = {}
        t = self.sample_t(x)
        x_t = t * x + (1 - t) * e
        flow_diag = (v_t(t, t, x_t) - (x - e)).float().square().sum(dim=-1)
        metrics["flow_diag"] = flow_diag.mean()
        lv = self.w_s_t(t.float(), t.float()).squeeze(-1).float()
        flow_diag = flow_diag * lv.exp() / x_0.shape[-1] - lv
        # the distillation term costs ~2 extra forwards: done on a quarter of steps, divided by that share (same expected gradient)
        if self.training and self.distill_prob < 1.0 and float(torch.rand((), generator=self._skip_rng)) >= self.distill_prob:
            return self.p_equal * flow_diag, metrics, t
        s, t = self.sample_s_t(x)
        x_s = s * x + (1 - s) * e
        vt, dvdt = torch.func.jvp(v_t, (s, t, x_s), (torch.zeros_like(s), torch.ones_like(t), torch.zeros_like(x_s)))
        x_t = x_s + (t - s) * vt
        dxdt = vt + (t - s) * dvdt
        u_tgt = grad_x_only(lambda y: v_t(t, t, y), x_t)
        flow_distill = (dxdt - u_tgt).float().square().sum(dim=-1)
        metrics["flow_distill"] = flow_distill.mean()
        lv2 = self.w_s_t(s.float(), t.float()).squeeze(-1).float()
        flow_distill = flow_distill * lv2.exp() / x_0.shape[-1] - lv2
        w = (1 - self.p_equal) / (self.distill_prob if self.training else 1.0)
        return self.p_equal * flow_diag + w * flow_distill, metrics, t
