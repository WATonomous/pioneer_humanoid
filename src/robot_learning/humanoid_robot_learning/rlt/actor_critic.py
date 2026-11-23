"""TD3 on the RL token, as in the paper: the actor sees the token, the robot state and the base's proposed chunk
(zeroed for half the training samples) and outputs a chunk; its loss is -Q + beta * ||a - a_ref||^2, so it edits the
base's actions instead of searching freely. Twin critics, target networks, delayed actor updates.
All actions are in the base policy's normalized space ([-1, 1] per dimension, MIN_MAX), flattened to C * A values.
"""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn


def mlp(i: int, o: int, hidden: int, layers: int) -> nn.Sequential:
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        d = hidden
    return nn.Sequential(*mods, nn.Linear(d, o))


class Actor(nn.Module):
    def __init__(self, z_dim: int, s_dim: int, a_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.net = mlp(z_dim + s_dim + a_dim, a_dim, hidden, layers)
        nn.init.zeros_(self.net[-1].weight)    # start as the base policy: a = a_ref
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: Tensor, s: Tensor, a_ref: Tensor) -> Tensor:
        # a residual on the reference (in tanh space), squashed into the action range; a zeroed (dropped) a_ref
        # leaves the network on its own
        return torch.tanh(torch.atanh(a_ref.clamp(-0.999, 0.999)) * (a_ref.abs().sum(-1, keepdim=True) > 0)
                          + self.net(torch.cat([z, s, a_ref], -1)))


class Critic(nn.Module):
    def __init__(self, z_dim: int, s_dim: int, a_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.q1 = mlp(z_dim + s_dim + a_dim, 1, hidden, layers)
        self.q2 = mlp(z_dim + s_dim + a_dim, 1, hidden, layers)

    def forward(self, z: Tensor, s: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([z, s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class TD3:
    def __init__(self, z_dim: int, s_dim: int, a_dim: int, device, beta: float = 1.0, gamma: float = 0.98,
                 lr: float = 3e-4, tau: float = 0.005, policy_noise: float = 0.1, noise_clip: float = 0.2,
                 ref_dropout: float = 0.5, actor_every: int = 2, hidden: int = 256, layers: int = 2,
                 fixed: torch.Tensor | None = None, smooth_t: float = 0.0, smooth_c: float = 0.0,
                 chunk: tuple[int, int] | None = None):
        """``fixed``: bool mask over the a_dim action values the actor may not change (they always equal the
        reference), e.g. the gripper command during insertion.

        Smoothness (CAPS, Mysore et al. 2021, applied to the actor's correction a - a_ref rather than the action: the
        base's own motion legitimately changes between decisions). Without it the corrections jump every chunk and the
        arm shakes at 5 Hz; filtering at run time instead cost success. ``smooth_t``: weight on the change of the
        correction between consecutive decisions (from replay's next state); ``smooth_c``: weight on its second
        difference within a chunk of ``chunk`` = (C, A) frames x values."""
        self.fixed = None if fixed is None else fixed.to(device)
        self.smooth_t, self.smooth_c, self.chunk = smooth_t, smooth_c, chunk
        if smooth_c and chunk is None:
            raise ValueError("smooth_c needs chunk=(C, A)")
        self.actor = Actor(z_dim, s_dim, a_dim, hidden, layers).to(device)
        self.critic = Critic(z_dim, s_dim, a_dim, hidden, layers).to(device)
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.beta, self.gamma, self.tau = beta, gamma, tau
        self.policy_noise, self.noise_clip, self.ref_dropout, self.actor_every = policy_noise, noise_clip, ref_dropout, actor_every
        self.updates = 0

    def _drop(self, a_ref: Tensor) -> Tensor:
        keep = (torch.rand(a_ref.shape[0], 1, device=a_ref.device) >= self.ref_dropout).float()
        return a_ref * keep

    def fix(self, a: Tensor, a_ref: Tensor) -> Tensor:
        return a if self.fixed is None else torch.where(self.fixed, a_ref, a)

    @torch.no_grad()
    def act(self, z: Tensor, s: Tensor, a_ref: Tensor, noise: float = 0.0) -> Tensor:
        a = self.actor(z, s, a_ref)
        a = (a + noise * torch.randn_like(a)).clamp(-1, 1) if noise else a
        return self.fix(a, a_ref)

    @torch.no_grad()
    def act_wiggle(self, z: Tensor, s: Tensor, a_ref: Tensor, std: float, C: int) -> Tensor:
        """The actor's chunk plus one Gaussian offset per chunk of C frames (train.py's exploration), kept on at run
        time: a smoothness-penalized actor stops wiggling the peg past the rim, and this puts a small, controlled
        wiggle back (light penalty, 4 mm: 44% without it, 65% with std 0.02, half the original actor's jerk)."""
        a = self.actor(z, s, a_ref).view(len(z), C, -1)
        a = a + std * torch.randn(len(z), 1, a.shape[-1], device=a.device)
        return self.fix(a.clamp(-1, 1).reshape(len(z), -1), a_ref)

    def update(self, b: dict[str, Tensor]) -> dict[str, float]:
        with torch.no_grad():
            n = (self.policy_noise * torch.randn_like(b["a"])).clamp(-self.noise_clip, self.noise_clip)
            a2 = self.fix((self.actor_t(b["z2"], b["s2"], b["ref2"]) + n).clamp(-1, 1), b["ref2"])  # acting sees the reference
            q1t, q2t = self.critic_t(b["z2"], b["s2"], a2)
            # the reward is 1 once (success) and 0 otherwise, so no return exceeds 1: clipping the target stops the
            # early overestimation (Q reached 2.2) that the actor chased in the first runs
            y = (b["r"] + self.gamma * (1 - b["d"]) * torch.min(q1t, q2t)).clamp(0.0, 1.0)
        q1, q2 = self.critic(b["z"], b["s"], b["a"])
        lc = (q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()
        self.opt_c.zero_grad(set_to_none=True)
        lc.backward()
        self.opt_c.step()
        out = {"critic": lc.item(), "q": q1.mean().item()}
        self.updates += 1
        if self.updates % self.actor_every == 0:
            a = self.fix(self.actor(b["z"], b["s"], self._drop(b["ref"])), b["ref"])
            bc = (a - b["ref"]).pow(2).mean(-1)
            la = (-self.critic(b["z"], b["s"], a)[0] + self.beta * bc).mean()
            smooth = {}
            if self.smooth_t or self.smooth_c:
                # on the full reference, as the actor runs (the dropout above is only for the Q term)
                res = self.fix(self.actor(b["z"], b["s"], b["ref"]), b["ref"]) - b["ref"]
                if self.smooth_t:
                    res2 = self.fix(self.actor(b["z2"], b["s2"], b["ref2"]), b["ref2"]) - b["ref2"]
                    live = 1 - b["d"]                       # a terminal transition has no next decision
                    smooth["smooth_t"] = ((res - res2).pow(2).mean(-1) * live).sum() / live.sum().clamp(min=1)
                    la = la + self.smooth_t * smooth["smooth_t"]
                if self.smooth_c:
                    r = res.view(-1, *self.chunk)
                    smooth["smooth_c"] = (r[:, 2:] - 2 * r[:, 1:-1] + r[:, :-2]).pow(2).mean()
                    la = la + self.smooth_c * smooth["smooth_c"]
            self.opt_a.zero_grad(set_to_none=True)
            la.backward()
            self.opt_a.step()
            with torch.no_grad():
                for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.lerp_(p, self.tau)
            out.update(actor=la.item(), bc=bc.mean().item(), **{k: v.item() for k, v in smooth.items()})
        return out


class Replay:
    def __init__(self, cap: int, dims: dict[str, int], device):
        self.data = {k: torch.zeros(cap, d, device=device) if d else torch.zeros(cap, device=device) for k, d in dims.items()}
        self.cap, self.n, self.i = cap, 0, 0

    def add(self, **items: Tensor) -> None:
        k = next(iter(items.values())).shape[0]
        idx = (torch.arange(k) + self.i) % self.cap
        for name, v in items.items():
            self.data[name][idx] = v.detach().float().to(self.data[name].device)
        self.i = (self.i + k) % self.cap
        self.n = min(self.n + k, self.cap)

    def sample(self, batch: int) -> dict[str, Tensor]:
        idx = torch.randint(0, self.n, (batch,))
        return {k: v[idx] for k, v in self.data.items()}
