"""The fight policy's action distribution for rsl_rl: a move plus, with each
punch, its technique. Plugged in through the actor's distribution_cfg
({"class_name": "humanoid_boxing.distribution:HybridDistribution", ...}) and
class_name ("humanoid_boxing.distribution:FightMLPModel"); the env's action is
[move index (as a float), technique values...].
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules.distribution import Distribution
from torch.distributions import Categorical


class _HybridDeterministic(nn.Module):
  def __init__(self, num_classes: int):
    super().__init__()
    self.num_classes = num_classes

  def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
    k = self.num_classes
    return torch.cat([mlp_output[..., :k].argmax(-1, keepdim=True).float(), mlp_output[..., k:]], -1)


class HybridDistribution(Distribution):
  """A move (categorical over `num_classes`) plus continuous technique values
  (diagonal Gaussian, state-independent std). The action is
  [move index, technique...]; the MLP outputs [move logits, technique means]."""

  # PPO's entropy bonus on the technique Gaussian grows its std without bound
  # when technique isn't yet paying (it hit the cap within ~300 iterations
  # and made every punch random), so its share of the entropy is scaled down
  # and its std capped.
  GAUSSIAN_ENTROPY_SCALE = 0.1

  # A move starts only in guard (skills.MovePlayer.ready), ~2-4% of steps; on
  # the rest the action does nothing. Unmasked, those steps' log-probs were
  # pure noise in the policy gradient and their entropy hid that the choice in
  # guard had collapsed (fight_v5/v6: entropy ~2.7 of 2.9 while red chose
  # step_forward 99% of the time). So log-probs count only on decision steps
  # (technique: only for a punch chosen there), and entropy and KL are the mean
  # over decision steps. FightMLPModel sets `ready` from the observation.
  ready: torch.Tensor | None = None

  def __init__(self, output_dim: int, num_classes: int, init_std: float = 0.2, std_range=(0.05, 0.4)) -> None:
    super().__init__(output_dim)
    self.punch_ids: torch.Tensor | None = None
    self.num_classes = num_classes
    self.k = output_dim - 1
    self.log_std_param = nn.Parameter(torch.full((self.k,), float(torch.log(torch.tensor(init_std)))))
    self.log_std_range = (float(torch.log(torch.tensor(std_range[0]))), float(torch.log(torch.tensor(std_range[1]))))
    self._cat: Categorical | None = None
    self._normal: torch.distributions.Normal | None = None
    Categorical.set_default_validate_args(False)
    torch.distributions.Normal.set_default_validate_args(False)

  def update(self, mlp_output: torch.Tensor) -> None:
    n = self.num_classes
    self._cat = Categorical(logits=mlp_output[..., :n])
    std = self.log_std_param.clamp(*self.log_std_range).exp().expand_as(mlp_output[..., n:])
    self._normal = torch.distributions.Normal(mlp_output[..., n:], std)

  def sample(self) -> torch.Tensor:
    return torch.cat([self._cat.sample()[..., None].float(), self._normal.sample()], -1)  # type: ignore

  def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
    return _HybridDeterministic(self.num_classes)(mlp_output)

  def as_deterministic_output_module(self) -> nn.Module:
    return _HybridDeterministic(self.num_classes)

  @property
  def input_dim(self) -> int:
    return self.num_classes + self.k

  @property
  def mean(self) -> torch.Tensor:
    return torch.cat([self._cat.probs.argmax(-1, keepdim=True).float(), self._normal.mean], -1)  # type: ignore

  @property
  def std(self) -> torch.Tensor:
    spread = 1.0 - self._cat.probs.max(-1, keepdim=True).values  # type: ignore
    return torch.cat([spread, self._normal.stddev], -1)  # type: ignore

  def _ready(self, like: torch.Tensor) -> torch.Tensor:
    r = self.ready
    if r is None or r.shape != like.shape:
      return torch.ones_like(like, dtype=torch.bool)
    return r

  def _decision_mean(self, x: torch.Tensor) -> torch.Tensor:
    """x on decision steps, scaled so its batch mean is the mean over decision steps."""
    r = self._ready(x).float()
    return x * r / r.mean().clamp_min(1e-3)

  @property
  def entropy(self) -> torch.Tensor:
    h = self._cat.entropy() + self.GAUSSIAN_ENTROPY_SCALE * self._normal.entropy().sum(-1)  # type: ignore
    return self._decision_mean(h)

  @property
  def params(self) -> tuple[torch.Tensor, ...]:
    return (self._cat.logits, self._normal.mean, self._normal.stddev)  # type: ignore

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    move = outputs[..., 0].long()
    lp_cat = self._cat.log_prob(move)  # type: ignore
    lp_tech = self._normal.log_prob(outputs[..., 1:]).sum(-1)  # type: ignore
    ready = self._ready(lp_cat)
    if self.punch_ids is None:
      from humanoid_boxing.skills import MOVES, PUNCHES

      self.punch_ids = torch.tensor([MOVES.index(m) for m in PUNCHES], device=outputs.device)
    punch = torch.isin(move, self.punch_ids)
    return torch.where(ready, lp_cat, 0.0) + torch.where(ready & punch, lp_tech, 0.0)

  def kl_divergence(self, old_params, new_params) -> torch.Tensor:
    (ol, om, os_), (nl, nm, ns) = old_params, new_params
    kl_cat = torch.distributions.kl_divergence(Categorical(logits=ol), Categorical(logits=nl))
    N = torch.distributions.Normal
    return self._decision_mean(kl_cat + torch.distributions.kl_divergence(N(om, os_), N(nm, ns)).sum(-1))


READY_OBS = 28  # index of skills.own_state's "ready" flag in the raw actor observation (fight_actor_obs)


class FightMLPModel(MLPModel):
  """MLPModel that tells its HybridDistribution which samples are decision
  steps (fighter in guard), from the raw (un-normalised) observation."""

  def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
    raw = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
    if isinstance(self.distribution, HybridDistribution):
      self.distribution.ready = raw[..., READY_OBS] > 0.5
    return self.obs_normalizer(raw)
