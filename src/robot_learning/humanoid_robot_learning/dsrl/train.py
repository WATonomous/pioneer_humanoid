"""DSRL on a frozen Pioneer-1 (tabletop_pick): an actor-critic chooses the noise the flow sampler starts from, per chunk.

The base policy is untouched; the RL policy only steers which of Pioneer-1's own plans gets executed. Reward 1 when
the task succeeds (held 0.4 s), else 0; one RL step = one replan (16 frames, 0.64 s). The critic regresses each chunk's
Monte Carlo return (gamma^steps-to-success, 0 on failure) instead of bootstrapping, which kept overestimating (Q near 1
while the true return is ~0.4). The actor maximises Q with a KL penalty to the base noise N(0, I), so it can only move
away from Pioneer-1's own behaviour where the critic sees a clear gain. Training layouts are seeds
>= 10000; evaluation uses held-out layouts (default 1100-1299, the ones the base was scored on).

  python -m humanoid_robot_learning.dsrl.train --policy <checkpoint> --out runs/dsrl --hours 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from humanoid_robot_learning.dsrl.base import FrozenPioneer1
from humanoid_robot_learning.dsrl.envs import Envs
from humanoid_robot_learning.sim.evaluate import robot_for_action_dim, wilson


def mlp(i, o, h=512):
    return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


class Actor(nn.Module):
    """Tanh-squashed Gaussian over the noise; starts close to N(0, I) (= the base policy)."""

    def __init__(self, obs_dim, act_dim, w_max, init_std=0.55):
        super().__init__()
        self.net, self.w_max, self.init_std = mlp(obs_dim, 2 * act_dim), w_max, init_std
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias[:act_dim], 0.0)
        nn.init.constant_(self.net[-1].bias[act_dim:], math.log(init_std))

    def dist(self, obs):
        mu, log_std = self.net(obs).chunk(2, -1)
        return mu, log_std.clamp(-5, 1)

    def forward(self, obs, deterministic=False):
        mu, log_std = self.dist(obs)
        u = mu if deterministic else mu + log_std.exp() * torch.randn_like(mu)
        w = self.w_max * torch.tanh(u)
        logp = (-0.5 * ((u - mu) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - torch.log(self.w_max * (1 - torch.tanh(u) ** 2) + 1e-6).sum(-1)
        return w, logp


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1, self.q2 = mlp(obs_dim + act_dim, 1), mlp(obs_dim + act_dim, 1)

    def forward(self, obs, w):
        x = torch.cat([obs, w], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Buffer:
    """(features, noise, Monte Carlo return) of every chunk of finished episodes; FIFO, so it holds recent policies."""

    def __init__(self, cap, obs_dim, act_dim, device):
        self.o = torch.zeros(cap, obs_dim, dtype=torch.float16, device=device)
        self.w = torch.zeros(cap, act_dim, dtype=torch.float16, device=device)
        self.g = torch.zeros(cap, device=device)
        self.cap, self.n, self.i = cap, 0, 0

    def add(self, o, w, g):
        k = len(o)
        idx = (torch.arange(k, device=o.device) + self.i) % self.cap
        self.o[idx], self.w[idx], self.g[idx] = o.half(), w.half(), g
        self.i, self.n = (self.i + k) % self.cap, min(self.cap, self.n + k)

    def sample(self, b):
        j = torch.randint(0, self.n, (b,), device=self.o.device)
        return self.o[j].float(), self.w[j].float(), self.g[j]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--policy", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--warmup", type=int, default=5000, help="transitions with the base policy's own noise before learning")
    p.add_argument("--utd", type=float, default=0.25, help="gradient steps per new transition")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--w_max", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--kl", type=float, default=1.0, help="weight of the per-dimension KL to the base noise N(0, I)")
    p.add_argument("--buffer", type=int, default=150_000, help="recent chunks kept (FIFO): the returns are on-policy-ish")
    p.add_argument("--eval_seed", type=int, default=1100)
    p.add_argument("--eval_n", type=int, default=200)
    p.add_argument("--eval_every_min", type=float, default=60.0)
    p.add_argument("--max_s", type=float, default=30.0)
    p.add_argument("--eval_actor", type=Path, help="no training: score the base and this saved actor on the eval layouts")
    args = p.parse_args()

    dev = torch.device("cuda")
    base = FrozenPioneer1(args.policy)
    robot = robot_for_action_dim(base.action_dim)
    envs = Envs(args.envs, base.cameras, robot, max_s=args.max_s)
    act_dim = base.horizon * base.action_dim
    obs_dim = None
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "a")

    def say(**kw):
        kw["t_min"] = round((time.time() - t_start) / 60, 1)
        print(json.dumps(kw), flush=True)
        log.write(json.dumps(kw) + "\n")
        log.flush()

    t_start = time.time()
    train_seeds = itertools.count(10_000)
    nets = {}

    def build(dim):
        nets["actor"] = Actor(dim, act_dim, args.w_max).to(dev)
        nets["critic"] = Critic(dim, act_dim).to(dev)
        nets["opt_a"] = torch.optim.Adam(nets["actor"].parameters(), args.lr)
        nets["opt_c"] = torch.optim.Adam(nets["critic"].parameters(), args.lr)
        nets["buf"] = Buffer(args.buffer, dim, act_dim, dev)

    def choose(feat, mode):
        if mode == "base":
            return torch.randn(len(feat), act_dim, device=dev)
        with torch.no_grad():
            return nets["actor"](feat, deterministic=(mode == "mean"))[0]

    def run(mode, seeds, collect):
        """Roll out episodes on ``seeds`` (an iterator); collect=True keeps going forever (training) and stores
        transitions; otherwise runs each seed once and returns {seed: success}."""
        nonlocal obs_dim
        ids = list(range(envs.n))
        seed_of, results, active = {}, {}, {}
        first = [next(seeds, None) for _ in ids]
        live = [i for i, s in zip(ids, first) if s is not None]
        for i, (o, task) in zip(live, envs.reset(live, [first[i] for i in live])):
            seed_of[i] = first[i]
            active[i] = {"obs": o, "task": task, "prefix": None, "traj": []}
        while active:
            order = sorted(active)
            ctx, mask, feat = base.encode([active[i]["obs"] for i in order], [active[i]["task"] for i in order])
            feat = feat.float()
            if obs_dim is None:
                obs_dim = feat.shape[1]
                build(obs_dim)
            w = choose(feat, mode)
            noise = w.view(len(order), *base.noise_shape)
            prefixes = [active[i]["prefix"] for i in order]
            d = base.prefix_len
            prefix = torch.stack(prefixes) if all(x is not None for x in prefixes) else None
            if prefix is None and any(x is not None for x in prefixes):   # mixed batch: sample with and without prefix
                plan = torch.empty(len(order), base.horizon, base.action_dim, device=dev)
                has = [k for k, x in enumerate(prefixes) if x is not None]
                no = [k for k, x in enumerate(prefixes) if x is None]
                plan[has] = base.chunk(ctx[has], None if mask is None else mask[has], noise[has], torch.stack([prefixes[k] for k in has]))
                plan[no] = base.chunk(ctx[no], None if mask is None else mask[no], noise[no])
            else:
                plan = base.chunk(ctx, mask, noise, prefix)
            actions = base.unnormalise(plan[:, :base.n_exec])
            replies = envs.chunk(order, actions)
            finished = []
            for k, (i, (o, task, success, timeout, frames)) in enumerate(zip(order, replies)):
                if success or timeout:
                    if collect:   # every chunk of the episode gets its discounted return
                        traj = active[i]["traj"] + [(feat[k], w[k])]
                        steps_left = torch.arange(len(traj) - 1, -1, -1, device=dev, dtype=torch.float32)
                        g = float(success) * args.gamma ** steps_left
                        nets["buf"].add(torch.stack([t[0] for t in traj]), torch.stack([t[1] for t in traj]), g)
                        stats.append((success, frames))
                        new_transitions[0] += len(traj)
                    else:
                        results[seed_of[i]] = (success, frames)
                    finished.append(i)
                else:
                    active[i].update(obs=o, task=task, prefix=plan[k, base.n_exec:base.n_exec + d])
                    if collect:
                        active[i]["traj"].append((feat[k], w[k]))
            for i in finished:
                del active[i]
                s = next(seeds, None)
                if s is not None:
                    (o, task), = envs.reset([i], [s])
                    seed_of[i] = s
                    active[i] = {"obs": o, "task": task, "prefix": None, "traj": []}
            if collect:
                n_new, new_transitions[0] = new_transitions[0], 0
                yield n_new
        if not collect:
            yield results

    def evaluate(mode, tag):
        t = time.time()
        res = next(run(mode, iter(range(args.eval_seed, args.eval_seed + args.eval_n)), collect=False))
        k = sum(s for s, _ in res.values())
        lo, hi = wilson(k, len(res))
        say(eval=tag, success=k, n=len(res), ci=[round(lo, 3), round(hi, 3)], minutes=round((time.time() - t) / 60, 1),
            mean_frames_success=round(float(np.mean([f for s, f in res.values() if s] or [0])), 1))
        json.dump({str(s): v for s, v in res.items()}, open(args.out / f"eval_{tag}.json", "w"))
        return k

    def update():
        a, c = nets["actor"], nets["critic"]
        o, w, g = nets["buf"].sample(args.batch)
        q1, q2 = c(o, w)
        lc = F.mse_loss(q1, g) + F.mse_loss(q2, g)
        nets["opt_c"].zero_grad(set_to_none=True)
        lc.backward()
        nets["opt_c"].step()
        wn, _ = a(o)
        mu, log_std = a.dist(o)
        s0 = math.log(a.init_std)
        kl = (s0 - log_std + (torch.exp(2 * log_std) + mu ** 2) / (2 * a.init_std ** 2) - 0.5).mean()   # per dimension
        lq = torch.min(*c(o, wn))
        lactor = -lq.mean() + args.kl * kl
        nets["opt_a"].zero_grad(set_to_none=True)
        lactor.backward()
        nets["opt_a"].step()
        return {"q": q1.mean().item(), "g": g.mean().item(), "lc": lc.item(), "kl": kl.item(), "q_actor": lq.mean().item()}

    stats, new_transitions = [], [0]
    base_k = evaluate("base", "base")
    if args.eval_actor:
        nets["actor"].load_state_dict(torch.load(args.eval_actor, map_location=dev, weights_only=False)["actor"])   # our own file (it stores the run args)
        k = evaluate("sample", "actor_loaded")
        say(msg="eval only", base=base_k, actor=k, n=args.eval_n)
        envs.close()
        return
    best = base_k
    collected, updates, credit, last_eval, last_log, info = 0, 0, 0.0, time.time(), time.time(), {}
    gen = run("base", train_seeds, collect=True)
    learning = False
    while time.time() - t_start < args.hours * 3600:
        new = next(gen)
        collected += new
        if not learning and collected >= args.warmup:
            learning = True
            gen.close()
            gen = run("sample", train_seeds, collect=True)
            say(msg="warmup done, learning", collected=collected)
        if learning:
            credit += new * args.utd
            while credit >= 1:
                info = update()
                updates += 1
                credit -= 1
        if time.time() - last_log > 300 and stats:
            recent = stats[-200:]
            say(collected=collected, updates=updates, episodes=len(stats), train_success=round(np.mean([s for s, _ in recent]), 3),
                **{k: round(v, 4) for k, v in info.items()})
            last_log = time.time()
        if learning and time.time() - last_eval > args.eval_every_min * 60:
            gen.close()
            k = evaluate("sample", f"actor_{updates}")
            if k > best:
                best = k
                torch.save({"actor": nets["actor"].state_dict(), "critic": nets["critic"].state_dict(), "updates": updates,
                            "success": k, "args": vars(args)}, args.out / "best.pt")
            last_eval = time.time()
            gen = run("sample", train_seeds, collect=True)
    gen.close()
    k = evaluate("sample", f"actor_final_{updates}")
    if k > best:
        torch.save({"actor": nets["actor"].state_dict(), "critic": nets["critic"].state_dict(), "updates": updates,
                    "success": k, "args": vars(args)}, args.out / "best.pt")
    say(msg="done", base=base_k, best=max(best, k))
    envs.close()


if __name__ == "__main__":
    main()
