"""Diagnostic for the Q-Planning self-improvement collapse: run the authors' Q-only fine-tune step on the demonstrations alone.

Their ``self_improve.finetune`` mixes demonstrations with the planner's rollouts; this does the same optimisation with the online half
removed (same preprocessor, optimiser groups, learning rates, gradient clipping and EMA target update, all from their code). If the planner
still degrades after this, the fine-tune procedure itself is the cause; if not, the rollout data is.

    python qplan_official_demo_finetune.py --q <q>/pretrained_model --demos <dataset root> --out <dir> --steps 400
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--q", required=True)
    p.add_argument("--demos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    args = p.parse_args()

    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from qplanning.data.mixture import load_demonstrations
    from qplanning.modeling_qfunction import load_q_function
    from qplanning.processor_qfunction import make_q_train_preprocessor

    q_policy, _ = load_q_function(args.q, device="cuda", eval_mode=False)
    offline = load_demonstrations(["humanoid/pioneer_v1_left_arm"], args.demos, q_policy)
    loader = DataLoader(offline, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=3, pin_memory=True)
    preprocess = make_q_train_preprocessor(args.q)
    cfg = q_policy.config
    groups = q_policy.get_optim_params()
    groups[0]["lr"] = args.lr
    groups[1]["lr"] = args.lr * cfg.optimizer_lr_backbone / cfg.optimizer_lr
    optimizer = AdamW(groups, weight_decay=1e-4)
    q_policy.train()
    losses, it = [], iter(loader)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        optimizer.zero_grad()
        loss, loss_dict = q_policy.forward(preprocess(batch))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_policy.parameters(), 10.0)
        optimizer.step()
        q_policy.update()
        losses.append(float(loss_dict["td_ce_loss"]))
        if (step + 1) % 50 == 0:
            print(f"[DEMO-FT] step {step + 1}/{args.steps} td_ce {sum(losses[-50:]) / 50:.4f}", flush=True)
    q_policy.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    q_policy.save_pretrained(str(out))
    for pattern in ("policy_pre*", "policy_post*"):
        for path in Path(args.q).glob(pattern):
            shutil.copy2(path, out / path.name)
    print(f"[DEMO-FT] -> {out}")


if __name__ == "__main__":
    main()
