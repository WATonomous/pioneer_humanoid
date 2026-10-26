# Flow transformer policy

A conditional flow-matching action-chunk policy trained from scratch on the simulator datasets
(no VLM or language input, and no pretrained π₀ weights). The architecture is in
[`policy.py`](policy.py):

- Two 480×640 cameras go through one ImageNet ResNet-18. It is trainable, with frozen BN, and
  gives 15×20 spatial tokens per camera.
- A state token and a flow-time token are added. All of these are cross-attention memory for a
  6-layer, 8-head, width-512 noncausal decoder over 50 action tokens.
- Training loss: `x_t = (1-t)·ε + t·A`, velocity target `A - ε`, masked MSE.
- Inference: 10 Euler steps from Gaussian noise, encoding the images once per chunk.

| Module | Role |
|---|---|
| `policy.py` | `FlowConfig`, `FlowPolicy`, `flow_loss` |
| `data.py` | `ChunkDataset`: LeRobot roots read in place, episode identities and deduplication, selection plans, phase pools, training-frame normalization |
| `train.py` | CLI: cosine-from-scratch or constant-LR branch, EMA 0.999, exact resume |

```bash
# from scratch (100k updates, batch 8 with phase quotas 3/2/2/1)
python -m humanoid_il.flow_transformer.train --root outputs/train/flow --dataset datasets/pioneer_peg_insert \
  --dataset datasets/corr_r1 --phase-config src/robot_learning/config/peg_insert_phases.json --steps 100000
# continue onto new data: keeps the checkpoint's normalization, constant LR, no warmup restart
python -m humanoid_il.flow_transformer.train --root outputs/train/flow_b --dataset ... --selection plan.json \
  --resume-from outputs/train/flow/checkpoints/100000.pt --schedule constant --lr 1e-5 --vision-lr 1e-6 --steps 150000
```

Evaluate, audit or collect corrections with any `<step>.pt` (EMA weights), through
`humanoid_robot_learning.policy_adapters.load_policy`:

```bash
python -m humanoid_mujoco_scenes.peg_insert.eval --checkpoint outputs/train/flow/checkpoints/100000.pt --device cuda \
  --layouts heldout_layouts.json --n-action-steps 50 --refinement-steps 10 --report outputs/eval/flow/heldout.json
```

Results against ACT are in [`docs/research/peg_insert_act_vs_flow.md`](../../../../docs/research/peg_insert_act_vs_flow.md).
