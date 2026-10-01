# SO-101 vial-pick manipulation — robot learning research log

**Task:** `Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval` (Isaac Lab sim), SO-101 arm picks up a vial, places it in a rack.
**Dataset:** HF `CursedRock17/so101_teleop_vials_sim_and_real` — 140 episodes, 26,516 frames. **125 sim / 15 real** (visually confirmed by frame inspection; episodes 0–124 sim, 125–139 real — a clean concatenation, not interleaved). No domain randomization visible in the sim portion (consistent black background/lighting across all sampled sim episodes).
**Hardware:** single RTX 4060 laptop GPU, 8GB VRAM.

## Result summary (best configs, larger-N where available)

| Policy | Config | Success rate |
|---|---|---|
| ACT | tuned arch (4 decoder layers, chunk=30, head LR 1e-4), 30K steps, `n_action_steps=10` | **47.5%** (N=40, 2 seeds) |
| SmolVLA | 15K steps, RTC (execution_horizon=10, guidance_weight=1.0) | **47.5%** (N=40, 2 seeds) |
| ACT | same, single seed | 50% (N=20) |
| SmolVLA | 15K, RTC gw=1.0, single seed | 55% (N=20) — did not replicate on a 2nd seed (40%) |

**ACT and SmolVLA are statistically tied at ~47.5% once evaluated at N=40.** Neither cleanly broke 50%.

## What worked

- **Closed-loop replanning over open-loop chunk execution.** Both ACT (default `n_action_steps=100`, i.e. execute full chunk blind) and SmolVLA (default 50) were open-loop by default. Overriding to replan every ~10 steps was the single largest, most consistent lever for both architectures (ACT: ~35%→50%; SmolVLA: ~20-30%→40%).
- **ACT architecture tuning:** shorter `chunk_size` (100→30), deeper transformer decoder (1→4 layers, matching the original ACT paper vs. lerobot's shallow default), higher head learning rate (1e-5→1e-4, backbone kept low). Converged faster and reached better loss at every step count than the untouched baseline.
- **RTC (Real-Time Chunking) for SmolVLA**, once (a) actually wired correctly — lerobot's `select_action()` refuses to run with RTC enabled; it must be driven manually via `predict_action_chunk()` with a hand-rolled `ActionQueue` — and (b) its `max_guidance_weight` tuned down from the 10.0 default to 1.0 (lower = less pull toward the previous, possibly-failing trajectory). At the default weight RTC underperformed simple truncation (35% vs 40%); at weight=1.0 it became the best SmolVLA config found.
- **Fixed two real bugs found along the way**, both now committed:
  - `LocalLeRobotPolicy.reset()` was a no-op — policy action-queue state leaked across episode boundaries in every multi-episode eval. Fixed to forward to the real `policy.reset()`.
  - RTC's guidance step needs live gradients (`torch.autograd.grad` inside `torch.enable_grad()`); wrapping the chunk call in `torch.inference_mode()` (as the eval harness does by default) permanently poisons tensors against that even under a nested `enable_grad()`. Fixed by using `torch.no_grad()` in the RTC driving loop instead.

## What didn't work / was ruled out

- **More ACT training steps alone.** Loss dropped monotonically the whole time (10K→60K), but success rate did not — it peaked around 20-40K and was noisy/non-monotonic (one checkpoint dipped to 15%). Classic small-dataset overfitting signature; more steps on the same 140 demos is not the lever.
- **Domain randomization at eval time** — DR-eval vs non-DR-eval gave statistically identical success rates (~35%) for the same checkpoint. Consistent with the dataset itself containing no DR in its sim portion — the policy was never taught to be robust to it either way.
- **ACT temporal ensembling** — blends overlapping chunk predictions every step. Failed completely (0%) due to the `reset()` bug (stale ensembler state persisting across episodes); after the fix it worked but still underperformed (40%) simple truncated replanning (50%) — likely because ensembling trades reactivity for smoothness, and this task's failures are dominated by precise, time-critical grasp/release moments where reactivity matters more.
- **pi0.5 and GR00T locally** — hardware wall, not a design choice. pi0.5 needs 24GB+ VRAM to fine-tune; GR00T needs 16GB+ just for bare inference. Both exceed our 8GB card by 2-3x; only feasible via a rented cloud GPU.
- **RTC guidance_weight=0.5** — worse (30%) than both 1.0 (45%) and 2.0 (40%). The relationship is non-monotonic; 1.0 is a genuine sweet spot, not "lower is always better."

## Diagnostics

- **Failure mode:** of 10 failed ACT episodes (50% checkpoint, full video-reviewed), 6/10 *did* successfully grasp the vial (some multiple times) but still failed — i.e. most failures are a placement/release-precision problem, not a perception/reach/grasp problem.
- **Why RTC only applies to SmolVLA/pi0/pi0.5 and not ACT:** those three use a flow-matching (iterative denoising, diffusion-family) action head; RTC's guidance is literally diffusion inpainting applied to action sequences — it needs the multi-step denoising trajectory to inject a correction into partway through. ACT's CVAE decodes the whole chunk in one shot, so there's no trajectory to hook into.

## Reusable artifacts

- `humanoid_so101_vial_task/utils/lerobot_interface.py` — env-var-gated research knobs added: `ACT_N_ACTION_STEPS_OVERRIDE`, `ACT_TEMPORAL_ENSEMBLE_COEFF`, `RTC_EXECUTION_HORIZON`, `RTC_MAX_GUIDANCE_WEIGHT`. Both bugfixes above are in this file.
- `scripts/lerobot_eval_rtc.py` (scratch, not yet committed) — the RTC-driving eval script (`RTCDrivenPolicy` class). Robot-specific bits are isolated to calls through `LeRobotSO101Interface`; the RTC orchestration itself (queue management, chunk replanning, guidance) is robot-agnostic and portable to any embodiment with a flow-matching policy and a similar interface wrapper.

## Open threads / not pursued

- Filtering the 15 real episodes out of training (test whether sim-only training changes anything).
- Generating additional sim demos (scripted or teleop) — repo history shows a cuRobo scripted-demo generator was tried and explicitly abandoned in favor of real teleop; not revisited here.
- RTC with a simulated nonzero `inference_delay` (we used 0 throughout — sim is synchronous, so RTC's core "overlap real inference latency with execution" benefit was never actually exercised, only its guidance/blending side).
