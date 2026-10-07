# ComfyUI-MiniMaxH3-Optim

Five nodes for ComfyUI's native MiniMax H3 audio-video model: a scheduler and a
sampler designed as a pair, a one-stop optimiser, a budget planner, and the
benchmark that keeps the other four honest.

Nothing here reimplements the model. The nodes take the `MODEL`, `CONDITIONING`
and `LATENT` the stock MiniMax H3 nodes already produce, and they drop into the
shipped workflow in place of `BasicScheduler` / `KSamplerSelect`.

## Install

```
cd ComfyUI/custom_nodes
git clone <this repo>            # or copy the ComfyUI-MiniMaxH3-Optim folder here
```

Restart ComfyUI. Five nodes appear; nothing else in your install changes. The
pack imports nothing outside the standard library and ComfyUI itself.

Requires the native H3 support in ComfyUI (0.30 or newer; developed against
0.39.0) and the `EmptyMiniMaxH3LatentAV` / `MiniMaxH3ImageToVideo` nodes that
ship with it.

## The problem this solves

H3 runs on one sigma schedule that both streams share, with the audio stream
scaled onto it by a factor of four. ComfyUI's `simple` schedule walks a uniform
grid in the model's internal time and stops at `sigma = shift / (steps + shift - 1)`.
For the 20 step H3 workflow that is:

| | video | audio |
|---|---|---|
| last sigma the model sees | 0.3871 | 0.1364 |
| then one step straight to | 0.0 | 0.0 |

The audio branch is handed that final step at a noise level four times higher
than the video branch's, and it is the branch with the smallest margin: the
audio latent is 244 tokens of the 22 420 in the pack, so a bad landing there is
both audible and cheap to fix.

`MiniMax H3 Scheduler` keeps the checkpoint's trained body exactly as it is and
spends the last few steps on a band that descends in the *audio* stream's own
sigma, so no single step asks either stream to cross a large gap:

| 14 steps, mode `h3_balanced` | video | audio |
|---|---|---|
| last sigma the model sees | 0.0552 | 0.0144 |
| terminal band | 4 steps, each x1.90 in audio space | |

Same step count, same model, same workflow. The audio branch finishes where the
model's own sigma table starts instead of 4x above it.

## The five nodes

| node | what it does | why you want it |
|---|---|---|
| **MiniMax H3 Scheduler** | `MODEL → (SIGMAS, STRING)` | the schedule above, with a report of every step |
| **MiniMax H3 Sampler** | `→ SAMPLER` | one model evaluation per step, closed-form integration of the flow ODE |
| **MiniMax H3 Optimiser** | `MODEL → (MODEL, STRING)` | block-sparse attention with H3 sink protection |
| **MiniMax H3 Budget** | `→ (INT, INT, STRING)` | snaps a duration to the 17k+5 frame grid, estimates tokens, FLOPs and time |
| **MiniMax H3 Bench** | `MODEL, CONDITIONING, LATENT → STRING` | times one real denoising evaluation on your GPU |

All `STRING` outputs are meant for a `Preview as Text` node; the reports are
where the nodes explain what they actually did.

## Recommended pairings

The three tiers the workflow already implies, with the nodes set to match.
Step counts are the workflow's, the presets are these nodes'.

| | Draft | Balanced | Max quality |
|---|---|---|---|
| steps | 8 (turbo LoRA) | 14 | 20–25 |
| scheduler mode | `h3_turbo8` | `h3_balanced` | `h3_max` |
| sampler mode | `balanced` | `balanced` | `max` |
| optimiser mode | `aggressive` | `balanced` | `balanced` or `audio_safe` |

`h3_max` grows its terminal band until the `sigma_min` floor is reachable
inside `max_gap`, so at 20 steps it takes 13 body steps and a 7 step band and
lands at sigma_a 0.0038. Switch `auto_tail` on and the node tells you in the
report how many tail steps it took and why.

`MiniMax H3 Scheduler` also offers `simple_stock`, which reproduces ComfyUI's own
schedule exactly. Put it side by side with `h3_balanced` and the reports are a
direct A/B of the schedule alone.

## Why the sampler is different

The flow ODE has an exact solution:

```
dx/dsigma = (x - x0)/sigma
x(s_next) = s_next * ( x_n/s_n - integral_{s_n}^{s_next} x0(s)/s^2 ds )
```

The sampler replaces `x0(s)` with a polynomial fitted through the last few model
evaluations, so every step is closed form. Two consequences that matter here:

* it is **exact for a constant clean prediction whatever the step size**, which
  is what makes the scheduler's compressed terminal band safe to integrate;
* it never injects noise, so the packed audio latent keeps no hiss or flutter
  from a stochastic step.

The video and audio slices are integrated separately, each against its own
sigma, from the same single model evaluation.

## What the solver lab measured

`tools/h3_solver_lab.py` runs the sampler and the schedule against analytic flow
problems with an independent RK4 reference. Two problems: a gently curved clean
prediction (`x0 = 1 + 0.6 s^1.5`, "video-like drift") and a strongly curved one
(`1 + 1.2 s^2`). Error is relative, at 12 steps, on the H3 schedule:

| sampler mode | video-like body | video-like final | aggressive body | aggressive final |
|---|---|---|---|---|
| `euler_reference` | 7.55e-2 | 2.32e-2 | 1.18e-1 | 1.57e-2 |
| `linear` | 2.14e-2 | 1.76e-2 | 6.80e-2 | 2.76e-2 |
| `balanced` | **4.22e-3** | **6.26e-3** | **5.77e-5** | **2.22e-16** |
| `max` | **1.85e-3** | **4.22e-3** | **5.77e-5** | **4.44e-16** |

*body* = the state at the last non-zero sigma, so it is the integrator's own
error. *final* = after the step that lands on sigma = 0, which is where the
terminal rule shows up. Ranking the two effects: at 8 steps and up the terminal
rule moves the error by more than the choice of integrator does.

Three findings changed the defaults rather than decorating them:

* **The step guard and curvature damping are off.** Both sounded like prudent
  hardening. Both measurably lose: demoting the order on a wide step costs 50x
  on the aggressive problem (5.8e-5 to 3.2e-3) because wide steps are exactly
  where the higher order earns its keep, and damping the curvature costs 2.7x
  (4.2e-3 to 1.1e-2). They are still there as widgets, off, for hand-built
  schedules that behave nothing like these.
* **A cubic model is never worse than a quadratic** on either problem, at no
  extra cost per evaluation, so `max` uses it. `balanced` stays on the quadratic
  because a cubic through four noisy network predictions is more sensitive to
  that noise than a quadratic through three, and the lab's clean analytic curves
  cannot see that risk.
* **`terminal_extrap` stays at 1.0**, landing on the fitted model's clean
  prediction at sigma = 0 rather than the last evaluated one. It is the single
  biggest error term at low step counts, and 1.0 wins on both problems.

## Tools

All three run without a GPU, and the first two run without torch:

```
python3 tools/h3_node_smoke_test.py                  # loads the pack behind stubs, validates every node schema
python3 tools/h3_node_smoke_test.py --report h3_balanced 14   # prints the exact report the scheduler node gives you
python3 tools/h3_solver_lab.py                       # the numerical validation above, ~5 s
```

`MiniMax H3 Bench` needs the GPU and a wired model: it reports seconds per
evaluation, effective TFLOP/s and peak VRAM, and prints the numbers to put into
`MiniMax H3 Budget`. Run it once with the optimiser in `off` and once in
`balanced` to see what sparse attention actually buys on your card, which is the
only way to know — it is a kernel-level win and every GPU disagrees.

## What this pack does not do

* It does not reduce the number of model evaluations below the step count you
  ask for. Step skipping (feature forecasting) is a different technique with a
  different risk profile; this pack spends each step it is given better.
* It does not touch quantisation or the model weights. It works with the int8
  convrot checkpoint and with fp16/fp8 ones.
* Batch size is 1; that is a limit of the H3 model itself.

## Example workflow

`examples/h3_pack_diagnostics.json` loads the int8 convrot checkpoint, patches it
with the optimiser and the scheduler, and prints all three reports through
`Preview as Text` nodes. It generates nothing, so it is a fast way to confirm
your install: the optimiser report says which sparse-attention path was
selected, the scheduler report shows the whole sigma table with the terminal
band, and the budget report gives the token and FLOP cost of your canvas.

To use the pack in a real generation, take the shipped I2V workflow and make
three changes:

1. Replace `KSamplerSelect` with **MiniMax H3 Sampler** (mode `balanced`) and
   wire its `SAMPLER` output where the old one went.
2. Replace `BasicScheduler` with **MiniMax H3 Scheduler** (mode `h3_balanced`,
   steps from your existing switch) and wire `sigmas` the same way. Leave
   `MiniMax H3 Scheduler`'s second output on a `Preview as Text` node the first
   time you run it.
3. Insert **MiniMax H3 Optimiser** (mode `balanced`) between the model side of
   the switch and the guider, and put its report on another `Preview as Text`.

Nothing else needs rewiring: the nodes take and return the same types as the
stock ones, and the sampler runs through `SamplerCustomAdvanced` exactly like
`res_multistep` does.

## Documentation

* `docs/fedora-5070ti-guide.md` — the whole machine, not just the nodes: driver,
  PyTorch build, attention kernels, launch flags, model files, where the time
  goes, and the Windows 11 comparison.
