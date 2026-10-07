# MiniMax H3 on a 7950X3D / RTX 5070 Ti 16 GB / 64 GB DDR5 box

Fedora first. This is the reference configuration the node pack is tuned
against, and every step below is one you can check with a command rather than
believe on faith. Windows 11 stays useful as an A/B control; the last section
covers what to expect from it.

The order matters: driver, then torch, then attention kernels, then ComfyUI
flags, then the workflow. Skipping ahead means measuring the wrong thing.

## 1. Driver

Blackwell needs a 570 branch driver or newer, and 16 GB of VRAM means you want
the driver's own memory accounting to be accurate, which is a reason to stay on
RPMFusion's build rather than a hand-rolled one.

```
sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
                 https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda nvidia-settings
sudo dnf install kernel-devel-$(uname -r) kernel-headers-$(uname -r)   # if the module does not build
```

Check, in this order:

```
nvidia-smi                          # driver version, 16384MiB, "RTX 5070 Ti"
nvidia-smi --query-gpu=compute_cap --format=csv     # must print 12.0
glxinfo | grep "OpenGL renderer"    # or check that the desktop is not on llvmpipe
```

If `compute_cap` is not 12.0, the kernel module and the userspace libraries are
from different branches; `akmods --force` then reboot.

Notes that save time later:

* Secure Boot: the akmod builds an unsigned module. Either enrol the key
  (`sudo kmodgenca -a`, `mokutil --import /etc/pki/akmods/certs/public_key.der`)
  or disable Secure Boot. The module silently failing to load while the screen
  still works is the classic 30-minute confusion.
* Wayland is fine, but if you hit the black-screen-on-resume or VRAM-not-freed
  class of bugs, `nvidia-drm.modeset=1` (default now) plus a newer driver is the
  fix, not a kernel downgrade.

## 2. Python and PyTorch

Python 3.12 (3.11 also works). A venv inside the ComfyUI directory keeps the
system's tooling untouched:

```
cd ~/ComfyUI
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Older stable cu124/cu121 wheels have no sm_120 kernels and will either refuse to
run or drop to a very slow fallback, so verify rather than assume:

```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(), torch.cuda.get_device_name(0))"
```

Expected: `2.x.x+cu128`, `(12, 0)`, `NVIDIA GeForce RTX 5070 Ti`. Then a real
matmul, which is the only test that proves the kernels exist:

```
python - <<'EOF'
import torch, time
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3): c = a @ b
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(10): c = a @ b
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 10
print("%.1f TFLOP/s bf16" % (2 * 8192**3 / dt / 1e12))
EOF
```

A 5070 Ti should land near 100-140 TFLOP/s on this. If you see 15, you are on a
fallback kernel path and everything downstream will be slow for reasons that
look like ComfyUI's fault.

## 3. Attention kernels

H3's attention is about half of the FLOPs at 1344×768 (see the budget node), so
this is the single biggest lever after the step count.

```
pip install triton                    # required by most attention kernels below
pip install sageattention             # SA3 for Blackwell; check `pip show sageattention`
pip install flash-attn --no-build-isolation   # optional; sm_120 support is recent, see below
```

Practical warnings, all from the field rather than from theory:

* **SageAttention on Windows** is still 1.0.6/Triton-based for sm_120, worth
  roughly 2.1x, and not the Blackwell-optimised path. On Linux you can get the
  real thing. This is the main reason Linux is the primary install here.
* **Verify it on H3 before making it the default.** `--use-sage-attention` has
  produced black output on some Blackwell wheels for other model families. The
  test is one short generation, decoded and looked at, not a smoke test that the
  process did not crash.
* If the built-in flag misbehaves, the KJNodes `Patch Sage Attention` node lets
  you apply the same kernel per-model, which is easier to A/B than a restart.
* `--use-ck-attention` (comfy-kitchen) is the other path; the block-sparse
  attention in the optimiser node uses comfy-kitchen's kernels when available.

The node pack's optimiser is a *different* optimisation from these: it lowers
the attention work rather than making each unit faster. Run them together.

## 4. ComfyUI launch flags for 16 GB

Start from this and change one thing at a time:

```
python main.py \
  --use-sage-attention \
  --enable-triton-backend \
  --cache-ram 16 --high-ram \
  --fast-disk \
  --vram-headroom 1024 \
  --preview-method none
```

Why each one, and when to drop it:

* `--use-sage-attention` — the kernel win, if it verifies clean on your card.
* `--enable-triton-backend` — Triton for the ops that support it; also the
  backend SageAttention's fallback path needs.
* `--cache-ram 16 --high-ram` — with 64 GB you can hold the 21 GB int8 DiT plus
  the 15.7 GB text encoder across runs instead of re-reading them. The text
  encoder alone is a large part of the per-run startup; `--high-ram` tells
  ComfyUI to be generous with it.
* `--fast-disk` — NVMe-backed staging of model files.
* `--vram-headroom 1024` — keeps ComfyUI from filling the last gigabyte, which
  is where driver-side allocation failures produce confusing OOMs on a 16 GB
  card with a desktop attached.
* `--preview-method none` — previews decode latents mid-run and cost real time
  on a long generation. Turn it back on when iterating on a prompt.

Keep in reserve, with the symptom that calls for them:

| flag | use it when |
|---|---|
| `--reserve-vram 0.5` | another app (browser with video) shares the GPU |
| `--async-offload` | you see the CPU stall at the start of each step |
| `--disable-cuda-graphs` | a kernel or patch produces wrong output with graphs on |
| `--disable-comfy-compiler` | compile times dominate short runs, or a crash mentions inductor |

Do not set `--lowvram` or `--novram`: at 1344×768 / 124 frames the model fits in
16 GB with room for the latents, and forcing offload turns a 20-second step into
a multi-minute one.

## 5. Model files

From the official pruned sets, in the standard folders:

| file | folder | size |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` | ~21 GB |
| `qwen3vl_32b_nvfp4_awq.safetensors` (text encoder) | `models/text_encoders/` | ~15.7 GB |
| video VAE (fp16) | `models/vae/` | ~5.2 GB |
| audio VAE (fp32) | `models/vae/` or the audio vae folder the loader expects | ~605 MB |

Notes specific to this box:

* 64 GB of RAM holds the DiT and the text encoder simultaneously, so the first
  run pays the load and later runs reuse it. Verify with
  `--cache-ram 16 --high-ram` that the load time after the first run drops.
* Do not convert the int8 convrot checkpoint to bf16 "for speed": 21 GB of int8
  weights with bf16 activations is the intended configuration and the one the
  quantisation was validated in.
* 768p is the trained and locally supported ceiling; the 2K path is the hosted
  API. Do not talk yourself into upscaling the canvas past `MAX_PIXELS`
  (768×1344) — you will spend time and VRAM and get worse motion.

## 6. Where the time actually goes

At 1344×768, 5 seconds (124 frames), the packed sequence is 37 710 tokens and
one denoising evaluation is about 2.5 TFLOP. The budget node prints this for any
canvas and length you like; the 73-frame number below is from the solver lab:

| | value |
|---|---|
| packed tokens | 22 420 (video 22 176 + audio 244) |
| FLOPs per evaluation | 1 498 TFLOP |
| of which attention | 48% |
| at 150 TFLOP/s effective | 28.5 s per step |

So a 14 step run is ~7 minutes of DiT time at 35% utilisation, before prompt
encoding and the two VAE decodes. Two implications:

* Ceiling first: measure with the bench node, then decide. If you are getting
  much worse than this, the problem is the kernel path (section 3), not the
  sampler.
* Then buy steps back with the *sampler*: at 14 steps and above the pack spends
  its steps well, so once the kernels are right the next gain is `h3_max` at the
  same step count rather than more steps.

## 7. Verification checklist

Run these in order on a fresh install. Each has a number to compare against.

1. `nvidia-smi` shows 16384 MiB and `compute_cap` 12.0.
2. The bf16 matmul in section 2 gives >100 TFLOP/s.
3. `python3 custom_nodes/ComfyUI-MiniMaxH3-Optim/tools/h3_node_smoke_test.py`
   loads five nodes and prints "all checks passed".
4. `python3 .../tools/h3_solver_lab.py` finishes in ~5 seconds.
5. A short 8 step generation with `h3_turbo8` + `balanced` decodes with audible,
   clean audio and no NaN warnings in the console.
6. `MiniMax H3 Bench` reports seconds per evaluation; compare against the
   1 498 TFLOP figure to get your effective TFLOP/s.
7. Bench again with the optimiser in `off` — the difference is the sparse
   attention speedup on your card.
8. A/B the schedule: run the same seed and prompt with `simple_stock`, then with
   `h3_balanced`. Listen to the last second of audio. This is the change the
   pack exists for and it is audible.

## 8. Windows 11 as the A/B control

Keep the Windows venv, but know what it measures:

* Windows reserves roughly 0.7-4 GB of VRAM for the desktop, against ~140 MB on
  a lean Linux desktop. On a 16 GB card that is 5-25% of the budget.
* sm_120 SageAttention on Windows is the Triton fallback; the stack that works
  best there is `triton-windows` + `torch.compile` + SDPA, and it still leaves
  performance on the table.
* PyTorch stable wheels for Windows carry sm_120 kernels from cu128 onward; the
  nightly cu130 line is where the newest ones land.

Use it as a control, not as the target: run the same workflow on both, compare
the bench node's seconds-per-evaluation, and expect Linux to win on both VRAM
available and per-step time. If you need Windows for a specific node or model,
that is a reason to keep it — not a reason to move the whole project there.

## 9. Things that look like optimisations and are not

* **Lowering the step count below ~8** on the base model. The terminal band
  cannot reach a sensible floor with fewer steps, which is what the scheduler's
  `auto_tail` report tells you explicitly. Use the 8 step turbo LoRA instead.
* **Raising `shift_video`.** The checkpoint was trained at 12; the DiT inverts
  the schedule to its internal grid and a mismatched shift degrades prompt
  following before it helps anything.
* **`--lowvram` / `--novram`.** See section 4.
* **bf16 conversions of the int8 checkpoint.** See section 5.
* **More samplers at the same step count.** The lab's ranking puts the terminal
  rule and the schedule ahead of the integrator, and this pack is already first
  or second on the integrator for both test problems.
