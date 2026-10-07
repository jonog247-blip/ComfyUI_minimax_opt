"""MiniMax H3 benchmark node.

Times real denoising evaluations on the machine in front of you, through the
same path a generation takes: the guider packs the nested latent, the model's
``latent_shapes`` are set, the attention patches are live, and the model is
called once at a real sigma. The point is to make the two numbers that matter
measurable rather than assumed:

* seconds per evaluation, which turns the budget report into a real wall-clock
  estimate, and calibrates ``MiniMaxH3Budget``;
* the same figure with and without the sparse-attention optimiser, which is the
  only honest way to know what it buys on your GPU (it is a kernel-level win, so
  a 5070 Ti and a 4090 will not agree).

Run it on the shipped 14 step workflow twice, once with the optimiser node in
``off`` mode and once in ``balanced``: the difference is your speedup.
"""

from __future__ import annotations

import statistics
import time

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils

from . import h3_math as M
from .h3_sampler import sample_h3_flow


def _packed_tokens(samples):
    """Token count of the packed sequence, straight off the latent's own shapes."""
    streams = samples.unbind() if hasattr(samples, "unbind") else (samples,)
    video = streams[0]
    if video.dim() == 5:
        _, _, latent_t, height, width = video.shape
        frames = M.align_frame_count((int(latent_t) - 1) // 4 * 17 + 5)
        tokens = M.token_estimate(int(width) * 16, int(height) * 16, frames)
        return tokens["video_tokens"] + tokens["audio_tokens"], frames
    # a plain packed tensor: the last dim is the sequence
    return int(video.shape[-1]), 0


class MiniMaxH3Bench:
    """Times one denoising evaluation of the wired model and latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "positive": ("CONDITIONING", {
                    "tooltip": "The same conditioning your workflow uses. The text rows are part of "
                               "the packed sequence, so timing without them would flatter every number.",
                }),
                "latent": ("LATENT", {
                    "tooltip": "Wire the empty AV latent, so the shapes are exactly a real run's.",
                }),
                "sigma": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Noise level to evaluate at. 0.9 is representative of the body of a "
                               "generation; the tail costs the same.",
                }),
                "warmup": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1}),
                "repeats": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "measure"
    CATEGORY = "model/minimax"
    DESCRIPTION = ("Times one real MiniMax H3 denoising evaluation, reports seconds per step, "
                   "effective TFLOP/s and peak VRAM, and prints the value to put in the budget node.")

    def measure(self, model, positive, latent, sigma=0.9, warmup=2, repeats=3):
        samples = latent["samples"]
        tokens, frames = _packed_tokens(samples)
        flops = M.step_flops(tokens)

        guider = comfy.samplers.Guider_Basic(model)
        guider.set_conds(positive)
        sampler = comfy.samplers.KSAMPLER(sample_h3_flow, extra_options={})
        sigmas = torch.FloatTensor([float(sigma), 0.0])
        noise = torch.randn_like(samples) if not hasattr(samples, "unbind") \
            else samples.unbind()[0].new_zeros(samples.unbind()[0].shape)

        device = comfy.model_management.get_torch_device()
        timing = []
        for run in range(max(0, int(warmup)) + max(1, int(repeats))):
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
                if run == int(warmup):
                    torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            guider.sample(noise, samples, sampler, sigmas, denoise_mask=None,
                          callback=None, disable_pbar=True, seed=0)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if run >= int(warmup):
                timing.append(elapsed)

        peak = 0
        if torch.cuda.is_available() and device.type == "cuda":
            peak = torch.cuda.max_memory_allocated()

        best = min(timing)
        median = statistics.median(timing)
        effective = flops / (best * 1e12)
        lines = [
            "MiniMax H3 benchmark",
            "  device              : %s" % comfy.model_management.get_torch_device_name(device),
            "  packed tokens       : %d%s" % (tokens, "  (%d frames)" % frames if frames else ""),
            "  FLOPs per evaluation: %.2f TFLOP" % (flops / 1e12),
            "",
            "  warmup runs         : %d" % int(warmup),
            "  timed runs          : %d" % len(timing),
            "  best                : %.3f s" % best,
            "  median              : %.3f s" % median,
            "  all                 : %s" % ", ".join("%.3f" % t for t in timing),
        ]
        if peak:
            lines.append("  peak VRAM           : %.2f GB" % (peak / (1024 ** 3)))
        lines += [
            "",
            "  effective throughput: %.1f TFLOP/s" % effective,
            "  for the budget node : device_tflops 150, utilisation %.2f"
            % min(1.0, max(0.05, effective / 150.0)),
            "",
            "  This is one evaluation at sigma %.2f, so multiply by your step count for the",
            "  sampling cost. The first step of a real run also pays prompt encoding and the",
            "  last one pays both VAE decodes, neither of which is in this number." % float(sigma),
        ]
        return ("\n".join(lines),)
