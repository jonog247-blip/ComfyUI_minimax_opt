"""MiniMax H3 flow sampler.

One model evaluation per step, second order in the body, and a terminal step
that lands on the extrapolated clean prediction instead of jumping to zero from
whatever sigma the schedule happened to stop at.

The solver is built on the exact solution of the flow ODE rather than on a
Taylor expansion of it:

    dx/dsigma = (x - x0)/sigma
    x(s_next) = s_next * ( x_n/s_n - integral_{s_n}^{s_next} x0(s)/s^2 ds )

``x0(s)`` is replaced by a polynomial fitted through the last two or three
model evaluations, so the whole step is closed form. Two consequences that
matter on a 33B audio-video model:

* it is exact when the model's clean prediction is constant, and very good when
  it is locally smooth, so the compressed terminal band the H3 scheduler builds
  is safe to integrate in one larger step;
* it never injects noise, which keeps the packed audio latent free of the
  hiss and flutter that stochastic samplers add to the stereo stream.

Everything here is a plain k-diffusion style sampler function, so it runs
through ``comfy.samplers.KSAMPLER`` and inherits the framework's mask handling,
noise scaling, callbacks and progress reporting instead of reimplementing them.
"""

from __future__ import annotations

import logging
import math

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils

from . import h3_math as M

SAMPLER_MODES = ["balanced", "max", "linear", "euler_reference", "custom"]

PRESETS = {
    # order, curvature_damping, audio_order_boost, terminal_extrap, guard
    "balanced": dict(order=2, curvature_damping=1.0, audio_order_boost=0, terminal_extrap=1.0),
    "max": dict(order=3, curvature_damping=1.0, audio_order_boost=0, terminal_extrap=1.0),
    "linear": dict(order=1, curvature_damping=1.0, audio_order_boost=0, terminal_extrap=1.0),
    "euler_reference": dict(order=0, curvature_damping=1.0, audio_order_boost=0, terminal_extrap=0.0),
}


def _unwrap(obj, depth=3):
    """Walk the sampler's wrapper chain looking for an attribute."""
    seen = obj
    for _ in range(depth):
        if seen is None:
            return None
        yield seen
        seen = getattr(seen, "inner_model", None)


def _find_attr(model, name):
    for obj in _unwrap(model):
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _model_sampling(model):
    for obj in _unwrap(model):
        ms = getattr(obj, "model_sampling", None)
        if ms is not None:
            return ms
    return None


def _shifts(model, extra_args):
    """The video/audio flow shifts actually in use for this run."""
    options = (extra_args.get("model_options") or {}).get("transformer_options") or {}
    shift_v = options.get("minimax_h3_sigma_shift_video")
    shift_a = options.get("minimax_h3_sigma_shift_audio")
    if shift_v is None or shift_a is None:
        ms = _model_sampling(model)
        if shift_v is None:
            shift_v = getattr(ms, "shift", None)
        if shift_a is None:
            shift_a = getattr(ms, "audio_shift", None)
    return float(shift_v if shift_v is not None else M.SHIFT_VIDEO), \
        float(shift_a if shift_a is not None else M.SHIFT_AUDIO)


def _video_element_count(model, x):
    """Elements of the flat pack that belong to the video stream, or None.

    The guider packs the nested latent into one flat tensor, video first, so
    the split is the product of the video latent's non-batch dims.
    """
    shapes = _find_attr(model, "latent_shapes")
    if not shapes or len(shapes) < 2:
        return None
    try:
        count = math.prod(int(d) for d in shapes[0][1:])
    except (TypeError, ValueError):
        return None
    total = x.numel() // max(x.shape[0], 1)
    if count <= 0 or count >= total:
        return None
    return count


def _effective_order(sigmas, i, requested, step_ratio_limit, step_guard=False):
    """The order rule, via h3_math so the solver lab tests the same code.

    ``i`` is the index of the step about to be taken, so the span being judged is
    this step's, compared against the previous step's. The step that lands on
    sigma = 0 has no meaningful span and keeps the requested order, because
    extrapolating the fitted model there is the whole point of it.
    """
    order = int(requested)
    if not step_guard or order < 2 or step_ratio_limit <= 1.0 or i < 1:
        return order
    sigma_n, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
    if sigma_next <= 0.0 or float(sigmas[i - 1]) <= 0.0 or sigma_n <= 0.0:
        return order
    return M.order_for_step(math.log(sigma_n / sigma_next),
                            math.log(float(sigmas[i - 1]) / sigma_n),
                            order, step_ratio_limit)


def _step_slice(x_slice, x0_n, sigma_n, sigma_next, coeffs, terminal_extrap):
    """One closed-form step for a slice of the pack, with no model evaluation."""
    if sigma_next <= 0.0:
        if terminal_extrap == 0.0:
            return x0_n
        clean = coeffs[0]  # the fitted polynomial evaluated at sigma = 0
        return x0_n + terminal_extrap * (clean - x0_n)
    drop = M.antiderivative(sigma_next, coeffs) - M.antiderivative(sigma_n, coeffs)
    return (x_slice / sigma_n - drop) * sigma_next


def _euler_slice(x_slice, x0_n, sigma_n, sigma_next):
    if sigma_next <= 0.0:
        return x0_n
    return x_slice + (sigma_next - sigma_n) * (x_slice - x0_n) / sigma_n


def _coefficients(hist_sigmas, hist_x0, order, damping):
    if len(hist_x0) < 2:
        return [hist_x0[0]]
    return M.polyfit_sigma_model(hist_sigmas, hist_x0, order, damping)


def is_finite(coeffs):
    return all(c == c and abs(c) != float("inf") for c in coeffs)


@torch.no_grad()
def sample_h3_flow(model, x, sigmas, extra_args=None, callback=None, disable=None,
                   order=2, curvature_damping=1.0, audio_order_boost=0,
                   terminal_extrap=1.0, step_ratio_limit=2.0, step_guard=False,
                   guard=True, debug=False, **kwargs):
    """k-diffusion style sampler function: one NFE per step, no noise injection.

    ``model`` is the framework's X0 wrapper, so its output is the denoised
    clean prediction for the whole packed latent, video rows first.
    """
    extra_args = {} if extra_args is None else extra_args
    if sigmas is None or len(sigmas) < 2:
        return x

    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    pbar = comfy.utils.ProgressBar(total_steps) if not disable else None

    n_video = _video_element_count(model, x)
    shift_v, shift_a = _shifts(model, extra_args)
    if debug:
        _log_schedule(sigmas, n_video, shift_v, shift_a, order, audio_order_boost)

    hist_sigmas = []
    hist_x0_v = []
    hist_x0_a = []

    for i in range(total_steps):
        comfy.model_management.throw_exception_if_processing_interrupted()

        sigma_n = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        if sigma_n <= 0.0:
            break

        denoised = model(x, sigmas[i] * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i],
                      "denoised": denoised})

        x0_v = denoised if n_video is None else denoised[..., :n_video]
        x0_a = None if n_video is None else denoised[..., n_video:]

        eff_order = _effective_order(sigmas, i, order, step_ratio_limit, step_guard)
        coeffs_v = _coefficients(hist_sigmas, hist_x0_v, eff_order, curvature_damping) \
            if hist_x0_v else [x0_v]
        if x0_a is None:
            x_next = _advance(x, x0_v, sigma_n, sigma_next, coeffs_v, terminal_extrap)
        else:
            audio_order = min(eff_order + int(audio_order_boost), 3)
            audio_order = max(audio_order, eff_order)
            coeffs_a = _coefficients(hist_sigmas, hist_x0_a, audio_order, curvature_damping) \
                if hist_x0_a else [x0_a]
            if not is_finite(coeffs_v) or not is_finite(coeffs_a):
                coeffs_v, coeffs_a = [x0_v], [x0_a]
            x_next = _advance_split(x, x0_v, x0_a, sigma_n, sigma_next,
                                    coeffs_v, coeffs_a, terminal_extrap)

        if guard and not bool(torch.isfinite(x_next).all()):
            logging.warning("MiniMaxH3 sampler: non-finite step at sigma %.5f, "
                            "falling back to an Euler step", sigma_n)
            x_next = _advance_euler(x, denoised, sigma_n, sigma_next, n_video)

        x = x_next
        hist_sigmas.insert(0, sigma_n)
        hist_x0_v.insert(0, x0_v)
        if x0_a is not None:
            hist_x0_a.insert(0, x0_a)

        if pbar is not None:
            pbar.update(1)

    return x


def _advance(x, x0, sigma_n, sigma_next, coeffs, terminal_extrap):
    return _step_slice(x, x0, sigma_n, sigma_next, coeffs, terminal_extrap)


def _advance_split(x, x0_v, x0_a, sigma_n, sigma_next, coeffs_v, coeffs_a, terminal_extrap):
    n_video = x0_v.shape[-1]
    video = _step_slice(x[..., :n_video], x0_v, sigma_n, sigma_next, coeffs_v, terminal_extrap)
    audio = _step_slice(x[..., n_video:], x0_a, sigma_n, sigma_next, coeffs_a, terminal_extrap)
    return torch.cat((video, audio), dim=-1)


def _advance_euler(x, denoised, sigma_n, sigma_next, n_video):
    if n_video is None:
        return _euler_slice(x, denoised, sigma_n, sigma_next)
    video = _euler_slice(x[..., :n_video], denoised[..., :n_video], sigma_n, sigma_next)
    audio = _euler_slice(x[..., n_video:], denoised[..., n_video:], sigma_n, sigma_next)
    return torch.cat((video, audio), dim=-1)


def _log_schedule(sigmas, n_video, shift_v, shift_a, order, audio_order_boost):
    grid = M.schedule_audio_grid([float(s) for s in sigmas], shift_v, shift_a)
    lines = ["MiniMaxH3 sampler: %d steps, order %d (+%d for the audio slice), %s"
             % (len(sigmas) - 1, order, audio_order_boost,
                "AV split detected" if n_video else "single stream")]
    for i in range(len(sigmas) - 1):
        v = float(sigmas[i])
        if v <= 0.0:
            break
        next_v = float(sigmas[i + 1])
        vg = v / next_v if next_v > 0.0 else float("inf")
        ag = grid[i] / grid[i + 1] if grid[i + 1] > 0.0 else float("inf")
        lines.append("  %3d  sigma_v %9.5f  sigma_a %9.5f  step x%6.3f  audio x%6.3f"
                     % (i, v, grid[i], vg, ag))
    logging.info("\n".join(lines))


class MiniMaxH3SamplerSelect:
    """Builds the sampler object; feeds SamplerCustomAdvanced or SamplerCustom."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (SAMPLER_MODES, {
                    "default": "balanced",
                    "tooltip": (
                        "balanced: a local quadratic x0 model with the terminal extrapolation. The "
                        "default, and the best all-rounder in the solver lab. "
                        "max: cubic model. Measured never worse than balanced and up to 2x better on "
                        "a curved-drift problem, at no extra cost per step. "
                        "linear: a straight-line x0 model, for when you want the conservative option. "
                        "euler_reference: plain Euler with a plain landing, i.e. what the stock "
                        "sampler does, for A/B runs. "
                        "custom: use the widgets below."),
                }),
            },
            "optional": {
                "order": ("INT", {
                    "default": 2, "min": 0, "max": 3,
                    "tooltip": "Only used when mode = custom. Degree of the local x0 model: 0 is "
                               "plain Euler, 1 a straight line through two samples, 2 a quadratic "
                               "through three, 3 a cubic through four. Costs nothing extra per step.",
                }),
                "curvature_damping": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only used when mode = custom. 1.0 is an exact local fit and is what "
                               "the lab measures best overall; below 1.0 shrinks the fitted model "
                               "toward a straight line, which only ever helped when the true x0 "
                               "drifted much more slowly than the fit could describe.",
                }),
                "audio_order_boost": ("INT", {
                    "default": 0, "min": 0, "max": 1,
                    "tooltip": "Only used when mode = custom. Add one to the audio slice's order. It "
                               "is free to compute, but the lab finds no case where the extra order "
                               "helps the audio specifically, so it is off.",
                }),
                "terminal_extrap": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only used when mode = custom. On the last step the solver lands on "
                               "the local model's x0 at sigma = 0 (1.0) rather than on the last "
                               "evaluated x0 (0.0). The lab measures 1.0 as the more accurate of the "
                               "two; drop it if a specific checkpoint prefers the plain prediction.",
                }),
                "step_ratio_limit": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Only used when step_guard is on: demote one order when a step is this "
                               "many times wider in log-sigma than the step before it.",
                }),
                "step_guard": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Demote the order on a step that is far wider than the one before it. "
                               "Off by default because the lab measures it as a net loss on both test "
                               "problems: wide steps are where the higher order earns its keep. Kept "
                               "for hand-built schedules with genuinely wild jumps.",
                }),
                "guard": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check every step for non-finite values and fall back to a plain Euler "
                               "step if one appears. Cheap insurance, worth leaving on.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log the per-step sigma_v / sigma_a table and the step ratios once per run.",
                }),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = ("MiniMax H3 flow sampler: one model evaluation per step, a local polynomial "
                   "model of the clean prediction integrated in closed form, the audio slice carried "
                   "at its own sigma, and a terminal step that lands on the extrapolated clean "
                   "prediction rather than jumping to zero from wherever the schedule stopped.")

    def get_sampler(self, mode="balanced", order=2, curvature_damping=1.0,
                    audio_order_boost=0, terminal_extrap=1.0, step_ratio_limit=2.0,
                    step_guard=False, guard=True, debug=False):
        if mode == "custom":
            options = dict(order=int(order), curvature_damping=float(curvature_damping),
                           audio_order_boost=int(audio_order_boost),
                           terminal_extrap=float(terminal_extrap))
        else:
            options = dict(PRESETS.get(mode) or PRESETS["balanced"])
        options["step_ratio_limit"] = float(step_ratio_limit)
        options["step_guard"] = bool(step_guard)
        options["guard"] = bool(guard)
        options["debug"] = bool(debug)
        return (comfy.samplers.KSAMPLER(sample_h3_flow, extra_options=options),)
