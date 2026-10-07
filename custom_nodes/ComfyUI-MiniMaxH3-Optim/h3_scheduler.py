"""MiniMax H3 schedule node.

Drop-in replacement for ``BasicScheduler``: same first four inputs in the same
order, so it can be swapped into an existing H3 workflow without rewiring, and
it reads the flow shifts off the model so chaining the stock
``ModelSamplingMiniMaxH3`` node is picked up automatically.

Why this exists at all is in ``h3_math``: the shipped ``simple`` schedule stops
at sigma_v = 12/(steps+11), and because the audio stream runs four times lower,
whatever it stops at is what the audio decoder is handed. At 20 steps that is
sigma_a = 0.136. This node replaces the last few sigmas with a band that steps
down in audio space, so the same step budget finishes the audio near clean
without ever asking either stream to cross a large gap in one evaluation.
"""

from __future__ import annotations

import torch

from . import h3_math as M

SCHEDULE_MODES = ["h3_balanced", "h3_draft", "h3_max", "h3_audio_max",
                  "h3_turbo8", "h3_base20", "simple_stock", "custom"]

# terminal_steps, sigma_min, structure_hold, max_gap, auto_tail
SCHEDULE_PRESETS = {
    # the everyday setting: four terminal steps, body keeps the trained profile
    "h3_balanced": dict(terminal_steps=4, sigma_min=0.02, structure_hold=1.0,
                        max_gap=1.9, auto_tail=False),
    # for 6-10 step drafts, where the body needs the steps more than the tail does
    "h3_draft": dict(terminal_steps=3, sigma_min=0.02, structure_hold=1.0,
                     max_gap=2.0, auto_tail=False),
    # spend whatever the budget allows on the terminal band, and keep every step tight
    "h3_max": dict(terminal_steps=5, sigma_min=0.015, structure_hold=0.9,
                   max_gap=1.75, auto_tail=True),
    # the same, with the audio constraint set hard: no step may move sigma_a by more than 1.5x
    "h3_audio_max": dict(terminal_steps=5, sigma_min=0.015, structure_hold=0.9,
                         max_gap=1.5, auto_tail=True),
    # aligned with the 8-step turbo LoRA the official workflow ships with
    "h3_turbo8": dict(terminal_steps=3, sigma_min=0.03, structure_hold=1.0,
                      max_gap=2.0, auto_tail=False),
    # a conservative change to the stock 20-step base-model path
    "h3_base20": dict(terminal_steps=3, sigma_min=0.03, structure_hold=1.0,
                      max_gap=2.0, auto_tail=False),
    # exactly ComfyUI's 'simple' schedule, for A/B comparison
    "simple_stock": None,
}


class MiniMaxH3Scheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Connect the H3 model. Its model_sampling carries the video and audio "
                               "flow shifts, which this node reads.",
                }),
                "scheduler": (SCHEDULE_MODES, {
                    "default": "h3_balanced",
                    "tooltip": (
                        "h3_balanced: 4 terminal steps, native trained body. The default. "
                        "h3_draft: 3 terminal steps, for 6-10 step runs. "
                        "h3_max: auto-sized terminal band, tighter gaps, 18+ steps. "
                        "h3_audio_max: same, with the audio step constraint set to 1.5x. "
                        "h3_turbo8: sized for the official 8-step turbo LoRA. "
                        "h3_base20: a small change to the stock 20-step base path. "
                        "simple_stock: ComfyUI's own simple schedule, for comparison. "
                        "custom: use the explicit values below."),
                }),
                "steps": ("INT", {"default": 14, "min": 1, "max": 1000, "step": 1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "terminal_steps": ("INT", {
                    "default": 4, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Only used when scheduler = custom. Steps spent on the terminal band.",
                }),
                "sigma_min": ("FLOAT", {
                    "default": 0.02, "min": 0.005, "max": 0.5, "step": 0.005,
                    "tooltip": "Only used when scheduler = custom. The sigma floor the terminal band aims "
                               "for. The audio stream ends at roughly a quarter of it. Stay at or above "
                               "0.012 to remain inside the model's own sigma table.",
                }),
                "structure_hold": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Only used when scheduler = custom. Exponent on the body's time grid: "
                               "above 1.0 holds more steps at high sigma where layout is decided, below "
                               "1.0 packs the body more tightly near its end.",
                }),
                "max_gap": ("FLOAT", {
                    "default": 1.9, "min": 1.05, "max": 4.0, "step": 0.05,
                    "tooltip": "Only used when scheduler = custom. Largest relative move a single "
                               "terminal step may make to the audio stream's own sigma.",
                }),
                "auto_tail": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only used when scheduler = custom. Grow the terminal band until the "
                               "sigma_min floor is reachable inside max_gap, within the step budget.",
                }),
                "shift_video": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "0 uses the model's own video shift (12.0 for H3). Anything else overrides "
                               "the body's shape. The stock ModelSamplingMiniMaxH3 node is the better "
                               "place to change the shifts themselves, since the DiT needs them too.",
                }),
                "shift_audio": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "0 uses the model's own audio shift (3.0 for H3).",
                }),
            },
        }

    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "report")
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"
    DESCRIPTION = ("MiniMax H3 schedule: keeps the checkpoint's trained body and replaces the final "
                   "jump with a terminal band spaced in the audio stream's own sigma, so the audio "
                   "decoder is not handed a latent that was last denoised at four times its noise level.")

    def get_sigmas(self, model, scheduler, steps, denoise,
                   terminal_steps=4, sigma_min=0.02, structure_hold=1.0, max_gap=1.9,
                   auto_tail=False, shift_video=0.0, shift_audio=0.0):
        if denoise <= 0.0:
            return (torch.FloatTensor([]), "denoise is 0: nothing to sample")

        ms = model.get_model_object("model_sampling")
        shift_v = float(shift_video) if shift_video > 0.0 else float(getattr(ms, "shift", M.SHIFT_VIDEO))
        shift_a = float(shift_audio) if shift_audio > 0.0 else float(
            getattr(ms, "audio_shift", None) or M.SHIFT_AUDIO)

        total_steps = int(steps / denoise) if denoise < 1.0 else int(steps)

        if scheduler == "simple_stock":
            sigmas = M.t_uniform_sigmas(total_steps, shift_v) + [0.0]
            report = ("MiniMax H3 schedule: stock 'simple' equivalent, %d steps at shift %.2f.\n"
                      "This is what ComfyUI's BasicScheduler builds for H3, kept here for A/B runs.\n"
                      % (total_steps, shift_v))
            report += M.render_schedule_report(sigmas, shift_v, shift_a, total_steps)
        else:
            if scheduler == "custom":
                cfg = dict(terminal_steps=int(terminal_steps), sigma_min=float(sigma_min),
                           structure_hold=float(structure_hold), max_gap=float(max_gap),
                           auto_tail=bool(auto_tail))
            else:
                cfg = dict(SCHEDULE_PRESETS.get(scheduler) or SCHEDULE_PRESETS["h3_balanced"])
            sigmas, report, info = M.h3_schedule(total_steps, shift_v=shift_v,
                                                 shift_a=shift_a, **cfg)
            report = self._header(scheduler, cfg, ms, shift_v, shift_a, info) + "\n" + report

        if denoise < 1.0:
            sigmas = sigmas[-(steps + 1):]
            report += ("\n  denoise %.2f: kept the last %d steps of a %d step schedule"
                       % (denoise, len(sigmas) - 1, total_steps))

        return (torch.FloatTensor(sigmas).cpu(), report)

    @staticmethod
    def _header(scheduler, cfg, ms, shift_v, shift_a, info=None):
        native = getattr(ms, "sigmas", None)
        table = []
        if native is not None:
            try:
                table = [float(v) for v in native.detach().cpu().flatten()]
            except (AttributeError, TypeError):
                table = []
        tail_steps = info["tail_steps"] if info else cfg["terminal_steps"]
        lines = ["MiniMax H3 scheduler, mode %s" % scheduler,
                 "  terminal_steps %d%s, sigma_min %.4f, structure_hold %.2f, max_gap x%.2f,"
                 " auto_tail %s" % (tail_steps,
                                    "" if tail_steps == cfg["terminal_steps"] else " (auto from %d)"
                                    % cfg["terminal_steps"],
                                    cfg["sigma_min"], cfg["structure_hold"],
                                    cfg["max_gap"], cfg["auto_tail"])]
        if len(table) > 1:
            # the body is rebuilt in closed form from the shift, so confirm it still lands on
            # the table this checkpoint samples from, whatever shift node sits upstream
            n = len(table)
            worst = 0.0
            for i in range(64):
                idx = 1 + (i * (n - 2)) // 63
                value = M.sigma_from_t(idx / n, shift_v)
                worst = max(worst, abs(value - table[idx - 1]) / max(abs(table[idx - 1]), 1e-9))
            lines.append("  model sigma table   : %.6f .. %.6f, closed form matches to %.1e relative"
                         % (table[0], table[-1], worst))
        return "\n".join(lines)
