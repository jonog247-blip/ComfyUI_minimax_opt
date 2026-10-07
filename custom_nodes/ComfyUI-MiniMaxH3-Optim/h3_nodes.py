"""Supporting MiniMax H3 nodes: budget planning and the one-stop optimiser."""

from __future__ import annotations

from . import h3_math as M

OPTIMISER_MODES = ["off", "balanced", "aggressive", "audio_safe", "custom"]

# start_percent, end_percent, tau
SPARSE_PRESETS = {
    # layout and prompt conditioning form dense; the last step stays dense too
    "balanced": dict(start_percent=0.25, end_percent=0.95, tau=1.3),
    # sparser in the middle, still protects the first quarter and the final step
    "aggressive": dict(start_percent=0.35, end_percent=0.92, tau=1.6),
    # whatever it takes to leave the audio alone: text/audio rows exact, tail dense
    "audio_safe": dict(start_percent=0.3, end_percent=0.9, tau=1.2),
}


class MiniMaxH3Optimiser:
    """Applies the H3-specific patches that pay for themselves, in one node.

    Sparse attention is the big one. At the default 1344x768 over 73 frames the
    packed sequence is ~22.4k tokens, and at that length the attention score and
    value products are about half of the FLOPs in every one of the 50 blocks.
    Block-sparse attention with the H3 sink handling cuts most of that while
    keeping the text and audio rows exact, so it is a per-step win that does not
    touch the step count.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "mode": (OPTIMISER_MODES, {
                    "default": "balanced",
                    "tooltip": "off: pass the model through. balanced: sparse attention from 25% to 95% "
                               "of the schedule. aggressive: sparser middle. audio_safe: tightest audio "
                               "protection, for audio-led work. custom: use the explicit values below.",
                }),
            },
            "optional": {
                "apply_shift": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also set the video/audio flow shifts on this model, exactly as the stock "
                               "ModelSamplingMiniMaxH3 node does. Off by default: leave the checkpoint's "
                               "trained 12.0 / 3.0 alone unless you are deliberately retuning.",
                }),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "tau": ("FLOAT", {
                    "default": 1.3, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Sol-Attn threshold in score-distribution sigmas. Higher is sparser: 1.0 "
                               "keeps about 16% of key blocks exact, 1.5 about 7%.",
                }),
                "start_percent": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dense_blocks": ("STRING", {
                    "default": "",
                    "tooltip": "Transformer blocks that always run dense, e.g. '0, 1, 47-49'.",
                }),
                "min_tokens": ("INT", {"default": 12288, "min": 0, "max": 1 << 20, "step": 512}),
                "extra_tokens": ("INT", {"default": 256, "min": 0, "max": 256, "step": 64}),
                "sink_conditioning": (["exact_kv_and_rows", "exact_kv", "off"], {
                    "default": "exact_kv_and_rows",
                    "tooltip": "MiniMax H3 only. exact_kv_and_rows additionally runs the target audio "
                               "query rows dense, which is what keeps generated audio intact.",
                }),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = "model/patch/minimax"
    DESCRIPTION = ("MiniMax H3 optimiser: applies block-sparse attention with H3 sink protection "
                   "and optionally the stock flow-shift patch.")

    def apply(self, model, mode="balanced", apply_shift=False, shift_video=12.0, shift_audio=3.0,
              tau=1.3, start_percent=0.25, end_percent=0.95, dense_blocks="", min_tokens=12288,
              extra_tokens=256, sink_conditioning="exact_kv_and_rows", verbose=False):
        lines = ["MiniMax H3 optimiser"]
        out = model

        if apply_shift:
            out = _apply_flow_shift(out, float(shift_video), float(shift_audio))
            lines.append("  flow shifts set to video %.2f / audio %.2f" % (shift_video, shift_audio))

        if mode == "off":
            lines.append("  sparse attention: off")
            return (out, "\n".join(lines))

        if mode == "custom":
            cfg = dict(tau=float(tau), start_percent=float(start_percent),
                       end_percent=float(end_percent))
        else:
            cfg = dict(SPARSE_PRESETS.get(mode, SPARSE_PRESETS["balanced"]))
        tau = cfg["tau"]
        start_percent = cfg["start_percent"]
        end_percent = cfg["end_percent"]

        try:
            from comfy_extras.nodes_sparse_attention import apply_block_sparse_attention, parse_block_list
        except ImportError as exc:
            lines.append("  sparse attention unavailable on this build (%s)" % exc)
            lines.append("  everything else in this node still applies")
            return (out, "\n".join(lines))

        out = apply_block_sparse_attention(
            out,
            tau=tau,
            topk_ratio=0.0,          # sol-attn: adaptive threshold, no distillation required
            vsa=False,
            start_percent=start_percent,
            end_percent=end_percent,
            min_tokens=int(min_tokens),
            dense_blocks=parse_block_list(dense_blocks),
            sink_conditioning=sink_conditioning,
            extra_tokens=int(extra_tokens),
            verbose=bool(verbose),
        )
        lines.append("  sparse attention: sol-attn tau %.2f, %d%%-%.0f%% of the schedule, %d extra tokens"
                     % (tau, round(start_percent * 100), end_percent * 100, extra_tokens))
        lines.append("  sink conditioning: %s%s" % (
            sink_conditioning,
            "  (keeps the text and generated-audio rows exact)" if sink_conditioning != "off" else ""))
        if dense_blocks:
            lines.append("  always dense: %s" % dense_blocks)
        lines.append("  sparse attention is skipped automatically below %d tokens" % min_tokens)
        return (out, "\n".join(lines))


def _apply_flow_shift(model, shift_video, shift_audio):
    """Mirrors comfy_extras MiniMaxH3SigmaShift so the pack works standalone."""
    import comfy.model_sampling

    m = model.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    original = m.get_model_object("model_sampling")
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift_video, audio_shift=shift_audio)
    if hasattr(original, "noise_scale"):
        model_sampling.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", model_sampling)

    to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
    to["minimax_h3_sigma_shift_video"] = shift_video
    to["minimax_h3_sigma_shift_audio"] = shift_audio
    return m


class MiniMaxH3Budget:
    """Snaps a duration onto the model's frame grid and estimates the cost."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "seconds": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 150.0, "step": 0.1}),
                "steps": ("INT", {"default": 14, "min": 1, "max": 200, "step": 1}),
            },
            "optional": {
                "device_tflops": ("FLOAT", {
                    "default": 150.0, "min": 1.0, "max": 10000.0, "step": 5.0,
                    "tooltip": "Effective dense throughput of the GPU for this model's datatype. "
                               "An RTX 5070 Ti lands near 150 TFLOP/s for int8 weights with bf16 "
                               "activations; tighten this after running tools/benchmark_h3.py.",
                }),
                "utilisation": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("length_frames", "packed_tokens", "report")
    FUNCTION = "report"
    CATEGORY = "model/minimax"
    DESCRIPTION = "Snap a duration to MiniMax H3's 17k+5 frame grid and estimate tokens, FLOPs and time."

    def report(self, width, height, seconds, steps, device_tflops=150.0, utilisation=0.35):
        frame_count = M.align_frame_count(int(round(seconds * M.FPS)))
        canvas_w, canvas_h = M.adapt_canvas(width, height)
        tokens = M.token_estimate(canvas_w, canvas_h, frame_count)
        packed = tokens["video_tokens"] + tokens["audio_tokens"]
        text = M.render_budget_report(canvas_w, canvas_h, frame_count, steps,
                                      tflops=device_tflops, mfu=utilisation)
        text += "\n\n  feed length_frames into MiniMax H3 Image to Video for %.2f s of audio+video" % (
            frame_count / M.FPS)
        return (frame_count, packed, text)
