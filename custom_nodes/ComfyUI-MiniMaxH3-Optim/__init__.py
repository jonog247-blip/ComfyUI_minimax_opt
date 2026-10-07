"""ComfyUI-MiniMaxH3-Optim

Speed and quality nodes for ComfyUI's native MiniMax H3 audio-video model.

Four nodes, all optional, all optional-to-remove:

    MiniMax H3 Scheduler   the shipped 'simple' schedule leaves the audio stream
                           at sigma_a 0.136 on a 20 step run; this one finishes it
                           near clean inside the same step budget.
    MiniMax H3 Sampler     one model evaluation per step, exact integration of the
                           flow ODE, terminal step on the extrapolated prediction.
    MiniMax H3 Optimiser   block-sparse attention with H3 sink protection, where
                           attention is about half the FLOPs at 1344x768.
    MiniMax H3 Budget      frame grid snapping plus a token, FLOP and time estimate.

The schedule and the sampler are designed together: the sampler's exponential
step is exact for a constant clean prediction, which is what makes a compressed
terminal band safe to integrate, and the schedule's terminal band is spaced in
the audio stream's own sigma so the audio never crosses a large gap in one step.
"""

from .h3_nodes import MiniMaxH3Budget, MiniMaxH3Optimiser
from .h3_sampler import MiniMaxH3SamplerSelect
from .h3_scheduler import MiniMaxH3Scheduler

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Scheduler": MiniMaxH3Scheduler,
    "MiniMaxH3SamplerSelect": MiniMaxH3SamplerSelect,
    "MiniMaxH3Optimiser": MiniMaxH3Optimiser,
    "MiniMaxH3Budget": MiniMaxH3Budget,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Scheduler": "MiniMax H3 Scheduler",
    "MiniMaxH3SamplerSelect": "MiniMax H3 Sampler",
    "MiniMaxH3Optimiser": "MiniMax H3 Optimiser",
    "MiniMaxH3Budget": "MiniMax H3 Budget",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
