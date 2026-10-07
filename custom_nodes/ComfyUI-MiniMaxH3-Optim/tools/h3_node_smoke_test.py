"""Import and schema smoke test for the pack, with no torch and no GPU.

Stubs the small slice of ``torch`` and ``comfy`` the pack touches, then loads
the pack exactly the way ComfyUI's ``load_custom_node`` does. Catches the
mistakes that otherwise only surface as a red node in the UI: bad INPUT_TYPES
shapes, a default outside its own min/max, a missing mapping, a schedule that is
not strictly decreasing.

Usage:  python3 tools/h3_node_smoke_test.py
"""

import importlib.util
import os
import sys
import types

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK_NAME = os.path.basename(PACK_DIR)


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------

class _FakeTensor:
    def __init__(self, data):
        self.data = [float(v) for v in data]

    def cpu(self):
        return self

    def detach(self):
        return self

    def flatten(self):
        return self

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


def _install_stubs():
    torch = types.ModuleType("torch")
    torch.Tensor = _FakeTensor
    torch.FloatTensor = lambda data=None, *a, **k: _FakeTensor(data or [])
    torch.no_grad = lambda: (lambda fn: fn)
    torch.randn_like = lambda t: t
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.cat = lambda tensors, dim=-1: tensors[0]
    torch.isfinite = lambda t: types.SimpleNamespace(all=lambda: True)
    sys.modules["torch"] = torch

    comfy = types.ModuleType("comfy")

    mm = types.ModuleType("comfy.model_management")
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.get_torch_device = lambda: types.SimpleNamespace(type="cpu")
    mm.get_torch_device_name = lambda device: "stub device"
    comfy.model_management = mm

    samplers = types.ModuleType("comfy.samplers")

    class KSampler:
        def __init__(self, sampler_function, extra_options=None, inpaint_options=None):
            self.sampler_function = sampler_function
            self.extra_options = extra_options or {}

    class Sampler:
        def sample(self):
            pass

    samplers.KSAMPLER = KSampler
    samplers.Sampler = Sampler
    samplers.Guider_Basic = lambda model: None
    comfy.samplers = samplers

    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = lambda total: types.SimpleNamespace(update=lambda n: None)
    comfy.utils = utils

    ms = types.ModuleType("comfy.model_sampling")

    class _AV:
        pass

    class _CONST:
        pass

    ms.ModelSamplingAV = _AV
    ms.CONST = _CONST
    comfy.model_sampling = ms

    sys.modules["comfy"] = comfy
    for name, module in (("comfy.model_management", mm), ("comfy.samplers", samplers),
                         ("comfy.utils", utils), ("comfy.model_sampling", ms)):
        sys.modules[name] = module


class _StubModelSampling:
    """Mirrors ModelSamplingDiscreteFlow.set_parameters: ascending t, ascending table."""

    def __init__(self, shift=12.0, audio_shift=3.0):
        self.shift = shift
        self.audio_shift = audio_shift
        table = []
        for k in range(1, 1001):
            t = k / 1000.0
            table.append(shift * t / (1.0 + (shift - 1.0) * t))
        self.sigmas = _FakeTensor(table)

    @property
    def sigma_min(self):
        return self.sigmas.data[0]

    @property
    def sigma_max(self):
        return self.sigmas.data[-1]


class _StubModel:
    def __init__(self):
        self._ms = _StubModelSampling()

    def get_model_object(self, name):
        assert name == "model_sampling"
        return self._ms

    def clone(self):
        return self


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_input_types(name, cls):
    schema = cls.INPUT_TYPES()
    assert isinstance(schema, dict), "%s INPUT_TYPES is not a dict" % name
    assert "required" in schema, "%s has no required inputs" % name
    for group in ("required", "optional"):
        for key, spec in schema.get(group, {}).items():
            if not isinstance(spec, tuple) or len(spec) < 2:
                raise AssertionError("%s.%s spec is not (type, options)" % (name, key))
            kind, options = spec[0], spec[1]
            if isinstance(kind, list):
                assert kind, "%s.%s has an empty combo" % (name, key)
                if "default" in options:
                    assert options["default"] in kind, \
                        "%s.%s default %r not in its own options" % (name, key, options["default"])
            if isinstance(kind, str) and kind in ("INT", "FLOAT"):
                if "default" in options and "min" in options and "max" in options:
                    assert options["min"] <= options["default"] <= options["max"], \
                        "%s.%s default outside min/max" % (name, key)
    return schema


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--report":
        _install_stubs()
        spec = importlib.util.spec_from_file_location(
            PACK_NAME, os.path.join(PACK_DIR, "__init__.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACK_NAME] = module
        spec.loader.exec_module(module)
        mode = sys.argv[2]
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 14
        _, report = module.NODE_CLASS_MAPPINGS["MiniMaxH3Scheduler"]().get_sigmas(
            _StubModel(), mode, steps, 1.0)
        print(report)
        return

    _install_stubs()

    spec = importlib.util.spec_from_file_location(
        PACK_NAME, os.path.join(PACK_DIR, "__init__.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACK_NAME] = module
    spec.loader.exec_module(module)
    print("pack loaded as %s" % PACK_NAME)

    mappings = module.NODE_CLASS_MAPPINGS
    displays = module.NODE_DISPLAY_NAME_MAPPINGS
    assert mappings, "NODE_CLASS_MAPPINGS is empty"
    print("registered %d nodes: %s" % (len(mappings), ", ".join(sorted(mappings))))
    for key, cls in mappings.items():
        assert key in displays, "%s has no display name" % key
        schema = check_input_types(key, cls)
        print("  %-26s inputs %2d required, %2d optional, outputs %s"
              % (key, len(schema["required"]), len(schema.get("optional", {})),
                 tuple(getattr(cls, "RETURN_TYPES", ()))))

    # the scheduler is the one node whose maths can be checked without a GPU
    M = sys.modules[PACK_NAME + ".h3_math"]
    SCHEDULE_PRESETS = sys.modules[PACK_NAME + ".h3_scheduler"].SCHEDULE_PRESETS

    scheduler_cls = mappings["MiniMaxH3Scheduler"]
    model = _StubModel()
    cases = [("h3_balanced", 14), ("h3_draft", 8), ("h3_turbo8", 8),
             ("h3_max", 20), ("h3_audio_max", 20), ("h3_base20", 20),
             ("simple_stock", 20), ("custom", 12)]
    for mode, steps in cases:
        sigmas, report = scheduler_cls().get_sigmas(model, mode, steps, 1.0)
        values = list(sigmas)
        assert values[-1] == 0.0, "%s does not end at 0" % mode
        assert all(values[i] > values[i + 1] for i in range(len(values) - 1)), \
            "%s is not strictly decreasing" % mode
        assert values[0] <= 1.0 + 1e-9, "%s starts above sigma 1" % mode
        stats = M.schedule_stats(values)
        tail_note = ""
        cfg = SCHEDULE_PRESETS.get(mode)
        if cfg:
            # the terminal band is the part this pack builds, and it must honour max_gap
            _, _, info = M.h3_schedule(steps, M.SHIFT_VIDEO,
                                       shift_a=M.SHIFT_AUDIO, **cfg)
            cut = len(values) - 1 - info["tail_steps"]
            tail_gaps = M.gaps(M.schedule_audio_grid(values[cut:]))
            assert tail_gaps and max(tail_gaps) <= cfg["max_gap"] * 1.001, \
                "%s terminal band breaks max_gap (%.3f > %.2f)" % (
                    mode, max(tail_gaps), cfg["max_gap"])
            tail_note = "  tail x%.2f in %d steps" % (max(tail_gaps), len(values) - cut - 1)
        print("  %-14s steps %2d  final sigma_v %.5f  final sigma_a %.5f  max gaps x%.2f / x%.2f%s"
              % (mode, stats["steps"], stats["final_video_sigma"], stats["final_audio_sigma"],
                 stats["max_video_gap"], stats["max_audio_gap"], tail_note))
        assert report.strip(), "%s produced an empty report" % mode

    custom, _ = scheduler_cls().get_sigmas(model, "custom", 12, 1.0)
    stock, _ = scheduler_cls().get_sigmas(model, "simple_stock", 12, 1.0)
    assert list(custom) != list(stock), "custom mode is not reaching the schedule builder"
    balanced, _ = scheduler_cls().get_sigmas(model, "h3_balanced", 12, 1.0)
    assert list(custom) == list(balanced), "custom default differs from the balanced preset"
    assert report.count("\n") > 5, "the report should carry the per-step table"

    # denoise < 1 keeps the tail rather than truncating it
    partial, _ = scheduler_cls().get_sigmas(model, "h3_balanced", 10, 0.5)
    assert len(partial) == 11, "denoise slicing produced %d sigmas" % len(partial)
    assert M.schedule_stats(list(partial))["final_audio_sigma"] < 0.05

    # sampler presets, and the custom path honouring its widgets
    select = mappings["MiniMaxH3SamplerSelect"]()
    expected_orders = {"balanced": 2, "max": 3, "linear": 1, "euler_reference": 0}
    for mode, want in expected_orders.items():
        options = select.get_sampler(mode)[0].extra_options
        assert options["order"] == want, "%s built order %s" % (mode, options["order"])
        assert options["step_guard"] is False, "%s enabled the step guard by default" % mode
        print("  sampler %-16s order %d, terminal_extrap %.2f" %
              (mode, options["order"], options["terminal_extrap"]))
    custom = select.get_sampler("custom", order=3, curvature_damping=0.5,
                                audio_order_boost=1, terminal_extrap=0.0)[0].extra_options
    assert (custom["order"], custom["curvature_damping"], custom["audio_order_boost"],
            custom["terminal_extrap"]) == (3, 0.5, 1, 0.0), "custom mode ignored its widgets"
    print("  sampler custom          honours the widgets: %s" % custom)

    frames, packed, text = mappings["MiniMaxH3Budget"]().report(1344, 768, 5.0, 14)
    assert frames % 17 == 5, "frame count is off the 17k+5 grid"
    print("  budget: %d frames, %d packed tokens" % (frames, packed))
    assert "TFLOP" in text

    print("\nall checks passed")


if __name__ == "__main__":
    main()
