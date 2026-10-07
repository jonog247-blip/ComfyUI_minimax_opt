"""MiniMax H3 schedule and integration math.

Standard library only on purpose: the sampler, the schedule node and the
offline solver lab (``tools/h3_solver_lab.py``) all import this module, so the
schedule a user runs on a GPU and the schedule the lab validates are literally
the same code.

Terminology used throughout:
  sigma        flow-matching noise level, 1.0 = pure noise, 0.0 = clean target
  sigma_v      the sampler's sigma, i.e. the *video* stream's level
  sigma_a      the audio stream's own level, mapped from sigma_v in closed form
  t            the model's internal time, t = 1 - sigma_v

The two facts that shape everything here:

1. A flow sampler's output is essentially the model's clean prediction made at
   the *smallest* sigma the schedule reaches, because the flow ODE carries the
   state to that prediction's worth of denoising. The stock ``simple`` schedule
   stops at sigma_v = 12/(steps+11) (0.52 at 12 steps, 0.39 at 20), so at low
   step counts the model is asked to one-shot the entire remaining denoising.

2. H3 runs video and audio on one sampler schedule with different shifts, and
   sigma_a is four times smaller than sigma_v in the terminal region, so the
   audio stream finishes at sigma_a = sigma_v/4. Whatever the last video sigma
   is, the audio decoder is handed a latent that was last evaluated at a
   quarter of it. At 20 steps with ``simple`` that is sigma_a = 0.136: the
   audio branch never gets a clean look at its own terminal region.
"""

import math

SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0

FPS = 24
AUDIO_LATENT_FPS = 40
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344

FRAME_BLOCK = 17
FRAME_BASE = 5

HIDDEN = 5376
FFN = 14336
LAYERS = 50
HEADS = 56
HEAD_DIM = 128


# ---------------------------------------------------------------------------
# shift algebra (mirrors comfy.ldm.minimax.model.time_shift_sigma exactly)
# ---------------------------------------------------------------------------

def snr_shift(alpha, t):
    """The flow shift used by every ComfyUI flow model: sigma = a*t/(1+(a-1)t)."""
    if alpha == 1.0:
        return t
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def inverse_snr_shift(alpha, sigma):
    """Invert :func:`snr_shift` back onto the shared base grid."""
    if alpha == 1.0:
        return sigma
    return sigma / (alpha + sigma * (1.0 - alpha))


def audio_sigma(sigma_v, shift_v=SHIFT_VIDEO, shift_a=SHIFT_AUDIO):
    """The audio stream's sigma for a sampler sigma, as the DiT computes it."""
    return snr_shift(shift_a, inverse_snr_shift(shift_v, sigma_v))


def video_sigma(sigma_a, shift_v=SHIFT_VIDEO, shift_a=SHIFT_AUDIO):
    """Inverse of :func:`audio_sigma`."""
    return snr_shift(shift_v, inverse_snr_shift(shift_a, sigma_a))


def sigma_from_t(t, shift):
    return snr_shift(shift, t)


def t_from_sigma(sigma, shift):
    return inverse_snr_shift(shift, sigma)


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------

def native_table(shift, timesteps=1000):
    """The ascending sigma table ``ModelSamplingDiscreteFlow`` builds."""
    return [snr_shift(shift, (i + 1) / timesteps) for i in range(timesteps)]


def t_uniform_sigmas(steps, shift=SHIFT_VIDEO):
    """Continuous form of ComfyUI's ``simple`` schedule for one shift.

    ``simple`` samples the shifted table on a uniform *time* grid, which lands
    on ``sigma = shift*t/(1+(shift-1)*t)`` with t = 1 - m/steps. Reproduced in
    closed form so schedules can be built without a model instance.
    """
    steps = max(int(steps), 1)
    return [snr_shift(shift, 1.0 - m / steps) for m in range(steps)]


def geometric_span(sigma_start, sigma_end, count):
    """``count`` sigmas from sigma_start down to sigma_end, log spaced.

    Log spacing is the natural choice for the tail of a flow schedule: the
    integrator's per-step error scales with (step / sigma), so equalising the
    *ratio* between consecutive sigmas equalises the error per step. It is also
    exactly what keeps the audio consistent, because sigma_a tracks sigma_v
    within a constant factor in that region.
    """
    if count <= 0:
        return []
    if sigma_end <= 0.0 or sigma_start <= 0.0:
        raise ValueError("geometric_span needs strictly positive sigmas")
    ratio = (sigma_end / sigma_start) ** (1.0 / count)
    return [sigma_start * ratio ** j for j in range(1, count + 1)]


def _body_sigmas(body_steps, steps, shift_v, structure_hold):
    if body_steps <= 0:
        return []
    span = 1.0 - 1.0 / steps
    out = []
    for m in range(body_steps):
        u = m / (body_steps - 1) if body_steps > 1 else 0.0
        out.append(snr_shift(shift_v, 1.0 - (u ** float(structure_hold)) * span))
    return out


def h3_schedule(steps, shift_v=SHIFT_VIDEO, terminal_steps=4, sigma_min=0.02,
                structure_hold=1.0, shift_a=SHIFT_AUDIO, auto_tail=False, max_gap=1.9,
                auto_tail_max_fraction=0.5):
    """A video-sigma schedule shaped for the joint video+audio H3 latent.

    The body is the model's own trained profile: uniform in the model's
    internal time at the video shift, which is what ``simple`` does and what
    the checkpoint saw during training. The last ``terminal_steps`` sigmas are
    then replaced by a band that steps down *in audio space*, so the audio
    branch is never handed a step larger than ``max_gap`` and the same budget
    finishes far closer to zero instead of jumping there from ~0.5.

    ``structure_hold``
        Exponent on the body's time grid. 1.0 is the native body; above 1.0
        holds more steps at high sigma, where layout and prompt conditioning
        are decided. Below 1.0 packs the body more tightly near its end.

    ``max_gap``
        Largest relative move any single step is allowed to make to the audio
        stream's own sigma. Steps are built in audio space and mapped back, so
        this is a hard constraint on the tail; the body is the trained profile
        and is only reported against it.

    ``auto_tail``
        Grow ``terminal_steps`` until the requested ``sigma_min`` floor is
        reachable inside ``max_gap``, spending at most the step budget and at
        most ``auto_tail_max_fraction`` of it (the body still needs steps: it
        is where the layout and the prompt conditioning are decided). Without
        ``auto_tail``, ``terminal_steps`` is honoured and ``sigma_min`` becomes
        best effort: the band stops at whatever ``max_gap`` allows.

    Returns ``(sigmas, report, info)``; sigmas ends in 0.0 and ``info`` records
    how the budget was split.
    """
    steps = max(int(steps), 1)
    terminal_steps = int(max(0, min(terminal_steps, steps - 1)))
    cap = max(1, int(round(steps * auto_tail_max_fraction)))

    if auto_tail and terminal_steps < min(cap, steps - 1):
        for _ in range(8):
            body = _body_sigmas(steps - terminal_steps, steps, shift_v, structure_hold)
            last = body[-1] if body else 1.0
            if _tail_steps_needed(last, sigma_min, max_gap, shift_v, shift_a) <= terminal_steps:
                break
            terminal_steps = min(terminal_steps + 1, steps - 1)

    body = _body_sigmas(steps - terminal_steps, steps, shift_v, structure_hold)
    last = body[-1] if body else 1.0
    sigmas = body + _tail_sigmas(last, terminal_steps, sigma_min, max_gap, shift_v, shift_a)

    clean = []
    for s in sigmas:
        if clean and s >= clean[-1]:
            s = clean[-1] * 0.999
        clean.append(s)
    clean.append(0.0)

    st = schedule_stats(clean, shift_v, shift_a)
    info = {
        "body_steps": len(body),
        "tail_steps": terminal_steps,
        "body_last_sigma": last,
        "tail_steps_needed": _tail_steps_needed(last, sigma_min, max_gap, shift_v, shift_a),
        "floor_reached": st["final_audio_sigma"] <= audio_sigma(sigma_min, shift_v, shift_a) * 1.0001,
        "auto_tail": bool(auto_tail),
        "tail_gap": max_gap_of(clean[steps - terminal_steps:], shift_v, shift_a) if terminal_steps else 1.0,
    }
    report = render_schedule_report(clean, shift_v, shift_a, steps, info,
                                    requested_sigma_min=sigma_min, max_gap=max_gap)
    return clean, report, info


def max_gap_of(values, shift_v=SHIFT_VIDEO, shift_a=SHIFT_AUDIO):
    """Largest relative audio step inside a slice of a schedule."""
    g = gaps(schedule_audio_grid(list(values), shift_v, shift_a))
    return max(g) if g else 1.0


def _tail_sigmas(sigma_body_last, terminal_steps, sigma_min, max_gap, shift_v, shift_a):
    """The terminal band, spaced uniformly in the audio stream's own sigma."""
    if terminal_steps <= 0:
        return []
    a_last = audio_sigma(sigma_body_last, shift_v, shift_a)
    a_floor = audio_sigma(sigma_min, shift_v, shift_a)
    if a_floor <= 0.0 or a_last <= a_floor:
        ratio = max_gap
    else:
        ratio = min(max_gap, (a_last / a_floor) ** (1.0 / terminal_steps))
    out = []
    for j in range(1, terminal_steps + 1):
        sigma = video_sigma(a_last / ratio ** j, shift_v, shift_a)
        if out:
            sigma = min(sigma, out[-1] * 0.999)
        else:
            sigma = min(sigma, sigma_body_last * 0.999)
        out.append(sigma)
    return out


def _tail_steps_needed(sigma_body_last, sigma_min, max_gap, shift_v, shift_a):
    """How many steps the tail needs to reach the floor without a big gap."""
    a_last = audio_sigma(sigma_body_last, shift_v, shift_a)
    a_floor = audio_sigma(sigma_min, shift_v, shift_a)
    if a_floor <= 0.0 or a_last <= a_floor or max_gap <= 1.0:
        return 1
    return max(1, int(math.ceil(math.log(a_last / a_floor) / math.log(max_gap))))


def schedule_audio_grid(sigmas, shift_v=SHIFT_VIDEO, shift_a=SHIFT_AUDIO):
    return [audio_sigma(s, shift_v, shift_a) for s in sigmas]


def gaps(values):
    """Relative drop between consecutive values, ignoring the arrival at 0."""
    out = []
    for i in range(len(values) - 1):
        if values[i + 1] <= 0.0:
            break
        out.append(values[i] / values[i + 1])
    return out


def max_gap(values):
    g = gaps(values)
    return max(g) if g else 1.0


def last_positive(values):
    for v in reversed(values):
        if v > 0.0:
            return v
    return 0.0


def schedule_stats(sigmas, shift_v=SHIFT_VIDEO, shift_a=SHIFT_AUDIO):
    """Everything worth knowing about a schedule before spending GPU time."""
    audio = schedule_audio_grid(sigmas, shift_v, shift_a)
    vg, ag = gaps(sigmas), gaps(audio)
    return {
        "steps": len(sigmas) - 1,
        "final_video_sigma": last_positive(sigmas),
        "final_audio_sigma": last_positive(audio),
        "max_video_gap": max(vg) if vg else 1.0,
        "max_audio_gap": max(ag) if ag else 1.0,
        "mean_video_gap": sum(vg) / len(vg) if vg else 1.0,
        "audio": audio,
    }


def render_schedule_report(sigmas, shift_v, shift_a, requested_steps, info=None,
                           requested_sigma_min=None, max_gap=None):
    st = schedule_stats(sigmas, shift_v, shift_a)
    audio = st["audio"]
    lines = [
        "  requested steps     : %d" % requested_steps,
        "  scheduled steps     : %d" % st["steps"],
        "  shift video / audio : %.2f / %.2f" % (shift_v, shift_a),
        "  final sigma_v       : %.5f" % st["final_video_sigma"],
        "  final sigma_a       : %.5f   <- what the audio branch last saw" % st["final_audio_sigma"],
        "  largest video step  : x%.3f" % st["max_video_gap"],
        "  largest audio step  : x%.3f" % st["max_audio_gap"],
    ]
    if info:
        lines.append("  terminal band       : %d steps below sigma_v %.4f (tail steps x%.3f)"
                     % (info["tail_steps"], info["body_last_sigma"], info["tail_gap"]))
        lines.append("  body                : %d steps, native profile"
                     % info["body_steps"])
    if max_gap is not None and st["max_audio_gap"] > max_gap:
        lines.append("  note                : the largest audio step above is in the body, which keeps"
                     " the checkpoint's trained spacing; only the terminal band is held to x%.2f"
                     % max_gap)
    if requested_sigma_min is not None:
        reached = st["final_audio_sigma"] <= audio_sigma(requested_sigma_min, shift_v, shift_a) * 1.0001
        lines.append("  sigma_min request   : %.5f  (%s)"
                     % (requested_sigma_min, "reached" if reached else "not reached"))
        if not reached and info:
            if info.get("auto_tail"):
                lines.append("  to reach it         : auto_tail stopped at %d of %d steps, so max_gap or"
                             " the budget is the limit; raise max_gap or steps"
                             % (info["tail_steps"], info["body_steps"] + info["tail_steps"]))
            else:
                lines.append("  to reach it         : about %d terminal steps at a x%.2f audio step,"
                             " or turn on auto_tail"
                             % (info["tail_steps_needed"], max_gap if max_gap else 1.9))
    lines += [
        "",
        "   step     sigma_v     sigma_a   v-step   a-step",
    ]
    for i in range(len(sigmas) - 1):
        if sigmas[i + 1] > 0 and audio[i + 1] > 0:
            vg = "%6.3f" % (sigmas[i] / sigmas[i + 1])
            ag = "%7.3f" % (audio[i] / audio[i + 1])
        else:
            vg = ag = "   ---"  # the landing step: no ratio, the integrator extrapolates here
        lines.append("   %4d  %9.5f  %9.5f   %s   %s" % (i, sigmas[i], audio[i], vg, ag))
    lines.append("   %4d  %9.5f  %9.5f" % (len(sigmas) - 1, sigmas[-1], audio[-1]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# local x0 models and the exponential step (shared with the GPU sampler)
# ---------------------------------------------------------------------------

def order_for_step(current_span, previous_span, requested, ratio_limit=2.0):
    """The model order to actually use for a step, given the two log spans.

    A polynomial fitted over closely spaced points and then evaluated across an
    interval many times wider is the classic way for a multistep solver to ring,
    and a custom schedule can leave exactly such a step in a run, so one level is
    dropped when this step's span dwarfs the previous one's. Everywhere else the
    requested order is left alone.
    """
    order = int(requested)
    if order < 2 or ratio_limit <= 1.0 or current_span <= 0.0 or previous_span <= 0.0:
        return order
    if current_span > ratio_limit * previous_span:
        return order - 1
    return order


def polyfit_sigma_model(hist_sigmas, hist_x0, order, damping=1.0):
    """Fit x0(sigma) through the most recent samples, then damp the curvature.

    ``hist_sigmas`` is newest-first and strictly decreasing. Returns monomial
    coefficients ``[c0, c1, ...]`` for ``x0(s) = c0 + c1*s + c2*s**2``.

    ``damping`` scales the k>=1 coefficients by ``damping**k`` and then
    re-centres the polynomial so it still passes through the newest sample.
    damping == 1 is an exact local fit; below 1 leans on the constant model,
    which is the safe direction when the history is noisy.
    """
    n = min(int(order), len(hist_sigmas) - 1, 3) + 1
    n = max(n, 1)
    s = [float(x) for x in hist_sigmas[:n]]
    y = [float(v) for v in hist_x0[:n]]
    coeffs = _interpolate_poly(s, y)
    if damping != 1.0 and len(coeffs) > 1:
        for k in range(1, len(coeffs)):
            coeffs[k] *= damping ** k
        coeffs[0] += y[0] - _eval_poly(coeffs, s[0])
    return coeffs


def _interpolate_poly(xs, ys):
    """Exact polynomial through (xs, ys) as monomial coefficients (degree <= 3)."""
    n = len(xs)
    coeffs = [0.0] * n
    for i in range(n):
        basis = [1.0]
        denom = 1.0
        for j in range(n):
            if i == j:
                continue
            denom *= (xs[i] - xs[j])
            if denom == 0.0:
                return [ys[0]]  # duplicate sigmas: fall back to constant
            basis = _poly_mul_linear(basis, -xs[j])
        scale = ys[i] / denom
        for k, b in enumerate(basis):
            coeffs[k] += scale * b
    return coeffs


def _poly_mul_linear(coeffs, c):
    out = [0.0] * (len(coeffs) + 1)
    for k, v in enumerate(coeffs):
        out[k] += c * v
        out[k + 1] += v
    return out


def _eval_poly(coeffs, x):
    return sum(c * x ** k for k, c in enumerate(coeffs))


def antiderivative(sigma, coeffs):
    """``integral of x0(s)/s^2 ds`` for a monomial x0 model, up to a constant."""
    total = 0.0
    for k, c in enumerate(coeffs):
        if c == 0.0:
            continue
        if k == 0:
            total -= c / sigma
        elif k == 1:
            total += c * math.log(sigma)
        else:
            total += c * sigma ** (k - 1) / (k - 1)
    return total


def advance_scalar(x_n, sigma_n, sigma_next, coeffs, terminal_extrap=0.0, x0_n=None):
    """The scalar form of the sampler's update, for the same coefficients.

    The flow ODE ``dx/dsigma = (x - x0)/sigma`` has the exact solution

        x(s_next) = s_next * ( x_n/s_n - integral_{s_n}^{s_next} x0(s)/s^2 ds )

    so with a local polynomial x0 model the whole step is one closed form: no
    step-size limit, no ringing, and exact for a constant x0 whatever the step
    size. The GPU sampler evaluates this same expression on tensors, which is
    why this function exists rather than a tensor-only implementation.

    ``terminal_extrap`` only applies to the last step, where the integrator
    lands on the extrapolated x0 at sigma = 0 (1.0) or on the last evaluated x0
    (0.0).
    """
    if sigma_next <= 0.0:
        x0_terminal = _eval_poly(coeffs, 0.0)
        if x0_n is None:
            x0_n = _eval_poly(coeffs, sigma_n)
        return x0_n + terminal_extrap * (x0_terminal - x0_n)
    drop = antiderivative(sigma_next, coeffs) - antiderivative(sigma_n, coeffs)
    return sigma_next * (x_n / sigma_n - drop)


def euler_scalar(x_n, sigma_n, sigma_next, x0_n):
    """Plain Euler reference step on the same ODE."""
    if sigma_n <= 0.0:
        return x_n
    return x_n + (sigma_next - sigma_n) * (x_n - x0_n) / sigma_n


# ---------------------------------------------------------------------------
# geometry / cost estimates (mirrors the H3 nodes, used by the budget node)
# ---------------------------------------------------------------------------

def align_frame_count(n):
    n = max(int(n), FRAME_BASE)
    while n % FRAME_BLOCK != FRAME_BASE:
        n += 1
    return n


def video_latent_t(frame_count):
    if frame_count <= 5:
        return 2
    return ((frame_count - 5) // FRAME_BLOCK) * 5 + 2


def audio_latent_t(frame_count):
    return round((frame_count / FPS) * AUDIO_LATENT_FPS)


def adapt_canvas(width, height):
    """768 short edge, 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def token_estimate(width, height, length):
    """Tokens the packed H3 sequence will carry for this canvas and length."""
    frame_count = align_frame_count(length)
    video = video_latent_t(frame_count) * (height // 32) * (width // 32)
    audio = audio_latent_t(frame_count) * 2
    return {"frames": frame_count, "video_tokens": video, "audio_tokens": audio,
            "seconds": frame_count / FPS}


def step_flops(tokens, layers=LAYERS):
    """FLOPs for one denoising evaluation of a packed H3 sequence.

    Attention is quadratic in the packed length while the projections and the
    MLP are linear, which is why the sparse attention knob matters as much as
    the step count on this model.
    """
    per_token = (
        2 * 4 * HIDDEN * HIDDEN                    # q, k, v, o projections
        + 2 * 3 * HIDDEN * FFN                     # gate, up, down
        + 4 * HEADS * HEAD_DIM * tokens            # scores + weighted values
    )
    return per_token * tokens * layers


def render_budget_report(width, height, length, steps, tflops=150.0, mfu=0.35):
    canvas_w, canvas_h = adapt_canvas(width, height)
    tokens = token_estimate(canvas_w, canvas_h, length)
    packed = tokens["video_tokens"] + tokens["audio_tokens"]
    flops = step_flops(packed)
    attn = 4 * HEADS * HEAD_DIM * packed * packed * LAYERS
    per_step = flops / (tflops * 1e12 * mfu)
    lines = [
        "MiniMax H3 budget estimate",
        "  canvas              : %dx%d (snapped from %dx%d)" % (canvas_w, canvas_h, width, height),
        "  frames              : %d  (%.2f s at 24 fps, 17k+5 grid)" % (
            tokens["frames"], tokens["seconds"]),
        "  packed tokens       : %d  (video %d + audio %d)" % (
            packed, tokens["video_tokens"], tokens["audio_tokens"]),
        "  FLOPs per evaluation: %.2f TFLOP, attention %.0f%% of it" % (
            flops / 1e12, 100.0 * attn / flops),
        "  assumed throughput  : %.0f TFLOP/s at %.0f%% utilisation" % (tflops, 100 * mfu),
        "  estimated per step  : %.1f s" % per_step,
        "  estimated total     : %.1f s for %d steps   (%.1f min)" % (
            per_step * steps, steps, per_step * steps / 60.0),
        "",
        "  These are arithmetic bounds, not measurements: the first step also pays",
        "  prompt encoding and the last one pays both VAE decodes. Real numbers come",
        "  from tools/benchmark_h3.py on your own machine.",
    ]
    return "\n".join(lines)
