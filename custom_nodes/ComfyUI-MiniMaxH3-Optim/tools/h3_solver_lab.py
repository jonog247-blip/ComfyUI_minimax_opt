"""Offline solver lab for the MiniMax H3 sampler.

Runs on CPU with the standard library only, so it works on a machine with no
torch and no GPU. It answers two questions with numbers instead of vibes:

  1. how much of the trajectory does the integrator resolve at a given step
     count (the sampler's job), and
  2. how much error is left over because the schedule never travels to small
     sigma (the scheduler's job).

Those are separate error terms and they are measured separately here. The
second one is the one that dominates at the step counts people actually run,
which is why this pack ships a scheduler and not just a sampler.

Usage:  python3 tools/h3_solver_lab.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_math as M


# ---------------------------------------------------------------------------
# analytic test problems
# ---------------------------------------------------------------------------

class Straight:
    """Ideal rectified flow: constant x0, so every correct integrator is exact."""

    name = "constant x0 (ideal rectified flow)"

    def __init__(self, eps=1.0):
        self.eps = eps

    def x0(self, sigma):
        return 1.0

    def start(self, sigma):
        return (1.0 - sigma) * self.x0(sigma) + sigma * self.eps

    def exact(self, sigma_n, x_n, sigma_next):
        # with constant x0 the ratio (x - x0)/sigma is invariant along the ODE
        return self.x0(sigma_n) + (sigma_next / sigma_n) * (x_n - self.x0(sigma_n))


class Curved:
    """x0 drifts with sigma, which is the error a real video model injects.

    ``x0(sigma) = 1 + amp * sigma**power`` models the usual behaviour: the
    model's estimate of the clean video is poor at high sigma and settles as
    sigma falls. Tracking that drift is what a second-order step is for, and
    crossing it in one final step is what a short schedule cannot avoid.
    """

    def __init__(self, amp=0.6, power=1.5, eps=1.0):
        self.amp, self.power, self.eps = amp, power, eps
        self.name = "curved x0 (amp=%.2f power=%.1f)" % (amp, power)

    def x0(self, sigma):
        return 1.0 + self.amp * sigma ** self.power

    def start(self, sigma):
        return (1.0 - sigma) * self.x0(sigma) + sigma * self.eps

    def exact(self, sigma_n, x_n, sigma_next):
        """Fine RK4 over the interval: an independent reference."""
        return _rk4_interval(self, sigma_n, x_n, sigma_next, substeps=4000)


def _f(problem, sigma, x):
    if sigma <= 1e-12:
        return 0.0
    return (x - problem.x0(sigma)) / sigma


def _rk4_interval(problem, sigma_n, x_n, sigma_next, substeps):
    h = (sigma_next - sigma_n) / substeps
    s, x = sigma_n, x_n
    for _ in range(substeps):
        k1 = _f(problem, s, x)
        k2 = _f(problem, s + h / 2, x + h * k1 / 2)
        k3 = _f(problem, s + h / 2, x + h * k2 / 2)
        k4 = _f(problem, s + h, x + h * k3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        s += h
    return x


# ---------------------------------------------------------------------------
# runners: the pack's integrator, exactly as the GPU sampler evaluates it
# ---------------------------------------------------------------------------

def integrate(problem, sigmas, order=2, damping=1.0, terminal_extrap=0.0,
              guard=False, guard_terminal=False, ratio_limit=2.0):
    """Returns (x at the last non-zero sigma, x after the final step to 0, demotions).

    ``guard`` calls ``h3_math.order_for_step`` with this step's log span and the
    previous step's, which is the same rule the GPU sampler applies live, so this
    measures the shipped behaviour rather than a model of it.
    """
    x = problem.start(sigmas[0])
    hist_s, hist_x0, coeffs = [], [], None
    x_body = x
    demotions = 0
    for i in range(len(sigmas) - 1):
        sigma_n, sigma_next = sigmas[i], sigmas[i + 1]
        if sigma_n <= 0.0:
            break
        x0_n = problem.x0(sigma_n)
        eff_order = order
        if guard and i >= 1 and hist_s:
            if sigma_next > 0.0 or guard_terminal:
                eff_order = M.order_for_step(math.log(sigma_n / sigma_next),
                                             math.log(sigmas[i - 1] / sigma_n),
                                             order, ratio_limit)
                demotions += 1 if eff_order < order else 0
        coeffs = M.polyfit_sigma_model(hist_s, hist_x0, eff_order, damping) if hist_s else [x0_n]
        if sigma_next <= 0.0:
            x = M.advance_scalar(x, sigma_n, sigma_next, coeffs,
                                 terminal_extrap=terminal_extrap, x0_n=x0_n)
            break
        x = M.advance_scalar(x, sigma_n, sigma_next, coeffs)
        x_body = x
        hist_s.insert(0, sigma_n)
        hist_x0.insert(0, x0_n)
    return x_body, x, demotions


def integrate_euler(problem, sigmas):
    x, x_body = problem.start(sigmas[0]), problem.start(sigmas[0])
    for i in range(len(sigmas) - 1):
        if sigmas[i] <= 0.0:
            break
        if sigmas[i + 1] <= 0.0:
            x = problem.x0(sigmas[i])
            break
        x = M.euler_scalar(x, sigmas[i], sigmas[i + 1], problem.x0(sigmas[i]))
        x_body = x
    return x_body, x


_REF_CACHE = {}


def reference_body(problem, sigma_last):
    """Exact ODE solution at sigma_last, independent of any sampler."""
    key = (type(problem).__name__, getattr(problem, "eps", None),
           getattr(problem, "amp", None), getattr(problem, "power", None),
           round(float(sigma_last), 12))
    if key not in _REF_CACHE:
        _REF_CACHE[key] = _integrate_to(problem, 1.0, problem.start(1.0), sigma_last)
    return _REF_CACHE[key]


def _integrate_to(problem, sigma_from, x, sigma_to, pieces=400):
    edges = [sigma_from + (sigma_to - sigma_from) * i / pieces for i in range(pieces + 1)]
    for a, b in zip(edges, edges[1:]):
        x = _rk4_interval(problem, a, x, b, substeps=600)
    return x


def rel_error(value, ref):
    return abs(value - ref) / max(abs(ref), 1e-12)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def sweep(problem, label):
    print()
    print("%s   [%s]" % (problem.name, label))
    print()
    print("   steps | Euler body | solver body |  H3 body | 'simple' final | H3 final x0 | H3 final extrap")
    print("   ------+------------+-------------+----------+----------------+-------------+----------------")
    for steps in (4, 6, 8, 10, 12, 16, 20, 25):
        simple = M.t_uniform_sigmas(steps, M.SHIFT_VIDEO) + [0.0]
        h3, _, _ = M.h3_schedule(steps, M.SHIFT_VIDEO,
                                 terminal_steps=4 if steps >= 8 else 2, sigma_min=0.02)

        ref = reference_body(problem, M.last_positive(h3))

        _, e_final_plain = integrate_euler(problem, simple)
        eb_euler, _ = integrate_euler(problem, simple)
        eb_solver, _, _ = integrate(problem, simple, 2, 1.0)
        hb_solver, h_final_x0, _ = integrate(problem, h3, 2, 1.0, terminal_extrap=0.0)
        _, h_final_ex, _ = integrate(problem, h3, 2, 1.0, terminal_extrap=1.0)

        print("   %5d | %10.3e | %11.3e | %8.3e | %14.3e | %11.3e | %15.3e"
              % (steps, rel_error(eb_euler, ref), rel_error(eb_solver, ref),
                 rel_error(hb_solver, ref), rel_error(e_final_plain, 1.0),
                 rel_error(h_final_x0, 1.0), rel_error(h_final_ex, 1.0)))


PRESETS = [
    # the sampler node's own choices, plus two single-knob diagnostics
    ("euler_reference", 0, 1.00, False, False),
    ("linear", 1, 1.00, False, False),
    ("balanced", 2, 1.00, False, False),
    ("max", 3, 1.00, False, False),
    ("balanced, step_guard", 2, 1.00, True, False),
    ("balanced, damping 0.85", 2, 0.85, False, False),
]


def preset_table(problem, label):
    """Body and landing error for every integrator the sampler node offers."""
    counts = (8, 12, 20)
    schedules = {n: M.h3_schedule(n, M.SHIFT_VIDEO, terminal_steps=4 if n >= 8 else 2,
                                 sigma_min=0.02)[0] for n in counts}
    refs = {n: reference_body(problem, M.last_positive(schedules[n])) for n in counts}
    print()
    print("%s   [integrator variants, H3 schedule, 4-step terminal band]" % label)
    print()
    print("   %-18s" % "variant" + "".join("%16s" % ("%d steps" % n) for n in counts))
    for want_body in (True, False):
        print("   %s" % ("body: state at the last non-zero sigma"
                         if want_body else "final: after the step that lands on sigma = 0"))
        for lab, order, damping, guard, guard_terminal in PRESETS:
            cells = []
            for n in counts:
                xb, xf, demotions = integrate(problem, schedules[n], order, damping,
                                              terminal_extrap=1.0, guard=guard,
                                              guard_terminal=guard_terminal)
                err = rel_error(xb, refs[n]) if want_body else rel_error(xf, 1.0)
                cells.append("%16.3e" % err if not guard else "%11.3e/%d" % (err, demotions))
            print("   %-18s" % lab + "".join(cells))
    print("   (guarded rows show the number of demoted steps after the slash)")
    print()

def main():
    print("MiniMax H3 solver lab")
    print("=" * 100)
    print()
    print("body  = error of the state at the schedule's last non-zero sigma (the integrator's error)")
    print("final = error after the step that lands on sigma=0 (the integrator's terminal rule)")
    print("'simple' keeps the stock schedule; H3 uses the pack's terminal band at the same step count")

    sweep(Straight(eps=1.0), "sanity: must be zero everywhere")
    sweep(Curved(0.6, 1.5), "video-like drift")
    sweep(Curved(1.2, 2.0), "aggressive drift")

    for problem, label in ((Curved(0.6, 1.5), "video-like drift"),
                           (Curved(1.2, 2.0), "aggressive drift")):
        print("=" * 100)
        preset_table(problem, label)

    print()
    print("=" * 100)
    print("Where the stock schedule leaves the two streams (nothing to do with the sampler)")
    print()
    print("   steps | simple sigma_v | simple sigma_a | H3 sigma_v | H3 sigma_a | H3 steps used")
    for steps in (4, 6, 8, 12, 20):
        simple = M.t_uniform_sigmas(steps, M.SHIFT_VIDEO)
        h3, _, _ = M.h3_schedule(steps, M.SHIFT_VIDEO, terminal_steps=4 if steps >= 8 else 2,
                                 sigma_min=0.02)
        print("   %5d | %14.5f | %14.5f | %10.5f | %10.5f | %13d"
              % (steps, simple[-1], M.audio_sigma(simple[-1]),
                 M.last_positive(h3), M.audio_sigma(M.last_positive(h3)),
                 len(h3) - 1))

    print()
    print("=" * 100)
    print("12-step schedule, side by side")
    simple = M.t_uniform_sigmas(12, M.SHIFT_VIDEO)
    h3, _, _ = M.h3_schedule(12, M.SHIFT_VIDEO, terminal_steps=4, sigma_min=0.02)
    print(M.render_schedule_report(h3, M.SHIFT_VIDEO, M.SHIFT_AUDIO, 12,
                                   info=M.h3_schedule(12, M.SHIFT_VIDEO, terminal_steps=4,
                                                      sigma_min=0.02)[2],
                                   requested_sigma_min=0.02, max_gap=1.9))
    print()
    print("  stock 'simple' at the same count:")
    print("    final sigma_v %.5f  final sigma_a %.5f  max video step x%.3f  max audio step x%.3f"
          % (simple[-1], M.audio_sigma(simple[-1]),
             M.max_gap(simple + [0.0]), M.max_gap(M.schedule_audio_grid(simple + [0.0]))))

    print()
    print("=" * 100)
    print(M.render_budget_report(1344, 768, 73, 12, tflops=150.0, mfu=0.35))


if __name__ == "__main__":
    main()
