#!/usr/bin/env python3
"""
Closed-loop validation simulator for the tuning wizard (tuning-theory.md,
validation ladder rung 2). Not an operator tool.

Simulates a single continuous-batching endpoint held AT the wizard's derived
concurrency limit N* (closed loop: a completion immediately admits a
successor), which is exactly the regime the stationary size-biased model
describes. KV occupancy is tracked as I + min(generated, L) per active
request; "overflow" is any instant with total occupancy above C_tokens.

For each grid point it:
  1. draws a calibration trace from the workload distribution,
  2. derives N* via the wizard's Chernoff program at target epsilon,
  3. simulates and measures the realized overflow time-fraction,
  4. reports pass/fail (realized <= epsilon) plus mean utilization
     (the stranded-capacity view) and the true overflow onset N.

Acceptance (rung 2): realized overflow <= epsilon on >= 95% of grid points.
The cohort stressor intentionally violates cross-slot independence (the
model's documented soft spot) and is reported separately, not gated.
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tuning_wizard import JointTraceModel, chernoff_n_star, gaussian_n_star  # noqa: E402

C_TOKENS = 315_000.0     # 21875 blocks x 16 tokens x 0.9, the guide reference scale
EPSILON = 0.01           # grid target; resolvable within the sim horizon below
DECODE_RATE = 20.0       # tokens/s per active sequence
SIM_SECONDS = 3000.0
DT = 0.25
WARMUP_FRAC = 0.2
TRACE_SIZE = 5000


def make_sampler(kind: str, isl_scale: float, osl_scale: float, rng: random.Random):
    if kind == "deterministic":
        return lambda: (isl_scale, osl_scale)
    if kind == "exponential":
        return lambda: (rng.expovariate(1 / isl_scale), rng.expovariate(1 / osl_scale))
    if kind.startswith("lognormal"):
        sigma = float(kind.split(":")[1])
        mu_i = math.log(isl_scale) - sigma ** 2 / 2  # mean-preserving
        mu_l = math.log(osl_scale) - sigma ** 2 / 2
        return lambda: (rng.lognormvariate(mu_i, sigma), rng.lognormvariate(mu_l, sigma))
    if kind == "correlated":  # long inputs beget long outputs (rho ~ 0.8)
        def s():
            z = rng.gauss(0, 1)
            zi = 0.8 * z + 0.6 * rng.gauss(0, 1)
            return (isl_scale * math.exp(0.7 * z - 0.245), osl_scale * math.exp(0.7 * zi - 0.245))
        return s
    raise ValueError(kind)


def simulate(sampler, n_limit: int, rng: random.Random, cohort_q: float = 0.0):
    """Returns (overflow_fraction, mean_utilization, p99_utilization)."""
    active = []  # [isl, osl, generated]
    last = None
    for _ in range(n_limit):
        i, l = sampler()
        active.append([i, l, rng.random() * l])  # random initial ages: warm start
    steps = int(SIM_SECONDS / DT)
    warmup = int(steps * WARMUP_FRAC)
    over = 0
    utils = []
    for step in range(steps):
        occ = 0.0
        for req in active:
            req[2] += DECODE_RATE * DT
            if req[2] >= req[1]:
                if cohort_q > 0 and last is not None and rng.random() < cohort_q:
                    i, l = last  # cohort effect: repeat the previous admission
                else:
                    i, l = sampler()
                last = (i, l)
                req[0], req[1], req[2] = i, l, 0.0
            occ += req[0] + min(req[2], req[1])
        if step >= warmup:
            if occ > C_TOKENS:
                over += 1
            utils.append(occ / C_TOKENS)
    utils.sort()
    n = len(utils)
    return (over / max(1, n), sum(utils) / n, utils[int(0.99 * (n - 1))])


def overflow_onset(sampler, n_start: int, rng: random.Random) -> int:
    """Smallest N whose overflow fraction exceeds epsilon (searched upward):
    how much real capacity the bound leaves on the table."""
    n = n_start
    while n < n_start * 3:
        frac, _, _ = simulate(sampler, n, random.Random(rng.randrange(1 << 30)))
        if frac > EPSILON:
            return n
        n = max(n + 1, int(n * 1.06))
    return n


def main() -> int:
    grid = []
    for kind in ("deterministic", "exponential", "lognormal:0.7", "lognormal:1.1",
                 "lognormal:1.4", "correlated"):
        for isl_scale, osl_scale in ((2000.0, 400.0), (500.0, 1500.0), (4000.0, 150.0)):
            grid.append((kind, isl_scale, osl_scale))

    rows, failures = [], []
    print(f"{'workload':<16} {'ISL':>5} {'OSL':>5} {'N*':>5} {'overflow':>9} "
          f"{'util':>6} {'p99util':>8} {'onset':>6}  verdict")
    for gi, (kind, i_s, o_s) in enumerate(grid):
        rng = random.Random(1000 + gi)
        sampler = make_sampler(kind, i_s, o_s, rng)
        trace_i, trace_l = zip(*(sampler() for _ in range(TRACE_SIZE)))
        model = JointTraceModel(list(trace_i), list(trace_l))
        n_star, _ = chernoff_n_star(model, C_TOKENS, EPSILON)
        if n_star < 1:
            continue
        frac, util, p99 = simulate(sampler, n_star, random.Random(2000 + gi))
        onset = overflow_onset(sampler, n_star, random.Random(3000 + gi))
        ok = frac <= EPSILON
        if not ok:
            failures.append((kind, i_s, o_s, frac))
        rows.append(ok)
        print(f"{kind:<16} {i_s:>5.0f} {o_s:>5.0f} {n_star:>5} {frac:>9.4f} "
              f"{util:>6.2f} {p99:>8.2f} {onset:>6}  {'pass' if ok else 'FAIL'}")

    # Documented soft spot: cohort correlation across slots (not gated).
    print("\ncohort stressor (cross-slot correlation q=0.5; documented soft spot):")
    for kind in ("lognormal:1.1",):
        rng = random.Random(77)
        sampler = make_sampler(kind, 2000.0, 400.0, rng)
        trace_i, trace_l = zip(*(sampler() for _ in range(TRACE_SIZE)))
        model = JointTraceModel(list(trace_i), list(trace_l))
        n_star, _ = chernoff_n_star(model, C_TOKENS, EPSILON)
        frac, util, p99 = simulate(sampler, n_star, random.Random(78), cohort_q=0.5)
        print(f"  {kind}: N*={n_star} overflow={frac:.4f} util={util:.2f} p99util={p99:.2f} "
              f"({'still within eps' if frac <= EPSILON else 'exceeds eps, as documented'})")

    pass_rate = sum(rows) / len(rows)
    print(f"\ngrid pass rate: {pass_rate:.0%} ({sum(rows)}/{len(rows)}), acceptance >= 95%")
    for f in failures:
        print(f"  FAIL detail: {f}")
    # Gaussian comparison on the heaviest tail, for the envelope narrative.
    rng = random.Random(4242)
    sampler = make_sampler("lognormal:1.4", 2000.0, 400.0, rng)
    ti, tl = zip(*(sampler() for _ in range(TRACE_SIZE)))
    m = JointTraceModel(list(ti), list(tl))
    n_g = gaussian_n_star(m.e_f, m.var_f, C_TOKENS, 2.33)  # one-sided z for eps=0.01
    fg, _, _ = simulate(sampler, n_g, random.Random(4243))
    print(f"\ngaussian-at-z2.33 on lognormal:1.4: N={n_g} overflow={fg:.4f} "
          f"(vs eps={EPSILON}; shows why the Chernoff path is the default)")
    return 0 if pass_rate >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
