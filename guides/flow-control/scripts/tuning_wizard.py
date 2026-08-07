#!/usr/bin/env python3
"""
Flow Control Tuning Wizard (v2)

Derives saturation-detector limits for llm-d flow control with quantified
confidence. Runs on the Python standard library alone. The math and its
assumptions are derived in ../tuning-theory.md; output labels state which
data source produced each number.

Input sources (best first):
  --from-benchmark JSON   per_request_lifecycle_metrics.json from any
                          llm-d-benchmark inference-perf run: exact joint stats
  --trace FILE.csv        per-request `isl,osl` rows: exact joint statistics
  --from-prometheus URL   vLLM metric histograms (marginals only)
  --isl-mean/... flags    summary moments (weakest source)

Modes (--mode): requests | tokens | hybrid | utilization
Run with no arguments for the interactive triage.
Machine-readable output: --json OUT.json (or '-' for stdout).
"""

import argparse
import csv
import json
import math
import random
import sys
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

# ==========================================
# Defaults, tagged (tuning-theory.md: "The capacity constant" tag scheme).
#   convention — starting value with no derivation; calibrate when possible
#   derived    — computed from stated inputs; the comment says which
#   policy     — a risk choice, not a measurement
#   stand-in   — must be measured on your deployment; default only unblocks a dry run
# ==========================================

ETA_PACKING = 0.90          # convention: KV packing efficiency; calibrate by regression
BLOCK_SIZE = 16             # convention: vLLM default PagedAttention block size
MNBT = 2048                 # convention: vLLM default max_num_batched_tokens
LOOKAHEAD_CAP_FRAC = 0.15   # convention: lookahead buffer cap as a share of the batch
TAU_MARGIN = 0.05           # convention: utilization-threshold damping margin
EPSILON = 0.001             # policy: overflow-probability target
CONFIDENCE = 0.95           # policy: bootstrap LCB level
DEADTIME_SEC = 0.30         # derived: 50ms EPP refresh + 200ms staleness + engine latency
DECODE_RATE = 20.0          # stand-in: tokens/s per active sequence
DISPATCH_RATE = 200.0       # stand-in: peak pool admissions/s
BOOTSTRAP_REPS = 200        # convention: resamples for the LCB
BOOTSTRAP_SEED = 17         # convention: fixed for reproducible output

# ==========================================
# Small numerics (stdlib only)
# ==========================================

def logsumexp(vals: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    m = max(vals)
    if m == -math.inf:
        return -math.inf
    if weights is None:
        return m + math.log(sum(math.exp(v - m) for v in vals))
    return m + math.log(sum(w * math.exp(v - m) for v, w in zip(vals, weights)))

def wmean(xs: Sequence[float], ws: Sequence[float], power: int = 1) -> float:
    tot = sum(ws)
    return sum(w * (x ** power) for x, w in zip(xs, ws)) / tot

def percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return math.nan
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

def solve3(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Gaussian elimination for the 3x3 normal equations of the hinge fit."""
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][3] / m[i][i] for i in range(3)]

# ==========================================
# Workload models (tuning-theory.md: "The stationary footprint model")
#
# F = footprint of a request observed in flight = I + U * L with the pair
# (I, L) size-biased by L and U ~ Uniform(0,1). Every model implements
# WorkloadModel; a new data source only needs these five members.
# ==========================================

class WorkloadModel(Protocol):
    tier: str          # human-readable provenance, printed with every result
    has_mgf: bool      # whether log_mgf_inflight is usable for the Chernoff bound
    mean_L: float      # E[L], output-length mean
    e_f: float         # E[F], mean in-flight footprint (tokens)
    var_f: float       # Var(F)

    def log_mgf_inflight(self, theta: float) -> float: ...

class JointTraceModel:
    """Exact statistics from per-request (isl, osl) samples. Tier 1."""

    tier = "from a joint trace: exact statistics"
    has_mgf = True

    def __init__(self, isl: List[float], osl: List[float]):
        if len(isl) != len(osl) or not isl:
            raise ValueError("trace must contain matched, non-empty isl/osl columns")
        self.isl, self.osl = isl, osl
        self.mean_L = sum(osl) / len(osl)
        e_il = sum(i * l for i, l in zip(isl, osl)) / len(isl)
        e_l2 = sum(l * l for l in osl) / len(osl)
        self.e_f = (e_il + e_l2 / 2.0) / self.mean_L
        e_i2l = sum(i * i * l for i, l in zip(isl, osl)) / len(isl)
        e_il2 = sum(i * l * l for i, l in zip(isl, osl)) / len(isl)
        e_l3 = sum(l ** 3 for l in osl) / len(osl)
        self.var_f = max(0.0, (e_i2l + e_il2 + e_l3 / 3.0) / self.mean_L - self.e_f ** 2)

    def log_mgf_inflight(self, theta: float) -> float:
        # E[e^{tF}] = E[e^{tI}(e^{tL}-1)] / (t E[L]); log1p keeps the tail exact.
        terms = [theta * i + theta * l + math.log1p(-math.exp(-theta * l))
                 for i, l in zip(self.isl, self.osl) if l > 0]
        return (logsumexp(terms) - math.log(len(terms))
                - math.log(theta * self.mean_L))

    def resample(self, rng: random.Random, cap: int = 2000) -> "JointTraceModel":
        n = min(len(self.isl), cap)
        idx = [rng.randrange(len(self.isl)) for _ in range(n)]
        return JointTraceModel([self.isl[j] for j in idx], [self.osl[j] for j in idx])


class MarginalModel:
    """Weighted marginal distributions (e.g. Prometheus histogram buckets),
    combined under an independence assumption. Tier 2 (independence forced:
    marginals cannot carry ISL/OSL correlation)."""

    tier = "from Prometheus marginals: independence assumed"
    has_mgf = True

    def __init__(self, isl_dist: List[Tuple[float, float]], osl_dist: List[Tuple[float, float]],
                 mean_l_conservative: Optional[float] = None):
        self.isl_dist = [(x, w) for x, w in isl_dist if w > 0]
        self.osl_dist = [(x, w) for x, w in osl_dist if w > 0 and x > 0]
        ix, iw = zip(*self.isl_dist)
        lx, lw = zip(*self.osl_dist)
        e_i, e_i2 = wmean(ix, iw), wmean(ix, iw, 2)
        e_l, e_l2, e_l3 = wmean(lx, lw), wmean(lx, lw, 2), wmean(lx, lw, 3)
        # A smaller E[L] in the MGF denominator is the conservative direction.
        self.mean_L = mean_l_conservative if mean_l_conservative else e_l
        self.e_f = e_i + e_l2 / (2.0 * e_l)
        e_f2 = e_i2 + e_i * e_l2 / e_l + e_l3 / (3.0 * e_l)
        self.var_f = max(0.0, e_f2 - self.e_f ** 2)

    def log_mgf_inflight(self, theta: float) -> float:
        ix, iw = zip(*self.isl_dist)
        lx, lw = zip(*self.osl_dist)
        log_e_ti = logsumexp([theta * x for x in ix], iw) - math.log(sum(iw))
        log_e_tl_m1 = (logsumexp([theta * x + math.log1p(-math.exp(-theta * x)) for x in lx], lw)
                       - math.log(sum(lw)))
        return log_e_ti + log_e_tl_m1 - math.log(theta * self.mean_L)


class MomentsModel:
    """Summary moments only. Size-biased corrections use gamma moment ratios;
    correlation is applied to E[I*L] only. Tier 2. The parametric gamma MGF
    gives a Chernoff bound under the distributional assumption."""

    tier = "from summary moments: gamma assumption"
    has_mgf = True

    def __init__(self, isl_mean: float, isl_std: float, osl_mean: float, osl_std: float,
                 rho: float = 0.0):
        self.im, self.istd, self.lm, self.lstd, self.rho = isl_mean, isl_std, osl_mean, osl_std, rho
        cv2 = (osl_std / osl_mean) ** 2 if osl_mean > 0 else 0.0
        self.mean_L = osl_mean
        e_l2 = osl_mean ** 2 * (1 + cv2)
        e_l3 = osl_mean ** 3 * (1 + cv2) * (1 + 2 * cv2)
        e_il = isl_mean * osl_mean + rho * isl_std * osl_std
        self.e_f = (e_il + e_l2 / 2.0) / osl_mean
        e_i2 = isl_mean ** 2 + isl_std ** 2
        e_f2 = e_i2 + isl_mean * e_l2 / osl_mean + e_l3 / (3.0 * osl_mean)
        self.var_f = max(0.0, e_f2 - self.e_f ** 2)
        # Gamma parameterization for the parametric MGF (theta < 1/scale).
        self._gi = self._gamma_params(isl_mean, isl_std)
        self._gl = self._gamma_params(osl_mean, osl_std)

    @staticmethod
    def _gamma_params(mean: float, std: float) -> Tuple[float, float]:
        if mean <= 0:
            return (0.0, 0.0)
        std = max(std, 1e-9)
        scale = std ** 2 / mean
        return (mean / scale, scale)  # (shape k, scale beta)

    def theta_max(self) -> float:
        betas = [b for _, b in (self._gi, self._gl) if b > 0]
        return 0.999 / max(betas) if betas else math.inf

    def log_mgf_inflight(self, theta: float) -> float:
        if theta >= self.theta_max():
            return math.inf
        ki, bi = self._gi
        kl, bl = self._gl
        log_e_ti = -ki * math.log1p(-theta * bi) if bi > 0 else 0.0
        e_tl = math.exp(-kl * math.log1p(-theta * bl)) if bl > 0 else 1.0
        if e_tl <= 1.0:
            return -math.inf
        return log_e_ti + math.log(e_tl - 1.0) - math.log(theta * self.mean_L)

# ==========================================
# The memory wall (tuning-theory.md: Gaussian + Chernoff bounds)
# ==========================================

def gaussian_n_star(e_f: float, var_f: float, c_tokens: float, z: float) -> int:
    """Largest N with N*E[F] + z*sqrt(N)*sigma_F <= C. Tier 1 given moments."""
    if e_f <= 0 or c_tokens <= 0:
        return 0
    s = math.sqrt(var_f)
    y = (-z * s + math.sqrt(z * z * var_f + 4 * e_f * c_tokens)) / (2 * e_f)
    return max(0, int(y * y))

def chernoff_n_star(model: WorkloadModel, c_tokens: float, eps: float) -> Tuple[int, float]:
    """Many-sources bound: N* = max N s.t. sup_theta [theta*C - N*Lambda(theta)]
    >= ln(1/eps). Returns (N*, argmax theta). Tier 1 for empirical models."""
    log_inv_eps = math.log(1.0 / eps)
    hi = getattr(model, "theta_max", lambda: math.inf)()
    theta_hi = min(hi, 40.0 / max(model.e_f, 1.0))
    thetas = [theta_hi * (10 ** (-(i / 24.0))) for i in range(120)]
    best_n, best_t = 0, thetas[-1]
    for t in thetas:
        lam = model.log_mgf_inflight(t)
        if not math.isfinite(lam) or lam <= 0:
            continue
        n = (t * c_tokens - log_inv_eps) / lam
        if n > best_n:
            best_n, best_t = n, t
    return max(0, int(best_n)), best_t

def bootstrap_lcb(trace: JointTraceModel, c_tokens: float, eps: float,
                  reps: int, seed: int, confidence: float) -> Tuple[int, int]:
    """Percentile bootstrap over the trace; the LCB absorbs finite-sample
    error. Returns (lower confidence bound on N*, resamples used)."""
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        m = trace.resample(rng)
        n, _ = chernoff_n_star(m, c_tokens, eps)
        vals.append(float(n))
    vals.sort()
    return int(percentile(vals, 1.0 - confidence)), reps

# ==========================================
# The compute wall (Tier 3: measured knee, or Little's law at an operating point)
# ==========================================

def little_point(throughput_rps: float, latency_sec: float) -> int:
    return int(math.floor(throughput_rps * latency_sec))

def fit_knee(pairs: List[Tuple[float, float]],
             boot: int = 200, seed: int = 7) -> Tuple[float, float, float]:
    """Two-segment hinge regression of TPOT on concurrency:
    y = a + b*x + c*max(0, x - k). Grid over k, OLS per k; bootstrap CI on k.
    Returns (knee, ci_low, ci_high)."""
    def best_k(sample: List[Tuple[float, float]]) -> Optional[float]:
        xs = sorted({x for x, _ in sample})
        if len(xs) < 4:
            return None
        best = (math.inf, None)
        for k in xs[1:-1]:
            sx = [[0.0] * 3 for _ in range(3)]
            sy = [0.0] * 3
            for x, y in sample:
                row = [1.0, x, max(0.0, x - k)]
                for i in range(3):
                    sy[i] += row[i] * y
                    for j in range(3):
                        sx[i][j] += row[i] * row[j]
            coef = solve3(sx, sy)
            if coef is None:
                continue
            sse = sum((y - (coef[0] + coef[1] * x + coef[2] * max(0.0, x - k))) ** 2
                      for x, y in sample)
            if sse < best[0]:
                best = (sse, k)
        return best[1]

    knee = best_k(pairs)
    if knee is None:
        raise ValueError("need >= 4 distinct concurrency levels in --sweep to fit a knee")
    rng = random.Random(seed)
    ks = []
    for _ in range(boot):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        k = best_k(s)
        if k is not None:
            ks.append(k)
    ks.sort()
    return knee, percentile(ks, 0.05), percentile(ks, 0.95)

# ==========================================
# Mode-specific sizing (tuning-theory.md: "Per-mode configuration")
# ==========================================

def tokens_mode_sizing(model, c_tokens: float, z: float,
                       n_requests: Optional[int] = None) -> Dict[str, float]:
    """maxTokenConcurrency for addEstimatedOutputTokens accounting.
    Calibrates outputRatio and margins by ratio-residual noise. Tier 2.

    The router books FULL prompt tokens (its meter reads the tokenized prompt,
    cached prefix included), so `model` here must carry FULL input lengths —
    never the prefix-discounted ones used for the KV memory wall. When
    `n_requests` (the request-wall limit) is given, the ceiling is expressed in
    the router's booked units as n_requests x E[booked tokens/request]: under
    prefix caching, booked tokens overstate KV held, and a KV-derived token
    ceiling would strangle admission far below the request wall."""
    if isinstance(model, JointTraceModel):
        ratio = sum(model.osl) / max(1e-9, sum(model.isl))
        resid = [l - ratio * i for i, l in zip(model.isl, model.osl)]
        mu_r = sum(resid) / len(resid)
        sigma_r = math.sqrt(sum((r - mu_r) ** 2 for r in resid) / max(1, len(resid) - 1))
    elif isinstance(model, MomentsModel):
        ratio = model.lm / max(1e-9, model.im)
        sigma_r = model.lstd  # no joint info: residual ~ full OSL spread (conservative)
    else:
        ix, iw = zip(*model.isl_dist)
        lx, lw = zip(*model.osl_dist)
        ratio = wmean(lx, lw) / max(1e-9, wmean(ix, iw))
        sigma_r = math.sqrt(max(0.0, wmean(lx, lw, 2) - wmean(lx, lw) ** 2))
    if n_requests is not None:
        # Booked units: full input + estimated output per request, ceiling at
        # the request wall. Margin covers estimation noise on the output term.
        if isinstance(model, JointTraceModel):
            mean_isl_full = sum(model.isl) / len(model.isl)
        elif isinstance(model, MomentsModel):
            mean_isl_full = model.im
        else:
            ix, iw = zip(*model.isl_dist)
            mean_isl_full = wmean(ix, iw)
        # booked/request = I + round(I x ratio) -> mean I x (1 + ratio)
        mean_booked = mean_isl_full * (1.0 + ratio)
        margin = z * sigma_r * math.sqrt(max(1.0, n_requests))
        return {"output_ratio": ratio, "sigma_resid": sigma_r,
                "expected_concurrency": float(n_requests),
                "max_token_concurrency": max(0, int(n_requests * mean_booked + margin))}
    s_expected = max(1.0, c_tokens / model.e_f)
    margin = z * sigma_r * math.sqrt(s_expected)
    return {"output_ratio": ratio, "sigma_resid": sigma_r,
            "expected_concurrency": s_expected,
            "max_token_concurrency": max(0, int(c_tokens - margin))}

def utilization_tau(c_tokens: float, isl_mean: float, endpoints: int,
                    dispatch_rate: float, active_seqs: float, decode_rate: float,
                    deadtime_sec: float, margin: float) -> Dict[str, float]:
    """kvCacheUtilThreshold derating for the telemetry dead time. Tier 1
    structure over Tier 2 inputs (supply observed rates where possible)."""
    admit_term = dispatch_rate * deadtime_sec * isl_mean / max(1, endpoints)
    drift_term = active_seqs * decode_rate * deadtime_sec
    tau = 1.0 - (admit_term + drift_term) / c_tokens - margin
    return {"tau": tau, "admit_term_tokens": admit_term, "drift_term_tokens": drift_term,
            "deadtime_sec": deadtime_sec}

def lookahead_buffer(active_batch: int, max_num_batched_tokens: int,
                     isl_mean: Optional[float]) -> int:
    """Engine local-queue allowance: ~one chunked-prefill batch of requests,
    capped at LOOKAHEAD_CAP_FRAC of the active batch."""
    cap = max(1, math.ceil(active_batch * LOOKAHEAD_CAP_FRAC))
    if not isl_mean or isl_mean <= 0:
        return cap
    return max(1, min(math.ceil(max_num_batched_tokens / isl_mean), cap))

def sensitivity_tornado(im: float, istd: float, lm: float, lstd: float,
                        c_tokens: float, z: float, rho: float) -> List[Tuple[str, int, int]]:
    """+/-20% one-at-a-time on the Gaussian N* (fast, labeled approximate).
    Returns [(input, n_low, n_high)] sorted by swing."""
    base_args = {"isl_mean": im, "isl_std": istd, "osl_mean": lm, "osl_std": lstd}
    def n_of(c: float, **over) -> int:
        a = dict(base_args)
        a.update(over)
        m = MomentsModel(rho=rho, **a)
        return gaussian_n_star(m.e_f, m.var_f, c, z)
    rows = []
    for name in base_args:
        lo = n_of(c_tokens, **{name: base_args[name] * 0.8})
        hi = n_of(c_tokens, **{name: base_args[name] * 1.2})
        rows.append((name, min(lo, hi), max(lo, hi)))
    rows.append(("kv_capacity", min(n_of(c_tokens * 0.8), n_of(c_tokens * 1.2)),
                 max(n_of(c_tokens * 0.8), n_of(c_tokens * 1.2))))
    rows.sort(key=lambda r: r[2] - r[1], reverse=True)
    return rows

# ==========================================
# Prometheus (Tier 0 input): vLLM token histograms + cache config
# ==========================================

PROM_QUERIES = {
    "isl_hist": 'sum by (le) (increase(vllm:request_prompt_tokens_bucket{{{sel}}}[{w}]))',
    "osl_hist": 'sum by (le) (increase(vllm:request_generation_tokens_bucket{{{sel}}}[{w}]))',
    "isl_mean": ('sum(increase(vllm:request_prompt_tokens_sum{{{sel}}}[{w}])) / '
                 'sum(increase(vllm:request_prompt_tokens_count{{{sel}}}[{w}]))'),
    "osl_mean": ('sum(increase(vllm:request_generation_tokens_sum{{{sel}}}[{w}])) / '
                 'sum(increase(vllm:request_generation_tokens_count{{{sel}}}[{w}]))'),
    "cache_info": 'vllm:cache_config_info{{{sel}}}',
}

def prom_query(base_url: str, promql: str) -> list:
    url = base_url.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"query": promql})
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.load(resp)
    if payload.get("status") != "success":
        raise RuntimeError(f"prometheus query failed: {promql}")
    return payload["data"]["result"]

def decumulate(hist: list, conservative_upper: bool) -> List[Tuple[float, float]]:
    """Cumulative {le -> count} buckets to [(value, weight)]. Conservative
    binning: upper bucket edge for MGF/moment inputs (over-weights the tail);
    the +Inf residue is placed at 2x the last finite edge with a warning."""
    edges = []
    for series in hist:
        le = series["metric"].get("le", "")
        val = float(series["value"][1])
        edges.append((math.inf if le in ("+Inf", "Inf") else float(le), val))
    edges.sort(key=lambda e: e[0])
    out, prev_edge, prev_cum = [], 0.0, 0.0
    for edge, cum in edges:
        w = max(0.0, cum - prev_cum)
        if w > 0:
            if math.isinf(edge):
                out.append((prev_edge * 2.0, w))
                print(f"  [!] {w:.0f} samples beyond the last histogram bucket; "
                      f"booked at {prev_edge * 2.0:.0f} tokens (conservative).")
            else:
                out.append((edge if conservative_upper else (prev_edge + edge) / 2.0, w))
        prev_edge, prev_cum = (edge if not math.isinf(edge) else prev_edge), cum
    return out

def _prom_scalar(url: str, promql: str) -> Optional[float]:
    try:
        res = prom_query(url, promql)
        v = float(res[0]["value"][1])
        return v if math.isfinite(v) and v > 0 else None
    except Exception:
        return None

def _anchor_mean(dist: List[Tuple[float, float]], exact_mean: Optional[float],
                 label: str) -> List[Tuple[float, float]]:
    """Rescale a decumulated distribution so its mean matches the histogram's
    exact _sum/_count mean. Bucket edges quantize coarsely (a value of 7200
    books as 10000 when a whole workload lands in one bucket); the exact mean
    is in the source and beats the edge. Shape (and the conservative
    upper-edge tail) is preserved; only the scale moves."""
    if not exact_mean or not dist:
        return dist
    x, w = zip(*dist)
    bucket_mean = wmean(x, w)
    if bucket_mean <= 0:
        return dist
    f = exact_mean / bucket_mean
    if abs(f - 1.0) > 0.02:
        print(f"  [i] {label}: histogram-edge mean {bucket_mean:.0f} rescaled to the "
              f"exact _sum/_count mean {exact_mean:.0f} (factor {f:.2f}).")
    return [(v * f, wt) for v, wt in dist]

def from_prometheus(url: str, window: str,
                    selector: str = "") -> Tuple[MarginalModel, Optional[int], Optional[int]]:
    isl = decumulate(prom_query(url, PROM_QUERIES["isl_hist"].format(w=window, sel=selector)), True)
    osl = decumulate(prom_query(url, PROM_QUERIES["osl_hist"].format(w=window, sel=selector)), True)
    isl = _anchor_mean(isl, _prom_scalar(url, PROM_QUERIES["isl_mean"].format(w=window, sel=selector)), "ISL")
    osl = _anchor_mean(osl, _prom_scalar(url, PROM_QUERIES["osl_mean"].format(w=window, sel=selector)), "OSL")
    # Lower-edge E[L] is the conservative direction for the MGF denominator.
    osl_lower = [(max(1.0, x / 2.0), w) for x, w in osl]
    mean_l_cons = wmean(*zip(*osl_lower))
    blocks = block_size = None
    for series in prom_query(url, PROM_QUERIES["cache_info"].format(w="", sel=selector)):
        m = series["metric"]
        blocks = int(float(m.get("num_gpu_blocks", blocks or 0))) or blocks
        block_size = int(float(m.get("block_size", block_size or 0))) or block_size
    return MarginalModel(isl, osl, mean_l_conservative=mean_l_cons), blocks, block_size

# ==========================================
# Config snippets (schema per llm-d-router main; flowControl-nested detector ref)
# ==========================================

def yaml_snippet(mode: str, n: Optional[int], t: Optional[int],
                 output_ratio: Optional[float], tau: Optional[float]) -> str:
    if mode == "utilization":
        return f"""apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: utilization-detector
  parameters:
    kvCacheUtilThreshold: {tau:.2f}   # dead-time-derated; see tuning-theory.md
    queueDepthThreshold: 5
flowControl:
  saturationDetector:
    pluginRef: utilization-detector
  defaultRequestTTL: "60s"
# Flow control is enabled explicitly: featureGates: ["flowControl"]"""
    lines = ["apiVersion: llm-d.ai/v1alpha1", "kind: EndpointPickerConfig", "plugins:"]
    if mode in ("tokens", "hybrid"):
        lines += ["- type: inflight-load-producer",
                  "  parameters:",
                  "    addEstimatedOutputTokens: true",
                  f"    outputRatio: {output_ratio:.2f}   # calibrated from your workload"]
    lines += ["- type: concurrency-detector", "  parameters:"]
    if mode in ("requests", "hybrid"):
        lines.append(f"    maxConcurrency: {n}            # per endpoint")
    if mode in ("tokens", "hybrid"):
        lines.append(f"    maxTokenConcurrency: {t}   # per endpoint")
    lines += [f"    concurrencyMode: {mode}",
              "    headroom: 0.0   # scheduling-filter slack only; never affects saturation",
              "flowControl:",
              "  saturationDetector:",
              "    pluginRef: concurrency-detector",
              '  defaultRequestTTL: "60s"',
              '# Enable the layer explicitly: featureGates: ["flowControl"]']
    return "\n".join(lines)

# ==========================================
# HTML report (inline SVG, no dependencies; palette per the llm-d docs viz set)
# ==========================================

_REPORT_CSS = """
.viz-root { color-scheme: light; font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  --surface-1:#fcfcfb; --page:#f9f9f7; --ink-1:#0b0b0b; --ink-2:#52514e; --ink-mut:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --series-1:#2a78d6; --series-2:#eb6834;
  --status-crit:#d03b3b; background:var(--page); color:var(--ink-1); margin:0; padding:24px; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root {
  color-scheme: dark; --surface-1:#1a1a19; --page:#0d0d0d; --ink-1:#ffffff; --ink-2:#c3c2b7;
  --ink-mut:#898781; --grid:#2c2c2a; --axis:#383835; --series-1:#3987e5; --series-2:#d95926;
  --status-crit:#d03b3b; } }
.viz-root section { background:var(--surface-1); border:1px solid var(--grid);
  border-radius:8px; padding:16px 20px; margin:0 auto 20px; max-width:860px; }
.viz-root h1 { font-size:20px; } .viz-root h2 { font-size:16px; }
.viz-root .sub { color: var(--ink-2); }
.viz-root table { border-collapse:collapse; margin-top:8px; font-variant-numeric:tabular-nums; }
.viz-root td, .viz-root th { padding:3px 12px 3px 0; text-align:left; color:var(--ink-2); }
.viz-root th { color:var(--ink-mut); font-weight:500; }
.viz-root svg text { fill:var(--ink-mut); font-size:11px; }
.viz-root svg .lbl { fill:var(--ink-2); font-size:12px; }
"""

def _svg_axes(w: int, h: int, pad: int, gridlines_y: List[float]) -> str:
    parts = [f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="var(--axis)"/>']
    for gy in gridlines_y:
        parts.append(f'<line x1="{pad}" y1="{gy:.1f}" x2="{w-pad}" y2="{gy:.1f}" '
                     'stroke="var(--grid)" stroke-width="1"/>')
    return "".join(parts)

def chart_two_walls(e_f: float, sd_f: float, z: float, c_tokens: float,
                    n_star: int, knee: Optional[float]) -> str:
    """Expected occupancy (with z-sigma band) vs concurrency; capacity line;
    chosen N* marker; compute knee marker when known. Single series."""
    w, h, pad = 820, 300, 46
    n_max = max(int(n_star * 1.6), int(knee * 1.3) if knee else 0, 10)
    y_max = max(c_tokens * 1.25, (e_f * n_max + z * math.sqrt(n_max) * sd_f))
    def X(n): return pad + (w - 2 * pad) * n / n_max
    def Y(v): return (h - pad) - (h - 2 * pad) * v / y_max
    pts, band_up, band_dn = [], [], []
    for i in range(0, n_max + 1, max(1, n_max // 80)):
        mu = e_f * i
        s = z * math.sqrt(i) * sd_f
        pts.append(f"{X(i):.1f},{Y(mu):.1f}")
        band_up.append(f"{X(i):.1f},{Y(mu + s):.1f}")
        band_dn.append(f"{X(i):.1f},{Y(max(0, mu - s)):.1f}")
    band = " ".join(band_up + band_dn[::-1])
    ticks = "".join(f'<text x="{X(n):.0f}" y="{h-pad+16}" text-anchor="middle">{n}</text>'
                    for n in range(0, n_max + 1, max(1, n_max // 5)))
    knee_mark = ""
    if knee:
        knee_mark = (f'<line x1="{X(knee):.1f}" y1="{pad}" x2="{X(knee):.1f}" y2="{h-pad}" '
                     'stroke="var(--series-2)" stroke-width="2" stroke-dasharray="6 4">'
                     '<title>Measured compute knee</title></line>'
                     f'<text class="lbl" x="{X(knee)+6:.0f}" y="{pad+14}">compute knee {knee:.0f}</text>')
    return f"""<svg viewBox="0 0 {w} {h}" role="img" aria-label="Expected KV occupancy versus concurrency">
{_svg_axes(w, h, pad, [Y(c_tokens)])}
<polygon points="{band}" fill="var(--series-1)" opacity="0.15"/>
<polyline points="{' '.join(pts)}" fill="none" stroke="var(--series-1)" stroke-width="2">
<title>Expected in-flight KV occupancy, z-band shaded</title></polyline>
<line x1="{pad}" y1="{Y(c_tokens):.1f}" x2="{w-pad}" y2="{Y(c_tokens):.1f}"
 stroke="var(--status-crit)" stroke-width="2" stroke-dasharray="2 4">
<title>KV capacity C_tokens = {c_tokens:.0f}</title></line>
<text class="lbl" x="{pad+4}" y="{Y(c_tokens)-6:.0f}">capacity {c_tokens:.0f} tokens</text>
<line x1="{X(n_star):.1f}" y1="{pad}" x2="{X(n_star):.1f}" y2="{h-pad}"
 stroke="var(--series-1)" stroke-width="2"><title>Chosen limit N* = {n_star}</title></line>
<text class="lbl" x="{X(n_star)+6:.0f}" y="{pad+30}">N* = {n_star}</text>
{knee_mark}{ticks}
<text x="{w/2:.0f}" y="{h-8}" text-anchor="middle">concurrent requests per endpoint</text>
</svg>"""

def chart_footprint_hist(model, c_over_n: float) -> str:
    """In-flight footprint distribution; the region beyond C/N* is the
    per-request share that overflows. Single series + threshold."""
    w, h, pad, bins = 820, 260, 46, 36
    if isinstance(model, JointTraceModel):
        rng = random.Random(11)
        tot_l = sum(model.osl)
        # Size-biased sample with uniform age: weight by L, footprint I + U*L.
        samples = []
        for _ in range(4000):
            r = rng.random() * tot_l
            acc = 0.0
            for i, l in zip(model.isl, model.osl):
                acc += l
                if acc >= r:
                    samples.append(i + rng.random() * l)
                    break
    else:
        mu, sd = model.e_f, math.sqrt(model.var_f)
        rng = random.Random(11)
        samples = [max(0.0, rng.gauss(mu, sd)) for _ in range(4000)]
    x_max = max(max(samples), c_over_n) * 1.05
    counts = [0] * bins
    for s in samples:
        counts[min(bins - 1, int(s / x_max * bins))] += 1
    y_max = max(counts)
    bw = (w - 2 * pad) / bins
    bars = []
    for i, c in enumerate(counts):
        bh = (h - 2 * pad) * c / y_max
        x0 = pad + i * bw
        bars.append(f'<rect x="{x0+1:.1f}" y="{h-pad-bh:.1f}" width="{bw-2:.1f}" height="{bh:.1f}" '
                    f'rx="2" fill="var(--series-1)"><title>{(i)*x_max/bins:.0f}-{(i+1)*x_max/bins:.0f} '
                    f'tokens: {c} of {len(samples)}</title></rect>')
    xt = pad + (w - 2 * pad) * c_over_n / x_max
    return f"""<svg viewBox="0 0 {w} {h}" role="img" aria-label="In-flight footprint distribution">
{_svg_axes(w, h, pad, [])}
{''.join(bars)}
<line x1="{xt:.1f}" y1="{pad}" x2="{xt:.1f}" y2="{h-pad}" stroke="var(--status-crit)"
 stroke-width="2" stroke-dasharray="6 4"><title>Per-request capacity share at N*</title></line>
<text class="lbl" x="{xt+6:.0f}" y="{pad+14}">C / N* = {c_over_n:.0f}</text>
<text x="{w/2:.0f}" y="{h-8}" text-anchor="middle">in-flight KV footprint (tokens, size-biased)</text>
</svg>"""

def chart_tornado(rows: List[Tuple[str, int, int]], base_n: int) -> str:
    w, h_row, pad = 820, 34, 46
    h = pad * 2 + h_row * len(rows)
    lo = min(r[1] for r in rows + [("", base_n, base_n)])
    hi = max(r[2] for r in rows + [("", base_n, base_n)])
    span = max(1, hi - lo)
    def X(v): return pad + 140 + (w - 2 * pad - 140) * (v - lo) / span
    bars = []
    for i, (name, nlo, nhi) in enumerate(rows):
        y = pad + i * h_row
        bars.append(
            f'<text class="lbl" x="{pad}" y="{y+18}">{name}</text>'
            f'<rect x="{X(nlo):.1f}" y="{y+6}" width="{max(2, X(nhi)-X(nlo)):.1f}" height="16" rx="4" '
            f'fill="var(--series-1)"><title>{name} +/-20%: N* ranges {nlo} to {nhi}</title></rect>')
    xb = X(base_n)
    return f"""<svg viewBox="0 0 {w} {h}" role="img" aria-label="Sensitivity of N-star to each input">
{''.join(bars)}
<line x1="{xb:.1f}" y1="{pad-6}" x2="{xb:.1f}" y2="{h-pad+6}" stroke="var(--ink-mut)"
 stroke-width="1"/><text x="{xb:.0f}" y="{pad-12}" text-anchor="middle">base {base_n}</text>
</svg>"""

def results_payload(res: dict) -> dict:
    """The machine-readable subset of a results dict: everything except the
    model object, with tuples normalized for JSON."""
    out = {k: v for k, v in res.items() if k != "model"}
    out["tornado"] = [{"input": n, "n_low": lo, "n_high": hi}
                      for n, lo, hi in (res.get("tornado") or [])]
    if out.get("knee_ci"):
        out["knee_ci"] = list(out["knee_ci"])
    return out

def build_report(res: dict) -> str:
    tor = res.get("tornado") or []
    tor_rows = "".join(f"<tr><td>{n}</td><td>{lo}</td><td>{hi}</td></tr>" for n, lo, hi in tor)
    knee_txt = f"{res['knee']:.0f}" if res.get("knee") else "not measured"
    sections = [f"""<section><h1>Flow Control Tuning Report</h1>
<p class="sub">Mode: <b>{res['mode']}</b> &middot; {res['tier']} &middot; epsilon = {res['eps']:.3g}
&middot; generated by tuning_wizard.py</p>
<table><tr><th>Quantity</th><th>Value</th></tr>
<tr><td>Memory-wall N* (point)</td><td>{res['n_point']}</td></tr>
<tr><td>Memory-wall N* (95% LCB &mdash; configure this)</td><td>{res['n_lcb']}</td></tr>
<tr><td>Compute knee</td><td>{knee_txt}</td></tr>
<tr><td>E[F] in-flight footprint</td><td>{res['e_f']:.0f} tokens</td></tr>
<tr><td>CV of footprint</td><td>{res['cv_f']:.2f}</td></tr>
<tr><td>C_tokens (per endpoint)</td><td>{res['c_tokens']:.0f}</td></tr></table>
<p class="sub">Validate in production: <code>vllm:num_preemptions_total</code> deltas ~ 0 and KV
p99 under watermark at peak. Re-tune when ISL/OSL drift (PSI &gt; 0.2). See tuning-theory.md.</p>
</section>"""]
    sections.append("<section><h2>The two walls</h2>"
                    + chart_two_walls(res["e_f"], math.sqrt(res["var_f"]), res["z"],
                                      res["c_tokens"], res["n_lcb"], res.get("knee"))
                    + "</section>")
    if res["n_lcb"] > 0:
        sections.append("<section><h2>In-flight footprint distribution</h2>"
                        + chart_footprint_hist(res["model"], res["c_tokens"] / res["n_lcb"])
                        + "</section>")
    if tor:
        sections.append("<section><h2>Sensitivity of N* to each input</h2>"
                        + chart_tornado(tor, res["n_point"])
                        + f"<table><tr><th>Input (+/-20%)</th><th>N* low</th><th>N* high</th></tr>"
                        + tor_rows + "</table></section>")
    sections.append(f"<section><h2>Configuration</h2><pre>{res['snippet']}</pre></section>")
    return ('<!doctype html><meta charset="utf-8"><title>Flow Control Tuning Report</title>'
            f"<style>{_REPORT_CSS}</style><body class=\"viz-root\">" + "".join(sections))

# ==========================================
# Self-check (tuning-theory.md: validation ladder, rung 1)
# ==========================================

def self_check() -> int:
    failures = []
    def check(name, ok, detail=""):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    # 1. Exponential identity: E[F] = I + L_mean (CV=1 doubles the naive L/2 term).
    rng = random.Random(42)
    isl = [1000.0] * 20000
    osl = [rng.expovariate(1 / 800.0) for _ in range(20000)]
    m = JointTraceModel(isl, osl)
    check("exponential size-bias identity E[F] = I + Lmean",
          abs(m.e_f - 1800.0) / 1800.0 < 0.03, f"E[F]={m.e_f:.0f} expect ~1800")
    # Variance identity for constant I, exponential L: Var(F) = Lmean^2.
    check("exponential variance identity Var(F) = Lmean^2",
          abs(m.var_f - 800.0 ** 2) / 800.0 ** 2 < 0.08, f"Var={m.var_f:.0f}")
    # Moments model reproduces the same correction.
    mm = MomentsModel(1000, 0, 800, 800)
    check("moments model matches exponential closed form",
          abs(mm.e_f - 1800.0) < 1.0, f"E[F]={mm.e_f:.0f}")

    # 2. Heavy tail: Chernoff must be no more permissive than Gaussian.
    heavy = [math.exp(rng.gauss(math.log(400), 1.2)) for _ in range(20000)]
    mh = JointTraceModel([500.0] * len(heavy), heavy)
    c = 400000.0
    n_g = gaussian_n_star(mh.e_f, mh.var_f, c, 2.0)
    n_c, _ = chernoff_n_star(mh, c, eps=0.0228)  # one-sided z=2 equivalent
    check("Chernoff <= Gaussian on lognormal tail", n_c <= n_g, f"chernoff={n_c} gaussian={n_g}")
    check("Chernoff nontrivial", 0 < n_c <= c / mh.e_f, f"n={n_c}")

    # 3. Bootstrap determinism under a fixed seed.
    a1 = bootstrap_lcb(mh, c, 0.05, reps=30, seed=9, confidence=0.95)
    a2 = bootstrap_lcb(mh, c, 0.05, reps=30, seed=9, confidence=0.95)
    check("bootstrap reproducible with fixed seed", a1 == a2)

    # 4. Knee fit recovers a synthetic breakpoint.
    pairs = [(float(x), 20.0 + (0.05 * x if x < 60 else 0.05 * 60 + 1.5 * (x - 60))
              + rng.gauss(0, 0.8)) for x in range(10, 121, 5) for _ in range(3)]
    knee, klo, khi = fit_knee(pairs, boot=60, seed=3)
    check("hinge fit recovers synthetic knee ~60", 50 <= knee <= 70, f"knee={knee:.0f}")

    # 5. Prefix working set: G groups over E endpoints deduct min(G, 2*ceil(G/E))
    # resident prefixes (spill factor measured on the reference stack; see
    # tuning-theory.md). G=1 must reproduce the single-prefix deduction.
    resident = lambda g, e: min(g, 2 * math.ceil(g / e))
    check("prefix working set: 64 groups / 8 endpoints -> 16 resident",
          resident(64, 8) == 16)
    check("prefix working set: G=1 unchanged", resident(1, 8) == 1)
    check("prefix working set: fewer groups than endpoints stays bounded",
          resident(4, 8) == 2)

    # 6. Client-cap clamp: rows recording more output than the client's
    # max_tokens are physically impossible and must not inflate the tail.
    dirty = JointTraceModel([1000.0] * 1000,
                            [500.0] * 990 + [1500.0] * 10)  # 1% impossible rows at cap 1000
    clean = JointTraceModel(dirty.isl, [min(l, 1000.0) for l in dirty.osl])
    check("client-max-tokens clamp reduces E[F]", clean.e_f < dirty.e_f,
          f"dirty={dirty.e_f:.0f} clean={clean.e_f:.0f}")

    # 7. Histogram mean anchoring: a distribution booked at upper bucket edges
    # rescales to the exact _sum/_count mean, preserving total weight.
    dist = [(10000.0, 480.0)]  # whole workload in one bucket, edge 10000
    anchored = _anchor_mean(dist, 7200.0, "self-check")
    ax, aw = zip(*anchored)
    check("mean anchoring hits the exact mean",
          abs(wmean(ax, aw) - 7200.0) < 1.0, f"mean={wmean(ax, aw):.0f}")
    check("mean anchoring preserves weight", abs(sum(aw) - 480.0) < 1e-9)
    check("mean anchoring no-ops without an exact mean",
          _anchor_mean(dist, None, "self-check") == dist)

    # 8. Token ceiling in booked units: anchored to the request wall, prefix or not.
    mm2 = MomentsModel(800, 100, 800, 200)
    tk = tokens_mode_sizing(mm2, 290000.0, 2.0, n_requests=200)
    booked = 800 * (1.0 + 800/800)
    check("token ceiling anchored to request wall in booked units",
          abs(tk["max_token_concurrency"] - 200*booked) < 0.15*200*booked,
          f"ceiling={tk['max_token_concurrency']} expect ~{200*booked:.0f}")
    check("token ceiling not KV-anchored when request wall given",
          tk["max_token_concurrency"] > 290000, f"{tk['max_token_concurrency']}")

    # 9. The --json contract: payload drops the model and serializes.
    dummy = {"mode": "requests", "model": object(), "knee_ci": (1.0, 2.0),
             "tornado": [("isl_mean", 1, 2)]}
    try:
        json.dumps(results_payload(dummy))
        serializable = True
    except TypeError:
        serializable = False
    check("results payload is JSON-serializable", serializable)

    print(f"\nself-check: {'PASS' if not failures else 'FAIL: ' + ', '.join(failures)}")
    return 1 if failures else 0

# ==========================================
# CLI / interactive triage
# ==========================================

def load_benchmark_report(path: str) -> JointTraceModel:
    """Read inference-perf's per_request_lifecycle_metrics.json (written by any
    llm-d-benchmark inference-perf run). Skips errored requests. Handles both
    the v0.6+ schema (output under info.response_metrics) and older files."""
    with open(path) as f:
        records = json.load(f)
    isl, osl = [], []
    for rec in records:
        if rec.get("error"):
            continue
        info = rec.get("info") or {}
        i = info.get("input_tokens")
        if i is None:
            i = ((info.get("request_metrics") or {}).get("text") or {}).get("input_tokens")
        o = (info.get("response_metrics") or {}).get("output_tokens", info.get("output_tokens"))
        if i and o:
            isl.append(float(i))
            osl.append(float(o))
    return JointTraceModel(isl, osl)


def load_trace(path: str) -> JointTraceModel:
    isl, osl = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip() or not row[0].strip()[0].isdigit():
                continue
            isl.append(float(row[0]))
            osl.append(float(row[1]))
    return JointTraceModel(isl, osl)

def z_from_eps(eps: float) -> float:
    """One-sided normal quantile via bisection on erf (stdlib has no ppf)."""
    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 - math.erf(mid / math.sqrt(2))) > eps:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def ask(prompt: str, default: Optional[str] = None) -> str:
    raw = input(prompt).strip()
    return raw if raw else (default or "")

def main() -> int:
    p = argparse.ArgumentParser(description="Flow Control Tuning Wizard v2 (see tuning-theory.md)")
    src = p.add_argument_group("Workload input (choose one tier)")
    src.add_argument("--from-prometheus", metavar="URL", help="Tier 0: pull vLLM histograms")
    src.add_argument("--window", default="24h", help="Prometheus lookback window")
    src.add_argument("--prom-selector", dest="prom_selector", default="",
                     help='label matcher to scope one pool, e.g. \'namespace="llm-d-flow-control"\'')
    src.add_argument("--print-queries", action="store_true",
                     help="print the PromQL and exit; run them yourself and supply the "
                          "moment flags computed from the histograms")
    src.add_argument("--trace", help="CSV of per-request isl,osl")
    src.add_argument("--from-benchmark", dest="from_benchmark", metavar="JSON",
                     help="per_request_lifecycle_metrics.json from an inference-perf run")
    src.add_argument("--isl-mean", type=float, dest="isl_mean")
    src.add_argument("--isl-std", type=float, dest="isl_std")
    src.add_argument("--osl-mean", type=float, dest="osl_mean")
    src.add_argument("--osl-std", type=float, dest="osl_std")
    src.add_argument("--correlation-coefficient", type=float, dest="rho", default=0.0)

    cap = p.add_argument_group("Capacity (per endpoint)")
    cap.add_argument("--gpu-blocks", type=int, dest="gpu_blocks")
    cap.add_argument("--block-size", type=int, dest="block_size", default=BLOCK_SIZE)
    cap.add_argument("--paged-attention-efficiency", type=float, dest="eta", default=ETA_PACKING,
                     help="packing efficiency; calibrate from telemetry when possible")
    cap.add_argument("--shared-prefix", type=int, dest="shared_prefix", default=0)
    cap.add_argument("--enable-prefix-caching", action="store_true")
    cap.add_argument("--client-max-tokens", type=int, dest="client_max_tokens", default=0,
                     help="the client-side max_tokens cap, if your workload sets one. "
                          "Trace/benchmark rows recording MORE output than the cap are "
                          "physically impossible (a known harness bug records "
                          "input+output as output); they are clamped to the cap and "
                          "counted in a warning.")
    cap.add_argument("--prefix-groups", type=int, dest="prefix_groups", default=1,
                     help="number of DISTINCT shared prefixes in the live working set "
                          "(default 1). Multi-tenant / multi-template workloads keep "
                          "several prefixes resident per endpoint; each one occupies KV "
                          "alongside request footprints.")

    comp = p.add_argument_group("Compute wall (Tier 3)")
    comp.add_argument("--throughput", type=float,
                      help="RPS per replica at a met-SLO operating point")
    comp.add_argument("--latency-sec", type=float, dest="latency_sec")
    comp.add_argument("--sweep", help="CSV of concurrency,tpot_ms rows for the knee fit")
    comp.add_argument("--max-num-batched-tokens", type=int, dest="mnbt", default=MNBT)

    stat = p.add_argument_group("Statistical targets")
    stat.add_argument("--epsilon", type=float, default=EPSILON,
                      help="overflow-probability target (preemption budget)")
    stat.add_argument("--confidence", type=float, default=CONFIDENCE, help="bootstrap LCB level")
    stat.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPS)
    stat.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    stat.add_argument("--z-score", type=float, dest="z_score", default=None,
                      help="override z (default derives one-sided z from --epsilon)")

    mode = p.add_argument_group("Mode & utilization inputs")
    mode.add_argument("--mode", choices=["requests", "tokens", "hybrid", "utilization"],
                      default="requests")
    mode.add_argument("--endpoints", type=int, default=1)
    mode.add_argument("--decode-rate", type=float, dest="decode_rate", default=DECODE_RATE,
                      help="tokens/s per active sequence (utilization mode)")
    mode.add_argument("--dispatch-rate", type=float, dest="dispatch_rate", default=DISPATCH_RATE,
                      help="observed peak pool admissions/s (utilization mode)")
    mode.add_argument("--deadtime", type=float, default=DEADTIME_SEC,
                      help="scrape + staleness + engine metric latency, seconds")
    mode.add_argument("--tau-margin", type=float, dest="tau_margin", default=TAU_MARGIN)

    p.add_argument("--report", metavar="OUT.html", help="write the HTML report")
    p.add_argument("--json", dest="json_out", metavar="OUT.json",
                   help="write results as JSON ('-' for stdout); for CI and config generation")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    # With --json -, stdout is a machine interface: it must carry the JSON
    # payload and nothing else. Route the human-readable narration to stderr.
    real_stdout = sys.stdout
    if args.json_out == "-":
        sys.stdout = sys.stderr

    if args.self_check:
        return self_check()
    if args.print_queries:
        print("# Run these against your Prometheus. The histograms give ISL/OSL")
        print("# distributions: compute mean/std and pass the --isl-*/--osl-* flags,")
        print("# or skip all of this with --from-prometheus.")
        for name, q in PROM_QUERIES.items():
            print(f"{name}: {q.format(w=args.window, sel=args.prom_selector)}")
        return 0

    interactive = len(sys.argv) == 1
    if interactive:
        print("=== Flow Control Tuning Wizard v2 ===")
        print("Derivations & confidence tiers: guides/flow-control/tuning-theory.md\n")
        bench = ask("Benchmark report per_request_lifecycle_metrics.json (blank to skip): ")
        if bench:
            args.from_benchmark = bench
        else:
            url = ask("Prometheus URL (blank to skip): ")
            if url:
                args.from_prometheus = url
            else:
                path = ask("Workload trace CSV `isl,osl` (blank to fall back to moments): ")
                if path:
                    args.trace = path
                args.isl_mean = float(ask("  Mean input tokens (ISL): "))
                args.isl_std = float(ask("  ISL stddev [=mean, exponential-like]: ",
                                         str(args.isl_mean)))
                args.osl_mean = float(ask("  Mean output tokens (OSL): "))
                args.osl_std = float(ask("  OSL stddev [=mean, exponential-like]: ",
                                         str(args.osl_mean)))
        if not args.from_prometheus:
            args.gpu_blocks = int(ask("KV blocks per replica (engine logs / cache_config_info): "))
            args.block_size = int(ask("Tokens per block [16]: ", "16"))
        args.mode = ask("Mode: requests, tokens, hybrid, or utilization "
                        "(tuning.md has the decision tree) [requests]: ", "requests").strip()
        if args.mode == "utilization":
            args.endpoints = int(ask("Model server replicas [1]: ", "1"))
            args.dispatch_rate = float(ask("Peak pool admissions/s "
                                           "[200]: ", "200"))
        thr = ask("Throughput RPS per replica at a met-SLO operating point "
                  "(blank to skip the compute wall): ")
        if thr:
            args.throughput = float(thr)
            args.latency_sec = float(ask("Mean end-to-end latency at that point (seconds): "))
        args.epsilon = float(ask("Overflow target epsilon [0.001]: ", "0.001"))

    # --- Build the workload model ------------------------------------------
    blocks, bsz = args.gpu_blocks, args.block_size
    if args.from_prometheus:
        print(f"Querying {args.from_prometheus} (window {args.window})...")
        model, pb, pbs = from_prometheus(args.from_prometheus, args.window, args.prom_selector)
        blocks, bsz = blocks or pb, pbs or bsz
        print("  [!] Prometheus histograms are marginals: ISL/OSL treated as independent "
              "(Tier 2). Supply --trace for exact joint statistics.")
    elif args.from_benchmark:
        model = load_benchmark_report(args.from_benchmark)
        print(f"Loaded {len(model.isl)} requests from {args.from_benchmark}.")
    elif args.trace:
        model = load_trace(args.trace)
        print(f"Loaded {len(model.isl)} requests from {args.trace}.")
    elif args.isl_mean is not None and args.osl_mean is not None:
        isl_mean = args.isl_mean
        if args.enable_prefix_caching and args.shared_prefix > 0:
            # Resident per-request input is the uncached remainder; the shared
            # prefix is charged once against capacity below.
            isl_mean = max(0.0, isl_mean - min(args.shared_prefix, isl_mean))
        isl_std = args.isl_std if args.isl_std is not None else isl_mean
        osl_std = args.osl_std if args.osl_std is not None else args.osl_mean
        model = MomentsModel(isl_mean, isl_std, args.osl_mean, osl_std, args.rho)
        # Token accounting books FULL prompts; keep an undiscounted twin.
        model_full = MomentsModel(args.isl_mean, isl_std, args.osl_mean, osl_std, args.rho)
    else:
        p.error("provide --from-benchmark, --trace, --from-prometheus, "
                "or --isl-mean/--osl-mean")

    # Physical validation of trace rows against the client output cap.
    if isinstance(model, JointTraceModel):
        if args.client_max_tokens > 0:
            cap_ = float(args.client_max_tokens)
            bad = sum(1 for l in model.osl if l > cap_)
            if bad:
                print(f"  [!] {bad} of {len(model.osl)} rows record more output than the "
                      f"--client-max-tokens cap ({args.client_max_tokens}) — physically "
                      "impossible; clamped to the cap. Verify the trace against the "
                      "engine's vllm:request_generation_tokens histogram.")
                model = JointTraceModel(model.isl, [min(l, cap_) for l in model.osl])
        else:
            med = sorted(model.osl)[len(model.osl) // 2]
            if med > 0 and max(model.osl) > 10 * med:
                print("  [!] OSL max is >10x the median. If your clients set max_tokens, "
                      "pass --client-max-tokens to reject impossible rows; a known "
                      "harness bug records input+output as output for some rows.")
    if not blocks:
        p.error("KV capacity unknown: provide --gpu-blocks (or a Prometheus with cache_config_info)")

    c_tokens = blocks * bsz * args.eta
    marginal_isl_note = ""
    if args.enable_prefix_caching and args.shared_prefix > 0:
        # Capacity holds the RESIDENT prefix working set, not one prefix. With
        # G distinct prefixes spread over E endpoints by affinity routing, each
        # endpoint keeps ~G/E resident, and under load prefixes spill to a
        # second endpoint: measured residency on the reference stack was 1.8x
        # G/E (B0 ladder, 64 groups x 8 endpoints: preemption onset implied
        # 14.4 resident groups/endpoint). The 2x factor below rounds that up
        # conservatively (measured; see RUNLOG Stage B / tuning-theory.md
        # footprint additivity).
        groups = max(1, args.prefix_groups)
        resident = min(groups, 2 * math.ceil(groups / max(1, args.endpoints)))
        c_tokens = max(0.0, c_tokens - resident * args.shared_prefix)
        marginal_isl_note = (f" ({resident} resident prefix(es) x {args.shared_prefix} "
                             f"tokens deducted from capacity)")
        if c_tokens <= 0:
            p.error(f"the resident prefix working set ({resident} x {args.shared_prefix} "
                    "tokens) exceeds per-endpoint KV capacity. Check --prefix-groups and "
                    "--endpoints (residency scales with groups/endpoints), or the prefix "
                    "working set genuinely does not fit and no concurrency limit can help.")
        # Every tier receives FULL prompt lengths (benchmark records and vLLM
        # histograms physically contain them; tuning.md tells the reader the
        # same for --isl-mean). Deduct the cached prefix from the per-request
        # resident input uniformly here. The moments branch already deducted
        # it from isl_mean at model construction; these two branches cover the
        # trace/benchmark and Prometheus tiers, which previously kept the full
        # prompt and under-sized the limit ~5x on prefix-heavy workloads.
        sp = float(args.shared_prefix)
        if 'model_full' not in dir():
            model_full = model
        if isinstance(model, JointTraceModel):
            model = JointTraceModel([max(0.0, i - min(sp, i)) for i in model.isl],
                                    model.osl)
        elif isinstance(model, MarginalModel):
            model = MarginalModel([(max(0.0, x - min(sp, x)), w) for x, w in model.isl_dist],
                                  model.osl_dist,
                                  mean_l_conservative=model.mean_L)

    z = args.z_score if args.z_score is not None else z_from_eps(args.epsilon)
    cv_f = math.sqrt(model.var_f) / model.e_f if model.e_f > 0 else 0.0

    # --- Memory wall --------------------------------------------------------
    n_gauss = gaussian_n_star(model.e_f, model.var_f, c_tokens, z)
    n_point, theta = chernoff_n_star(model, c_tokens, args.epsilon)
    if isinstance(model, JointTraceModel):
        n_lcb, _ = bootstrap_lcb(model, c_tokens, args.epsilon,
                                 args.bootstrap, args.seed, args.confidence)
        lcb_note = f"{args.confidence:.0%} bootstrap LCB over {args.bootstrap} resamples"
    else:
        n_lcb = min(n_point, n_gauss)
        lcb_note = "min(Chernoff, Gaussian); supply --trace for a true bootstrap LCB"

    # --- Compute wall -------------------------------------------------------
    knee = knee_ci = None
    if args.sweep:
        pairs = []
        with open(args.sweep, newline="") as f:
            for row in csv.reader(f):
                if row and row[0].strip() and row[0].strip()[0].isdigit():
                    pairs.append((float(row[0]), float(row[1])))
        knee, klo, khi = fit_knee(pairs)
        # Guard: a sweep that never reaches the degradation regime still yields a
        # hinge fit (typically snapping to the single-request -> batched TPOT step
        # at the bottom of the ladder), and configuring that "knee" would clamp the
        # limit to a tiny value. Require a material TPOT rise beyond the fitted
        # knee before trusting it; 15% mirrors the walls-divergence threshold.
        tpot_at = dict(pairs)
        top_c = max(c for c, _ in pairs)
        knee_level = min((c for c, _ in pairs if c >= knee), default=top_c)
        rise = (tpot_at[top_c] - tpot_at[knee_level]) / max(tpot_at[knee_level], 1e-9)
        if rise < 0.15:
            print(f"  [!] --sweep TPOT rises only {rise:.0%} beyond the fitted knee "
                  f"({knee:.0f}); the sweep never reaches the degradation regime, so "
                  "the knee is not usable. Ignoring the compute wall; extend the sweep "
                  "to higher concurrency and re-run.")
            knee = knee_ci = None
        else:
            knee_ci = (klo, khi)
            knee = klo  # configure the CI lower edge
    elif args.throughput and args.latency_sec:
        knee = float(little_point(args.throughput, args.latency_sec))

    # --- Combine ------------------------------------------------------------
    n_final = int(min(n_lcb, knee)) if knee else n_lcb
    isl_mean_est = (sum(model.isl) / len(model.isl) if isinstance(model, JointTraceModel)
                    else (args.isl_mean or model.e_f - model.mean_L / 2))
    buf = lookahead_buffer(max(1, n_final), args.mnbt, isl_mean_est)
    tok_model = locals().get("model_full") or model
    # The token meter books each request's FULL final footprint (input + estimated
    # output) for its whole lifetime, so the ceiling must be expressed in booked
    # units anchored to the request wall — always, not only under prefix caching.
    # A KV-anchored ceiling (time-average budget) against full-footprint booking
    # double-counts conservatism and lands near the cohort-synchronized wall
    # (~27% low measured on a decode-heavy workload).
    tok = (tokens_mode_sizing(tok_model, c_tokens, z, n_requests=n_final + buf)
           if args.mode in ("tokens", "hybrid") else None)
    tau = None
    if args.mode == "utilization":
        tau_res = utilization_tau(c_tokens, isl_mean_est, args.endpoints, args.dispatch_rate,
                                  active_seqs=max(1.0, c_tokens / model.e_f),
                                  decode_rate=args.decode_rate,
                                  deadtime_sec=args.deadtime, margin=args.tau_margin)
        tau = tau_res["tau"]

    tornado = None
    if isinstance(model, MomentsModel):
        tornado = sensitivity_tornado(model.im, model.istd, model.lm, model.lstd,
                                      c_tokens, z, model.rho)
    elif isinstance(model, JointTraceModel):
        im = sum(model.isl) / len(model.isl)
        istd = math.sqrt(sum((x - im) ** 2 for x in model.isl) / len(model.isl))
        lm = model.mean_L
        lstd = math.sqrt(sum((x - lm) ** 2 for x in model.osl) / len(model.osl))
        tornado = sensitivity_tornado(im, istd, lm, lstd, c_tokens, z, 0.0)

    snippet = yaml_snippet(args.mode, n_final + buf,
                           tok["max_token_concurrency"] if tok else None,
                           tok["output_ratio"] if tok else None, tau)

    # --- Report to terminal --------------------------------------------------
    line = "-" * 62
    print(f"\n{line}\nTUNING RESULTS  [{model.tier}]\n{line}")
    print(f"KV capacity per endpoint:  {c_tokens:,.0f} tokens{marginal_isl_note}")
    print(f"In-flight footprint E[F]:  {model.e_f:,.0f} tokens  (CV = {cv_f:.2f})")
    if args.mode == "utilization":
        pass  # concurrency-limit lines are not part of this mode's output
    else:
        print(f"Memory wall (Chernoff):    N* = {n_point}  (theta* = {theta:.2e}, eps = {args.epsilon})")
    if args.mode != "utilization":
        print(f"Memory wall (Gaussian):    N* = {n_gauss}  (z = {z:.2f}; fallback comparison)")
        print(f"Configure (LCB = lower confidence bound): N* = {n_lcb}  ({lcb_note})")
    if knee and args.mode != "utilization":
        src_txt = (f"hinge fit, 90% CI [{knee_ci[0]:.0f}, {knee_ci[1]:.0f}]" if knee_ci
                   else "Little's law at your operating point; a knee sweep is stronger")
        print(f"Compute wall:              {knee:.0f}  ({src_txt})  [Tier 3]")
        if n_lcb and abs(n_lcb - knee) / max(n_lcb, knee) > 0.15:
            binding = "compute" if knee < n_lcb else "memory"
            print(f"  [!] Walls diverge >15%: the {binding} wall binds. See tuning-theory.md "
                  "cross-check; the idle margin on the other wall is expected.")
    elif args.mode != "utilization":
        print("Compute wall:              not measured  [!] memory-only limit carries TPOT risk")
    if args.mode != "utilization":
        print(f"Lookahead buffer B:        {buf}")
        print(f"Per-endpoint limit:        {n_final} + {buf} = {n_final + buf}")
    if cv_f > 0.6 and args.mode == "requests":
        print(f"  [!] CV_F = {cv_f:.2f}: requests-mode margin strands ~{z * cv_f / math.sqrt(max(1, n_final)):.0%} "
              "of capacity. Consider hybrid or tokens mode (see tuning-theory.md, Choosing a mode).")
    if tok:
        print(f"tokens mode:               maxTokenConcurrency = {tok['max_token_concurrency']:,}"
              f"  (outputRatio = {tok['output_ratio']:.2f}, sigma_resid = {tok['sigma_resid']:.0f})")
    if tau is not None:
        print(f"utilization mode:          kvCacheUtilThreshold <= {tau:.2f} "
              f"(dead time {args.deadtime}s derating)")
        if tau <= 0.5:
            print("  [!] Derating consumed most of the threshold: at your dispatch rate and "
                  "dead time the utilization detector cannot protect this pool. Use the "
                  "concurrency detector.")
    print(f"\nValidate: watch vllm:num_preemptions_total (target ~0/hr at peak) and KV p99.")
    print(f"Re-tune:  when ISL/OSL marginals drift vs this calibration (PSI > 0.2).")
    print(f"\n{line}\nCONFIGURATION SNIPPET\n{line}\n{snippet}\n")
    if args.mode != "utilization":
        print(f"Check the engine: if --max-num-seqs is below {n_final + buf}, the engine "
              "caps concurrency before this limit ever applies.")

    res = {"mode": args.mode, "tier": model.tier, "eps": args.epsilon, "z": z,
           "n_point": n_point, "n_gauss": n_gauss, "n_lcb": n_lcb,
           "e_f": model.e_f, "var_f": model.var_f, "cv_f": cv_f,
           "c_tokens": c_tokens, "eta": args.eta, "endpoints": args.endpoints,
           "knee": knee, "knee_ci": knee_ci, "lookahead_buffer": buf,
           "per_endpoint_limit": (n_final + buf) if args.mode != "utilization" else None,
           "tokens": tok, "tau": tau, "model": model,
           "tornado": tornado, "snippet": snippet}
    if args.report:
        with open(args.report, "w") as f:
            f.write(build_report(res))
        print(f"\nReport written to {args.report}")
    if args.json_out:
        payload = json.dumps(results_payload(res), indent=2)
        if args.json_out == "-":
            sys.stdout = real_stdout
            print(payload)
        else:
            with open(args.json_out, "w") as f:
                f.write(payload + "\n")
            print(f"JSON written to {args.json_out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
