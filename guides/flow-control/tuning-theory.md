# Flow Control Capacity Model

This document derives the limits that the [tuning guide](tuning.md) tells you to configure
and the [tuning wizard](scripts/tuning_wizard.py) computes. Read it when you want to know
where a number comes from, what it assumes, and what observation would prove it wrong.

Terms used throughout. The *endpoint picker* (EPP) is the llm-d router component that admits
and schedules requests for a pool of model-server replicas, called *endpoints*. Its flow
control layer queues requests when a *saturation detector* reports the pool is full. The
detectors need limits: a per-endpoint concurrent-request limit (`maxConcurrency`), a
concurrent-token limit (`maxTokenConcurrency`), or utilization thresholds. Deriving those
limits is the subject here.

Each main result ends with a short note in a fixed form (*Assumes / When wrong / Check*)
stating the assumptions behind it, the direction of error when they fail, and the observable
that detects the failure.

## What each configuration bounds

A continuous-batching engine has three capacity axes, per replica:

| Axis | Hardware constraint | Saturates as | A request's claim |
|---|---|---|---|
| KV-token residency | VRAM (the PagedAttention block pool) | preemption and recompute | grows through decode, released at end of stream |
| Batch slots | `max_num_seqs`, memory bandwidth | inter-token latency degrades | one slot until end of stream |
| Prefill compute | per-iteration token budget (`max_num_batched_tokens`) | time-to-first-token grows | released at the first output token |

Each detector configuration bounds some of these axes and is blind to others:

| Configuration | Bounds | Blind to |
|---|---|---|
| `requests` mode (`maxConcurrency`) | slots directly; KV through request-count statistics | per-request footprint differences |
| `tokens` mode, default accounting | prefill backlog (tokens release at first output token) | KV growth during decode |
| `tokens` mode with `addEstimatedOutputTokens` | end-of-stream KV residency, estimated | error in the output estimate |
| `hybrid` mode | the larger of the request and token ratios, per endpoint | nothing structural; both limits still need values |
| `utilization-detector` thresholds | the measured state, after a delay | whatever happens during the delay |

Tuning reduces to one trade. Pick a scalar limit so that the probability of leaving the
feasible region (all three axes within capacity) stays below a target ε. The price is the
capacity the safety margin gives up. Both quantities are computable, and the margin formula
doubles as the rule for choosing between modes (see [Choosing a mode](#choosing-a-mode)).

The memory axis admits closed-form bounds built from workload statistics and one calibrated
constant. The compute axis does not. Inter-token latency as a function of concurrency depends
on kernels, hardware, and workload shape; it is measured or profiled, never derived. Chunked
prefill keeps the two axes separable, because it makes per-iteration cost close to linear in
token count. Each axis gets its own limit, and the smaller one binds.

## The memory limit

### The population you observe is not the population that arrives

Watch a busy pool at one instant. The requests in flight are not a fair sample of the
requests that arrive: a request that runs ten times longer is ten times more likely to be
present when you look. This is the inspection paradox. It is the same effect that makes a
rider arriving at a random time wait longer than half the average gap between buses.

Capacity math cares about the requests in flight, so it must use this size-biased population.
Formally: write `I` for a request's resident input tokens (after any prefix-cache discount)
and `L` for its output length. KV residency grows roughly linearly through decode, and
lifetime is proportional to `L`. Average the footprint ramp over a lifetime, weight by
lifetime, and the mean footprint of an in-flight request comes out as:

```
E[F] = ( E[I·L] + E[L²]/2 ) / E[L]
     = Ī + (L̄/2)·(1 + CV²)          when I and L are independent
```

where `CV = σ_L / L̄`, the coefficient of variation of output length.

The common shortcut `Ī + L̄/2` is the `CV = 0` case, and the gap is not small. For
exponential output lengths (`CV = 1`) the decode term doubles: `E[F] = Ī + L̄`. Measured
production output lengths are heavier than exponential: per-prompt distributions fit log-t
shapes with P99/P50 ratios near 10 (TIE, arXiv:2604.00499). So the correction is usually
larger still. Omitting it understates occupancy and over-admits.

Correlation between `I` and `L` shifts the cross term as well. Long documents tend to get
long answers, and long-output requests pull their long inputs into the in-flight population
with them. With only a correlation coefficient available, use
`E[I·L] = Ī·L̄ + ρ·σ_I·σ_L`. With a trace, compute the term exactly.

The variance of the in-flight footprint follows the same recipe and involves third moments
of `L`:

```
E[F²] = ( E[I²·L] + E[I·L²] + E[L³]/3 ) / E[L]
Var(F) = E[F²] − E[F]²
```

Third moments are where heavy tails enter. They are also why a trace is worth more than a
mean and a standard deviation.

One more step makes the sum over N in-flight requests tractable: treat the N footprints as
independent draws of `F`. At the operating point where the limit binds, this has a concrete
justification. When flow control holds concurrency at the limit, each of the N slots is a
renewal process, because a completion admits a successor. A snapshot of a renewal process is
exactly a size-biased interval at a uniform age. What the argument does not cover is
correlation across slots. Requests admitted together in a burst share prefixes and phase.

> Assumes: stationary workload; independent slots at the limit.
> When wrong: correlated admissions raise occupancy above the model; the error is in the
> over-admitting direction. Simulation with strongly correlated admissions exceeded an
> ε = 0.01 target by 80% (see [Assumptions and failure modes](#assumptions-and-failure-modes)).
> Check: the preemption counter, and the drift test in [Validation](#validation-and-re-tuning).

### The limit from moments

The limit answers a packing question: how many draws of `F` fit in the pool, with room for
bad luck? The mean occupancy of N requests is `N·E[F]`, and its spread grows as `√N·σ_F`.
Formally: given a per-endpoint token capacity `C` and a one-sided normal quantile `z`, the
largest N whose occupancy stays within capacity at z-sigma confidence solves
`N·E[F] + z·√N·σ_F ≤ C`:

```
√N* = ( −z·σ_F + √(z²·σ²_F + 4·E[F]·C) ) / ( 2·E[F] )
```

When only means and standard deviations are known, the wizard fills in the higher moments of
`L` from a gamma distribution with matching mean and variance.

> Assumes: footprints light-tailed enough for a normal approximation of the sum; gamma-shaped
> higher moments when only two moments are supplied.
> When wrong: heavy-tailed output lengths make both approximations optimistic. The sum's tail
> is set by the largest single request, which no normal curve sees. In simulation, this bound
> at its matched quantile realized twice its target overflow on a lognormal tail.
> Check: supply a trace and compare against the bound below; a large gap means the tail is
> doing work the moments cannot see.

### The limit from a trace

The moments bound sees two numbers per distribution. A trace lets the bound see the whole
shape, tail included. The idea comes from effective-bandwidth theory (Kelly, *Notes on
Effective Bandwidths*, 1996): treat each in-flight slot as a source placing random demand on
a shared link, and price the demand by an exponential moment, so that one very large request
costs what it should.

Formally: with per-request samples `{(I_j, L_j)}`, the moment generating function of the
in-flight footprint has a closed empirical form. Size-biasing and the uniform-age average
cancel into:

```
E[e^{θF}] = E[ e^{θI}·(e^{θL} − 1) ] / ( θ·E[L] )

Λ(θ) = log( Σ_j e^{θ·I_j}·(e^{θ·L_j} − 1) ) − log( θ·Σ_j L_j )
```

The limit is then the classical many-sources admission bound, with each in-flight request as
a source and the KV pool as the shared link:

```
N* = max N  such that  sup_θ [ θ·C − N·Λ(θ) ] ≥ log(1/ε)
```

The objective is concave in θ. A coarse grid with local refinement solves it in milliseconds.

Two properties decide when to use it. First, it is a bound: it does not under-margin any
footprint distribution the trace exhibits. Its cost is conservatism that grows with tail
weight, from near zero at CV ≤ 1 to about 2.5× on the heaviest simulated tail. The normal
approximation fails in the opposite direction, so the choice is between wasting capacity and
missing the ε target. Second, it converges to the moments answer on light-tailed traces.

> Assumes: the trace represents the traffic the limit will face, including its tail.
> When wrong: a trace too short to contain tail events understates risk exactly where it
> matters. Use thousands of requests spanning peak traffic, not hundreds from a quiet hour.
> Check: recompute on a later window; the drift test below formalizes this.

### The capacity constant

```
C = gpu_blocks × block_size × η − shared_prefix_tokens (if the prefix is cached)
```

`η` is a packing efficiency covering block rounding, fragmentation, copy-on-write sharing,
and the engine's protective watermark. It is a property of the deployment. Without data, 0.9
is a convention, not a derivation. With a running system, calibrate it: regress measured KV
utilization against tracked in-flight tokens, and read `η` off the slope. The EPP tracks the
second series; the engine exports the first. The regression residuals are autocorrelated,
because both series ride the same load. Treat the resulting interval as indicative rather
than exact.

### Two kinds of uncertainty

The target ε covers workload randomness at a fixed distribution. A second error comes from
the trace being finite: another sample of the same traffic would give a slightly different
limit. The wizard keeps the two apart. It bootstraps the trace, recomputes `N*` per resample,
and reports the lower confidence bound. Output reads like `N* = 118 (95% LCB; point 132)`,
with illustrative numbers, and the LCB is the value to configure. Each error has its own
remedy: collecting more data tightens the LCB, while changing ε is a policy decision. A
single safety factor that mixes the two hides both remedies.

## The compute limit

Inter-token latency versus concurrency has no closed form. Two ways to locate the knee:

- With a load test: sweep concurrency in steps, fit a two-segment regression of inter-token
  latency on concurrency, and take the breakpoint with a bootstrap confidence interval. Use
  the interval's lower edge.
- Without one: Little's law (`concurrency = throughput × latency`) converts an existing
  benchmark report's operating point into a concurrency reading. It tells you where you ran.
  If latency targets were met there, that is a lower bound on the knee, and nothing more.

The binding limit is the smaller of the memory and compute values. When both are available
they should agree within roughly 10–15% at the operating point. A larger gap is informative.
Memory far below compute means VRAM binds first. Compute far below memory means the memory
margin is idle headroom. The wizard prints this comparison whenever it can.

In the memory-bound regime the compute knee is not measurable at all: the engine's batch
stops growing when KV fills, so inter-token latency rises and then flattens with offered
concurrency, and the curve contains no breakpoint (measured on an 8-replica × TP2
H100-80GB deployment of Qwen3-32B, July 2026 — the sweep's hinge fit lands on the
single-request-to-batched step and the wizard rejects it). Little's-law readings are equally unusable there, because sojourn time inflates with
queueing and the estimate varies with whichever operating point is chosen (113–207 on the
same measured ladder). When preemptions or pinned KV appear before latency degrades, size
against the memory wall alone; the compute wall is hidden behind it and cannot bind.

## Numbers for each mode

All values are per endpoint. The detector multiplies by the endpoint count to form pool
capacity, so use per-replica `gpu_blocks`.

**`requests` mode.** Set `maxConcurrency = min(N*_memory, N_knee)`. The count is a proxy for
footprint, and the whole burden of footprint variation lands in the safety margin.

**`tokens` mode, default accounting.** In-flight tokens count the uncached prompt and release
at the first output token. The limit therefore bounds the prefill backlog and protects
time-to-first-token. It does not bound KV residency: under decode-heavy load the pool fills
with decoding requests the token counter no longer sees. Do not use this as the only
saturation signal; the tuning guide's decision tree never routes here alone. Used as a
prefill guard beside another signal, 2–4 × `max_num_batched_tokens` is the sizing convention
(one iteration executing, one to three staged).

**`tokens` mode with `addEstimatedOutputTokens: true`.** Accounting books
`I + min(round(I × outputRatio), client max_tokens, operator cap)` from dispatch to end of
stream. This is an end-of-stream residency claim charged from the start, so it is
conservative while a request is young. Sizing:

```
maxTokenConcurrency ≈ C − z·σ_resid·√S
```

with `S` the expected concurrent request count and `σ_resid` the standard deviation, in
tokens, of the residual `L − outputRatio·I` over calibration data. Calibrate `outputRatio`
from the same data (`Σ L_j / Σ I_j`); the shipped default of 1.5 is a placeholder. If the
residuals are wide because `L` is unrelated to `I`, widen the margin or use hybrid mode. A
better per-request length predictor is rarely the fix: admission needs population accuracy,
which the calibrated ratio already gives.

A related lever, useful in every mode: cap `max_tokens` at the gateway or client for the
fraction of requests that leave it unset. Truncating the footprint tail tightens every bound
above, and M/G/1 analysis of LLM serving (arXiv:2407.05347) shows that clipping a small
fraction of the tail buys a large reduction in waiting time. The token estimator already
honors client caps.

**`hybrid` mode.** The per-endpoint saturation is the larger of the request ratio and the
token ratio, so both limits apply at once. `maxConcurrency` guards slots and residency;
`maxTokenConcurrency` guards prefill. Configure each from its section above. This is the
recommended mode for heterogeneous or decode-heavy workloads.

**`utilization-detector`.** The detector reads the true state through a dead time Δ: the
EPP's model-server metrics refresh (50ms default), plus its staleness allowance (200ms
default), plus the engine's own metric latency. During Δ the detector is blind. Two things
happen in that window: the dispatch loop keeps admitting against the stale reading, and the
existing batch keeps growing on its own. The threshold must leave room for both:

```
kvCacheUtilThreshold ≤ 1 − ( λ_dispatch·Δ·Ī/endpoints + S·r_decode·Δ ) / C − margin
```

with `λ_dispatch` the observed peak pool admission rate, `r_decode` the per-sequence decode
rate, `S` the expected concurrent sequences per endpoint, and `margin` a damping allowance
(the wizard's `--tau-margin`, default 0.05, a convention). High admission rates and long
prompts can make this derating consume most of
the threshold. When it does, no threshold protects the pool through the blind window, and
the concurrency detector is the appropriate tool. That comparison, made with your numbers,
replaces any blanket advice about which detector production should use.

Two structural notes on this detector. Pool saturation is the unweighted mean of
per-endpoint scores, so one saturated endpoint among idle ones does not move the pool gate;
protecting individual replicas is the scheduling filter's job. And threshold controllers
near saturation oscillate. An admission decision takes effect roughly one prefill later than
the state it read, and analyses of this feedback delay (Mooncake, FAST 2025; arXiv:2606.15555)
show sustained admit/overload cycles when thresholds sit at the cliff edge. The derating
margin is what damps this. Do not tune it away.

> Assumes: the Δ inputs reflect your deployment (defaults given above); admission during the
> blind window is bounded by the observed dispatch rate.
> When wrong: the threshold admits through the blind window and the engine preempts.
> Check: preemption counter against utilization p99, as below.

## Choosing a mode

A static limit pays for safety with capacity: the margin it holds back is throughput given
up. In requests mode the margin, as a fraction of capacity, is approximately `z·CV_F/√N`,
with `CV_F` the coefficient of variation of the in-flight footprint. The wizard prints this
number. When footprints are near-uniform the margin is a rounding error, and requests mode is
the simplest correct choice. As `CV_F` grows the margin eats into usable capacity.
Re-denominating the limit in tokens (with output estimation) or using hybrid mode shrinks the
wasted share to the output-estimate error. Above `CV_F ≈ 0.6` the wizard recommends the
switch.

## Validation and re-tuning

Exceeding the memory limit is not a silent event. The engine preempts and recomputes, and it
counts doing so: `vllm:num_preemptions_total`. This gives the configured limit a production
test:

- Preemption deltas stay near zero at peak and KV-utilization p99 stays under the engine's
  watermark: the limit holds. A limit that passes a representative week is calibrated,
  whatever the model said.
- A steady preemption rate: the limit or the capacity constant is too optimistic. Lower the
  limit by the gap between the point estimate and the LCB, and re-observe.
- Zero preemptions with KV p99 far below target: capacity is idle. Raise toward the point
  estimate, or revisit the mode choice if the `CV_F` warning fired.

One unit caveat. ε is a fraction of time spent at risk of overflow, and the preemption
counter counts events. Converting between them exactly requires a model of how long overflow
episodes last, which this framework does not include. Treat the counter as the pass/fail
signal and ε as the design target that positions the limit.

Preemptions can also come from causes a gateway limit does not control, such as
engine-internal behavior or co-located workloads. Before lowering a limit in response to
preemptions, check that KV utilization was actually near the watermark when they occurred.

The limit is conditioned on the workload distribution it was derived from. Re-derive when
the input or output length distributions drift from the calibration data. The trigger is a
population-stability index above 0.2, or a Kolmogorov–Smirnov rejection, on either marginal;
both are computable from the engine's token histograms (`vllm:request_prompt_tokens`,
`vllm:request_generation_tokens`). Re-derive as well after model or
hardware changes, and after engine upgrades that change KV accounting.

## Assumptions and failure modes

Everything above, collected, with the evidence from the framework's own validation. The
checks were: closed-form identities in the wizard's `--self-check`; a closed-loop simulator
(`scripts/tuning_simulator.py`) that holds a synthetic continuous-batching pool at the
derived limit across 18 workload shapes (deterministic, exponential, lognormal at three tail
weights, correlated input/output, each at three input:output ratios); and a live
concurrency sweep, run July 2026 on an 8-replica × TP2 deployment of Qwen3-32B on
H100-80GB (20,385 KV blocks per replica) — "the measured deployment" wherever the
bullets below cite live numbers. On that sweep, each replica's first preemption occurred
at 107–135 concurrent requests against a derived wall of 110. The simulator is seeded:
`python3 scripts/tuning_simulator.py` reproduces every number below in about six seconds.

1. **Stationarity.** All bounds assume the calibration window represents the future. The
   drift test above is the guard. Simulation does not cover non-stationary mixes.
2. **Independent slots.** Justified at the limit by the renewal argument, violated by
   correlated admissions. In simulation, repeating the previous admission with probability
   0.5 pushed realized overflow to 0.018 against an ε of 0.01. Traffic that arrives in
   cohorts sharing prefixes should use a smaller ε or a wider margin. On the measured
   deployment: a closed-loop load generator that restarts every connection at
   each stage boundary synchronizes request lifetimes, and footprints then peak near
   `N·(Ī+L̄)` instead of averaging `N·(Ī+L̄/2)` — a synchronized wall of
   ≈90 requests per endpoint against a stationary wall of ≈110. Limits derived from the
   stationary model produced a steady preemption trickle under that load. Wave-like
   production traffic (cron fan-outs, retry storms) sits between the two walls.
3. **Tail representation.** The trace bound held on all 18 grid points (realized overflow at
   or below ε), with conservatism from roughly zero (CV ≤ 1) up to 2.5× (heaviest tail). The
   moments bound at its matched quantile realized 0.020 against a 0.01 target on the
   heaviest tail. The gamma fill-in for missing higher moments errs the same way; prefer a
   trace.
4. **Footprint additivity and the prefix working set.** The capacity constant subtracts
   the *resident prefix working set*, not one prefix: with `G` distinct cached prefixes
   spread over `E` endpoints by affinity routing, each endpoint holds
   `min(G, 2·⌈G/E⌉)` of them (`--prefix-groups`). The 2× spill factor is measured
   (the measured deployment, 64 prefix groups over 8 endpoints: ≈1.8× G/E resident per
   replica under load; preemption onset at 107–135 requests per replica matched the
   working-set-corrected wall of 110 within 3%, where the single-prefix model predicted
   161). With the working set accounted for, remaining prefix sharing means requests
   hold less physical KV than the sum of their footprints, so the residual error is
   conservative. The calibrated `η` regression absorbs part of it, because measured
   utilization already reflects sharing.
5. **ε versus events.** See the unit caveat in Validation.
6. **Per-EPP-replica scope.** All limits and queues are per EPP replica. Running R replicas
   multiplies effective limits by R and makes fairness hold per replica. Size limits with R
   in mind.

Relationship to planned work: the llm-d-router capacity-ledger design
([llm-d-router#2061](https://github.com/llm-d/llm-d-router/pull/2061), in review) replaces
these static projections with per-request accounting. The statistics above remain the
stationary description of what that accounting tracks exactly. Tokens mode with output
estimation is the static form of its first stage, so limits derived here transfer.

## Related work

- F. Kelly, *Notes on Effective Bandwidths* (1996): the many-sources admission bound used in
  the trace path.
- CacheOPT (arXiv:2503.13773): chance-constrained KV allocation with concentration bounds.
- TIE (arXiv:2604.00499): per-prompt output lengths fit log-t distributions; the measured
  P99/P50 near 10 cited above.
- Sarathi-Serve (OSDI 2024): chunked prefill makes iteration cost token-linear; the basis
  for treating the compute and memory axes separately.
- arXiv:2407.05347: M/G/1 analysis of LLM serving; the `max_tokens` clipping result.
- Mooncake (FAST 2025) and arXiv:2606.15555: feedback delay makes reactive admission
  oscillate near saturation; the basis for the threshold-damping note.
- Google Autopilot (EuroSys 2020): limits set from usage percentiles traded against an
  eviction rate; the pattern behind the preemption-budget loop.
- NVIDIA Dynamo `profile_sla` and SCOOT (WWW 2025): profile-derived engine limits; the
  compute wall's precedents.
