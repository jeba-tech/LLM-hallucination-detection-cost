# Consistency Is Not Correctness: Cost-Normalized Hallucination Detection on TruthfulQA

## Abstract
Four black-box hallucination-detection methods were compared on cost and
accuracy across 5,500 logged detections on TruthfulQA, using two Llama models
served on a free API tier. Only single-call verbalized confidence detects
hallucination above chance (AUROC 0.621, 95% CI [0.571, 0.670]); both
sampling-based methods — self-consistency and semantic entropy — perform
indistinguishably from random guessing at every sample count while costing up
to 6.5x more, so the intended cost-accuracy Pareto frontier collapses to a
single point at the cheapest method. Ranking the same data by F1 reverses this
conclusion, which we attribute to flag-rate artifacts rather than detection
quality.

## 1. Introduction

Large language models answer confidently whether or not they know the answer.
Detecting these fabrications automatically — without a human checking every
output — is what makes deployment in search, medical, or legal settings
defensible. A family of black-box detectors has emerged for this, most of them
built on the same intuition: sample a model's answer several times, and treat
disagreement between samples as evidence of fabrication.

Those methods are not free. Self-consistency and semantic entropy need one API
call *per sample*, so a 10-sample detector costs an order of magnitude more
than a single-call one. The literature reports detection accuracy in
isolation; a practitioner choosing a method also needs to know what the
accuracy costs.

This work set out to reproduce that cost-accuracy comparison on a model pair
the prior work had not tested, expecting to report a Pareto frontier trading
dollars against detection quality. **No such frontier was found.** Across 5,500
logged detections, only the cheapest method — a single call asking the model
to state its own confidence — detects hallucination better than chance. The
sampling-based methods perform indistinguishably from random guessing at every
sample count tested, while costing up to 6.5x more.

A secondary finding proved more interesting than the intended one: ranking the
same data by F1, the metric this literature usually reports, inverts the
conclusion. The worst method appears to win, because it flags three-quarters of
all answers against a 28% base rate and so harvests recall indiscriminately.
This is demonstrated directly in §4.2.

A likely explanation for the null result — proposed here, not established by
it — is that consistency-based detection rests on an assumption this benchmark
violates. These methods treat a wrong model as an *unsure* model whose samples
scatter. TruthfulQA is built from common misconceptions, which a model has
memorised and can repeat stably, so agreement would measure self-confidence
rather than truth. The pilot appeared to show exactly this, with hallucinated
answers scoring *higher* agreement than correct ones, but that pattern did not
survive the scaled run (§4.1); what the data supports is that consistency
carries no usable signal here, not why. Testing the explanation requires
contrasting benchmarks whose errors are systematic against ones whose errors
are random, which this experiment does not do (§5).

The contribution is a replication with a negative result and a methodological
caution about the metric used to report it. It is not a claim of a novel
research gap; §2 places it against the prior work it reproduces.

## 2. Related Work
- **SelfCheckGPT** (Manakul et al., 2023) — self-consistency sampling for
  black-box hallucination detection.
- **Semantic entropy** (Kuhn et al., Nature 2024) — clustering sampled
  generations by meaning, using entropy over clusters as an uncertainty signal.
- **FActScore** (Min et al., 2023) — atomic fact decomposition + retrieval
  verification.
- **HaluEval** (Li et al., 2023) — LLM-as-judge hallucination benchmark.
- **Cost-Effective Hallucination Detection for LLMs** (arXiv:2407.21424) —
  benchmarks scoring methods for cost across QA/fact-check/summarization and
  multiple LLMs; closest prior work to this project's framing.
- **Teaming LLMs to Detect and Mitigate Hallucinations** (arXiv:2510.19507) —
  plots detection performance against API cost with an explicit
  Pareto-frontier framing, same axes as used here.
- **The First Token Knows: Single-Decode Confidence** (arXiv:2605.05166) —
  shows a cheap single-pass confidence signal can beat semantic
  self-consistency at a fraction of the cost, the same tradeoff this project
  measures for its own model pair.
- **HalluciNot** (arXiv:2504.07069) — context/common-knowledge verification
  with an efficiency framing.
- **Verbalized confidence** (Lin et al. 2022; Tian et al. 2023) — eliciting a
  model's own confidence in words rather than reading it from token
  probabilities; the substitute baseline used here (see §3).

**Positioning.** Cost-normalized comparison of hallucination detectors is an
active, populated sub-area, not a gap this project discovered. The starting
point was a reproduction on two Llama models (8B and 70B) that the papers
above did not test, asking whether their qualitative finding — cheap
single-pass signals rivalling expensive sampling — holds on a different model
pair. It does, in a stronger form than expected: the expensive methods here do
not merely lose per dollar, they fail to beat chance at any price. Two things
this project adds beyond replication are the mechanism proposed for that
failure (§4.2) and the demonstration that F1 reverses the ranking (§4.2).

## 3. Method
- **Dataset**: TruthfulQA (`truthfulqa/truthful_qa`, generation config,
  validation split, 817 items). A 50-question pilot sample (seed 42) was later
  extended to 500 questions; the sample is *nested*, so the pilot set is the
  first 50 rows of the scaled set and pilot results remain valid under the
  larger run. Each item supplies a question and a human-written reference
  answer, used for grading.
- **Models**: `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`, both served
  by Groq. The 8B model ran the full n=500; the 70B model is capped at n=50 by
  its free-tier quota (§5). Google Gemini was the intended second provider and
  was abandoned: its free tier permits 20 requests/day, about 1.5 questions.
- **Generation format**: answers are elicited in short form (a phrase or single
  sentence, no explanation). This is load-bearing rather than cosmetic —
  self-consistency compares sampled answers by string match, and free-form
  paragraph answers never match, which collapses agreement to exactly 1/N for
  every question and makes the detector fire on everything. TruthfulQA is a
  short-answer benchmark, so constraining length is also how it is meant to be
  evaluated. Answers are compared after SQuAD-style normalization (casefold,
  strip articles and punctuation, collapse whitespace).
- **Detectors**, all four run on both models:
  1. *Verbalized confidence* — one call (temperature 0) asks for an answer plus
     a self-reported 0-100 confidence; below 80 is flagged.
  2. *LLM-as-judge* — one generation call plus one verifier call (both
     temperature 0) asking whether the answer is true; a FALSE verdict flags.
  3. *Self-consistency* — N ∈ {1,3,5,10} samples at temperature 0.7; flagged
     when the majority answer's share falls below 0.5.
  4. *Semantic entropy* — the same samples, clustered by embedding similarity
     (`all-MiniLM-L6-v2`, agglomerative, cosine, distance threshold 0.3);
     flagged when entropy over cluster sizes, normalized to [0,1], exceeds 0.6.
- **Substitution from the original design**: method 1 was planned as
  token-level log-prob uncertainty, following the uncertainty-estimation
  literature. It could not be run at all. Groq rejects the `logprobs` parameter
  on every model it serves, and Gemini returns "Logprobs is not enabled for
  this model" — so no free-tier API tested exposes token probabilities.
  Verbalized confidence (Lin et al. 2022; Tian et al. 2023) was substituted
  because it preserves method 1's role in the experiment: the cheapest possible
  single-call signal, anchoring the low-cost end of the cost axis. It is a
  weaker proxy than the log-prob signal it replaces (§5).
- **Sampling efficiency**: samples are drawn once per (question, model) at
  N=10, and the N-sweep scores prefixes of that draw; self-consistency and
  semantic entropy score the same samples. Since draws are i.i.d. at fixed
  temperature, a length-n prefix is distributed identically to an independent
  n-sample draw. Each N is charged only for the calls it consumes. This cuts
  41 calls per (question, model) to 13 without changing what is measured.
- **Labeling**: ground-truth hallucination label per generation via a
  reference-grounded grader (`llama-3.1-8b-instant` comparing the generation
  against the known-correct TruthfulQA answer), cached per (question, answer).
- **Cost accounting**: per-call token counts x published per-token pricing,
  summed per (method, model, N); wall-clock latency logged per call. Both
  models ran on free tiers, so **no money was actually spent** — the $ axis
  simulates what the same token volume would cost at published pay-as-you-go
  rates, which is the quantity the research question concerns.
- **Raw scores retained**: each detector's continuous signal (confidence,
  agreement ratio, normalized entropy) is logged alongside its binary
  prediction, so decision thresholds can be swept in analysis without
  re-running against rate-limited APIs.
- **Evaluation**: F1 is reported at each detector's a-priori threshold, and
  again as the best value reachable by sweeping that threshold post hoc.
  AUROC, computed from the retained continuous scores, is the primary metric,
  since it is threshold-free. Every AUROC carries a 95% confidence interval
  from 2,000 bootstrap resamples; a configuration is treated as detecting
  anything only when that interval excludes 0.5. LLM-as-judge returns a bare
  verdict rather than a score, so it has no sweepable threshold and no AUROC.
- **Scale of the run**: 5,500 logged detections — 500 questions x 10
  configurations on the 8B model, plus 50 x 10 on the 70B model.

## 4. Results

### 4.1 Pilot (n=50, both models) — underpowered

The first complete run covered 50 questions on both models (1,000 logged
detections). Headline numbers looked orderly: self-consistency F1 rose
monotonically with N on the 8B model (0.00 → 0.35 → 0.37 → 0.42 for
N=1/3/5/10), and LLM-as-judge scored highest overall on the 70B model
(F1 = 0.50).

Those numbers do not survive scrutiny:

- **Base rate.** 30.4% of graded answers were hallucinations, so a detector
  that simply flags *everything* achieves F1 = 0.466. Only one of the 20
  configurations exceeded that, and only barely.
- **Bootstrap intervals.** Across all 14 configurations producing a continuous
  score, **zero** had a 95% bootstrap CI for AUROC excluding 0.5. Point
  estimates ranged 0.40-0.64, with interval widths around ±0.15.

The correct conclusion from the pilot is therefore negative and about
*method*, not about detectors: at n=50 this design cannot distinguish any
detector from random guessing, and any ranking read off the F1 column would be
noise.

One suggestive pattern motivated the scaled run. Mean self-consistency
agreement was *higher* for hallucinated answers than for correct ones on both
models (8B: 0.43 vs 0.38; 70B: 0.71 vs 0.64) — i.e. the models appeared more
self-consistent when wrong, which would imply consistency-based detection is
not merely weak but *anti-correlated* on misconception-style questions.

**This did not replicate.** At n=500, self-consistency AUROC is 0.464-0.485
with intervals spanning 0.5 (e.g. N=10: 0.464 [0.407, 0.517]). The pilot's
apparent inversion was noise, exactly as its confidence intervals warned. It
is recorded here because the discipline that caught it — refusing to read a
point estimate whose interval covers chance — is the methodological point of
this section.

### 4.2 Scaled run (n=500, 8B model)

Scaling to 500 questions (5,000 detections on the 8B model) narrowed the AUROC
intervals from roughly ±0.15 to ±0.05. The 70B results remain at n=50; the
`n_questions` column in `results/summary.csv` records the asymmetry.

Base rate: 28.0% of graded answers were hallucinations, so flagging everything
yields F1 = 0.438.

| Method | Calls/question | Cost (500 q) | AUROC [95% CI] | Best swept F1 |
|---|---|---|---|---|
| **Verbalized confidence** | **1** | **$0.0037** | **0.621 [0.571, 0.670]** | **0.548** |
| LLM-as-judge | 2 | $0.0052 | — (binary verdict) | 0.336 (fixed) |
| Semantic entropy (N=3) | 3 | $0.0071 | 0.532 [0.478, 0.584] | 0.457 |
| Semantic entropy (N=10) | 10 | $0.0241 | 0.520 [0.465, 0.577] | 0.457 |
| Self-consistency (N=3) | 3 | $0.0071 | 0.485 [0.438, 0.530] | 0.450 |
| Self-consistency (N=10) | 10 | $0.0241 | 0.464 [0.407, 0.517] | 0.445 |

**Result: one of fourteen configurations exceeds chance.** Verbalized
confidence — a single call asking the model to state a 0-100 confidence — is
the only method whose AUROC interval excludes 0.5. Both sampling-based
methods sit at chance at every N, with intervals now tight enough to exclude
any useful effect rather than merely failing to detect one. Increasing N from
3 to 10 moves AUROC by less than 0.02 in either direction while costing 3.4x
more.

**The cost-accuracy frontier collapses to a single point.** The intended
deliverable of this project was a Pareto curve trading cost against accuracy.
No such trade-off exists in this data: the cheapest method dominates outright,
and the expensive methods are not merely worse per dollar but indistinguishable
from random guessing at any price. Figure `results/pareto_plot.png` shows the
frontier degenerating in this way.

**F1 alone would have produced the opposite conclusion.** Ranked by F1 at the
shipped threshold, self-consistency (0.39-0.40) appears to beat verbalized
confidence (0.099) fourfold. That ordering is an artifact of flag rate:
self-consistency flags 70-75% of all answers against a 28% base rate, so it
harvests recall indiscriminately, while verbalized confidence flags 5.6%
because the 8B model almost never rates itself below the a-priori threshold of
80. Once thresholds are swept, verbalized confidence reaches F1 = 0.548 and
self-consistency peaks at 0.450 — below the 0.438 flag-everything baseline.
The threshold-free AUROC, with intervals, is the only view of this data that
supports a conclusion.

### 4.3 Cross-model comparison (limited)

On the 70B model at n=50, verbalized confidence shows a similar point estimate
(AUROC 0.627) but with an interval spanning chance [0.444, 0.814], consistent
with the 8B result while not independently establishing it. No 70B
configuration reaches significance at this sample size, so the scale contrast
(8B vs 70B) is not resolved by this experiment and no claim is made about it.

## 5. Limitations
- **No token-level uncertainty baseline.** The strongest cheap signal in the
  literature (log-prob / semantic entropy over token probabilities) could not
  be measured at all: no free-tier API exposes log-probs. Verbalized
  confidence is a weaker proxy — models are known to be poorly calibrated when
  self-reporting — so the cheap end of the cost axis is likely represented
  pessimistically relative to what a log-prob method would achieve.
- **Sample size is quota-bound, and asymmetric across models.** Free-tier
  daily request caps, not statistical preference, set the sample sizes: the
  8B model (14,400 req/day) supports n=500, the 70B model (1,000 req/day)
  only n=50. Cross-model comparisons are therefore between a well-powered and
  an underpowered estimate and should not be read as a like-for-like contrast.
  The full 817-question benchmark was not run.
- **Thresholds are untuned.** Each detector's flag threshold was set a priori,
  not fitted, and verbalized-confidence scores cluster heavily at 80-100, so
  F1 is sensitive to that choice. Raw scores are logged specifically so this
  can be checked by sweeping thresholds post hoc; until that is done, reported
  F1 reflects one arbitrary operating point per method.
- **Ground-truth labels are model-generated.** Grading uses an LLM judge
  against the reference answer rather than human annotation, so label noise
  propagates directly into every F1 number. The grader is also one of the two
  models under test, which risks favouring its own outputs.
- **Cost axis is simulated and currently unverified.** No money was spent;
  dollar figures are token counts times published pay-as-you-go rates. The
  pricing constants in `config.py` are flagged `UNVERIFIED` and must be
  replaced with current published rates before any $ figure is reported.
- **Only 2 models, both small, both Llama.** Findings may not generalize to
  larger or paid models, and because both come from one family they do not
  support any claim about provider- or architecture-level differences. Gemini
  was the intended second provider; its free tier permits 20 requests/day,
  roughly 1.5 questions, so cross-provider comparison was abandoned.
- **Sampling temperature is unswept.** All sampling used temperature 0.7. Since
  every consistency-based method measures dispersion across samples, this
  single choice directly sets the scale of the signal those methods depend on.
  A reader may reasonably object that the sampling methods were handicapped by
  this setting rather than genuinely inert; the experiment cannot rule that out.
- **One benchmark, and the proposed mechanism is untested.** TruthfulQA is
  adversarially built from misconceptions, so it may represent a worst case for
  consistency-based methods rather than typical factual QA. The explanation
  offered in §1 — that these methods fail when model errors are systematic
  rather than random — is consistent with the null result but not demonstrated
  by it. Testing it requires running the same pipeline on a non-adversarial
  factual QA set and showing the methods recover there; until that is done, the
  finding should be read as scoped to misconception-style questions.
- **Not a novel research gap** — see §2; this is a reproduction/extension of
  recently published cost-aware hallucination-detection comparisons on a
  different model pair.

## 6. Conclusion

Across 5,500 logged detections on TruthfulQA, only one of fourteen scored
configurations detects hallucination better than chance: single-call
verbalized confidence (AUROC 0.621 [0.571, 0.670]). Sampling-based detection —
self-consistency and semantic entropy, the methods this literature is largely
built on — performs at chance at every sample count tested, while costing up
to 6.5x more.

The practical recommendation is therefore narrow but firm: on
misconception-style factual QA with a small model, do not pay for repeated
sampling. It buys nothing measurable here. Ask the model for its confidence in
the same call, and tune the threshold on held-out data, because the default
operating point is badly placed (F1 0.099 untuned vs 0.548 swept).

Three caveats bound this claim. Verbalized confidence is *above chance*, not
*good* — AUROC 0.62 and best-swept F1 0.548 against a 0.438 flag-everything
baseline is a weak detector, not a deployable one. TruthfulQA is adversarially
constructed around common misconceptions, which is close to a worst case for
consistency-based methods: a model that has stably memorised a falsehood will
repeat it, so agreement carries no truth signal by construction. And the
strongest cheap baseline in the literature, token-level log-prob uncertainty,
could not be measured at all because no free-tier API exposes log-probs, so
the cheap end of the cost axis is represented by a proxy that is probably
weaker than the state of the art.

The result that generalises beyond these models is methodological. The pilot
(n=50) produced an orderly, entirely spurious ranking — including an apparent
inversion of self-consistency that vanished at n=500 — and the F1 column at
full scale still ranks the methods in the reverse of the correct order.
Bootstrap intervals on a threshold-free metric were what separated signal from
artifact in both cases. Reported without them, this experiment would have
supported a confident and wrong conclusion.
