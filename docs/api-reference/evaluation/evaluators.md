# :material-magnify: Evaluators

::: evalsense.evaluation.evaluators


## QAGS-derived QA agreement

`get_qags_evaluator`, `QagsConfig`, `QagsScoreCalculator`, and `QagsScorerFactory`
keep their import names for compatibility. Their default output label is now
**QAGS-derived**. This implements bidirectional QA agreement, not the published
QAGS protocol.

[Wang et al. (2020)](https://aclanthology.org/2020.acl-main.450/) generate
answer-conditioned questions from the evaluated summary and compare answers
from the summary and its source. The
[authors' implementation](https://github.com/W4ngatang/qags) filters questions,
including a check against the answer used to generate each question. EvalSense
generates questions from both texts through configurable prompts, and compares
answers using ternary, exact-match or LLM-judge agreement. It does not implement
that answer-conditioned filtering procedure. The published QAGS validation does
not establish the validity of this variant.

### Questions and weighting

Prompts must request one question per line. The parser removes numbered or
bulleted list prefixes followed by whitespace, normalises Unicode to NFC, and
trims surrounding whitespace. A retained line must contain letters or digits and end
with exactly one question mark. Questions with identical resulting text are scored once across both generated
lists, in first-occurrence order. Case and internal whitespace are preserved,
alongside digits, negation and punctuation; they can be significant in units,
identifiers and quoted text.

This is syntactic filtering and text deduplication, not semantic deduplication
or a factual-grounding test. Prompt design and independent validation of question
relevance remain necessary. Each retained question has equal weight; the result
is conditional on the generated question set, not a census of distinct facts.

Score metadata retains both raw question outputs, the source(s) of each retained
question, and counts of candidate lines, invalid lines, duplicates and retained
questions. An empty retained set raises `QagsValidationError` before QA calls.

### Answers and scores

Ternary answers must be a complete `yes`, `no`, `true`, `false`, `unknown`, or
`I don't know`, ignoring case and surrounding whitespace, with an optional final
period or exclamation mark. A typographic apostrophe in `I don’t know` is accepted.
Explanations, JSON, conflicting labels and other responses fail validation.
Judge comparison responses follow the same rule, but only binary labels are
accepted, before any probability weighting or fallback is attempted.

Malformed answers fail the sample; they are never converted to `unknown` or
silently dropped from an otherwise smaller denominator. Empty answers also fail
in exact and judge modes. All modes require exactly one answer from each text
per retained question. Nonempty free-text answers in exact and judge modes are
not otherwise checked for answerability or correctness by the parser.

For valid ternary answers, `Accuracy` remains the fraction of equal labels,
including explicit `unknown`/`unknown` agreement. Such agreement is not evidence
that either text supports a fact. `Coverage` and `Groundedness` retain their
existing positive-answer definitions: matching `yes` pairs divided by reference
`yes` answers and prediction `yes` answers, respectively, or zero for an empty
positive denominator. These quantities are question-dependent proxies; this
change does not resolve the separate negative-fact/polarity issue (#57).

Exact mode averages case-insensitive, whitespace-trimmed answer matches. Judge
mode averages the binary comparisons or their existing probability-weighted
values. Judge calibration and log-probability limitations are unchanged.
The default dataset aggregation remains the mean of document scores.

Every returned score records `evaluation_protocol="evalsense.qags-derived.v2"`,
the comparison mode, question-weighting policy and invalid-answer policy.
Deterministic tests check these computations and error paths; they do not claim
clinical validation or reproduce the paper's benchmark results.

### Migration

Keep existing logs as historical results. Use a fresh project or a new evaluator
name when switching to this protocol, especially when overriding the default
name. Update downstream filters for the renamed default scores. Do not pool old
and new scores without accounting for the changed question set and validation.
With versioned components, also bump the evaluator's explicit version.
Ensure custom prompts request the accepted answer format. Inspect records a
scoring failure when validation raises; inspect failures alongside completed
scores rather than treating the remaining successful samples as the whole dataset.
