# :material-magnify: Evaluation

::: evalsense.evaluation

## Cache identity

Display names are not sufficient cache keys. Generation records now also include
fingerprints of the solver configuration, preprocessing version, and sample field
mapping. Evaluation records include the evaluator parameters and judge model
configuration. Changing an evaluator does not invalidate reusable generations.

Inspect-decorated built-in solvers and scorers supply construction parameters.
The built-in EvalSense evaluator factories declare their metric options explicitly.
For custom code, supply a non-empty `version` and change it when the implementation,
closure state, external files, or other undeclared inputs change. EvalSense does
not attempt to infer arbitrary Python behavior from source code or object reprs.

```python
steps = GenerationSteps("custom generation", custom_solver, version="2")
preprocessor = TaskPreprocessor("normalise", preprocess, version="3")
task = TaskConfig(
    dataset_manager=manager,
    generation_steps=steps,
    task_preprocessor=preprocessor,
    field_spec=record_to_sample,
    field_spec_version="1",
)
evaluator = Evaluator("custom score", custom_scorer, version="4")
```

A custom scorer can additionally declare `cache_parameters` for its configurable
settings. QAGS prompt implementations are custom code, so pass `version` to
`get_qags_evaluator`. Custom metric functions likewise require the containing
evaluator's version. Missing versions for opaque components raise
`CacheIdentityError` instead of silently reusing a name-only identity.

Versions remain the caller's responsibility for external dependencies, including
prompt files and mutable dataset/model revisions. Changes to runtime options
passed directly through `eval_kwargs`, `eval_retry_kwargs`, or `score_kwargs` are
not inferred from a component's construction parameters; update the appropriate
component version when using those options to change experimental conditions.
This change does not provide full environment or arbitrary-code fingerprinting.

Preprocessed JSON files are keyed by the dataset record, Hugging Face content
fingerprint, and preprocessing identity. Mapping from JSON records to Inspect
samples happens afterwards; changes to that mapping invalidate generation, not
the intermediate JSON. Legacy project records and logs remain readable, but old
name-only cache entries are not reused by newly configured experiments. Keep a
backup before using a project with older EvalSense releases, which do not know
about these identity fields.
