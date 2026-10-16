# :material-magnify: Result Analysers

::: evalsense.workflow.analysers

## Paired metric correlations

`MetricCorrelationAnalyser` joins scores using the full generation record,
source Inspect `eval_id`, sample ID (including its type), and epoch. Copies of a
generation log evaluated by different scorers can be compared; independent
generation attempts cannot be paired just because their sample IDs match.
Duplicate metric/observation keys raise `ValueError` rather than being overwritten.

The analyser preserves evaluator, scorer and inner-metric identity internally.
Inner metric names remain unchanged in tables unless they are ambiguous; only
then are evaluator/scorer names added. `metric_labels` changes display labels,
not the observations being joined. Ambiguous aliases are rejected. Filters still
receive the original scalar scorer name or inner metric name before aliasing.

The returned `paired_counts` matrix reports the number of finite matched pairs
used for each coefficient. `excluded_counts` reports matched pairs dropped because
at least one value is non-finite. Unmatched observations are in neither count.
Fewer than two finite pairs, or a constant vector, yields NaN. Pairwise exclusion
makes the sample population explicit; it does not establish that missingness is
unbiased. These are descriptive correlations, not confidence estimates.

All result matrices use the same row and column order and requested output format.
As before, the first column, `Metric`, contains row labels, including in NumPy
output. An empty selection returns empty tables and no figure.
