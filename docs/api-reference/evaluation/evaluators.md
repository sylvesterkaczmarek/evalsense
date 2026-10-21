# :material-magnify: Evaluators

::: evalsense.evaluation.evaluators


## BERTScore input lengths

`BertScoreCalculator` and `get_bertscore_evaluator` require each complete text to
fit the selected model's declared input limit. The prediction and reference are
checked separately, including special tokens, before calling the scoring backend.
Overlength inputs raise `BertScoreLengthError` with the input side, token count
and limit. The error does not contain the input text, and no partial score is
returned. The pipeline records the affected evaluation as failed; it does not
report a prefix-only result as successful.

Token counting uses BERTScore's tokenizer loader and its slow/fast tokenizer
selection, with truncation disabled. It preserves BERTScore's surrounding
whitespace handling, special tokens and prefix-space behavior for slow GPT-2 and
RoBERTa tokenizers. Word and character counts are not substitutes for token counts.
Tokenizer/configuration data are loaded lazily and reused for the same model and
tokenizer mode. Changing either setting, or cleaning up the evaluator, resets
that validation context.

The limit is the smallest finite positive integer declared by the tokenizer's
`model_max_length` or the model configuration's `max_position_embeddings` and
`n_positions`. Transformers' large placeholder values are not capacity claims.
The model configuration supplies a limit when the tokenizer has only a placeholder.
For the default DeBERTa configuration, the declared limit is 512 tokens including
special tokens. Other models use their own declarations; 512 is not hardcoded.
A model with no finite declaration is rejected rather than assigned a guessed
limit. This conservative check does not establish a model's useful context quality
or guarantee sufficient memory for every permitted input.

Within-limit scores, component names and hash metadata are unchanged. Increasing
`batch_size` does not increase the input limit. There is no automatic truncation,
chunking, or chunk-score averaging: those would change the evaluation protocol.
Use an appropriate long-context model or a separately specified and validated
segmentation/aggregation method for longer documents.

Existing cached results are not repaired or revalidated. Use a fresh project or
a new evaluator name when applying this policy to previously scored data; with
versioned evaluator components, also update their version. Preserve the original
logs as historical results, rather than treating them as length-validated scores.

See the [BERTScore authors' input-length limitation](https://github.com/Tiiiger/bert_score#practical-tips)
and their [tokenisation implementation](https://github.com/Tiiiger/bert_score/blob/master/bert_score/utils.py).
