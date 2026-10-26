# Evaluation

L4 uses causal front-camera frames, 2 FPS input and one decision per second.
Predictions are matched to annotations by `question_id`:

```json
{"question_id":"sample_id","model_name":"model","turns":[{"turn_idx":0,"video_time":1.0,"reply":"NO REPLY"}]}
```

L4 scoring uses the first non-silent decision on positives and silence across
the full session on negatives. AlertScore@τ is the harmonic mean of Hit@τ and
Negative Silence Rate (NSR), computed per category. Core Macro averages S1-S5;
S6 sudden appearance is reported separately. The scorer evaluates τ = 1, 2, 3,
5 seconds and uses each annotation's `alert_mode` to construct the time window.

Reference annotations remain source-frame indices in the released 10-FPS clips.
For scoring at input rate `f` (2 FPS by default), let `s = 10 // f` and
`N = len(image_samples)`; the scoring reference is
`min(ceil(t_ref_frame / s), len(range(0, N, s)) - 1) / f` seconds.
This maps to the first sampled frame at or after the reference and clamps to the
final sampled frame when the source reference lies beyond it. Predictions keep
the model's decision times: cumulative received-frame time for the default
protocol, including a final partial group, or the documented native model clock.
An explicit scalar `t_ref` is already in scoring seconds and is used unchanged.

The `strict_literal` parser treats empty output and `NO REPLY` as silence,
`ALERT` as an alert and other text as invalid. The `action_level` parser maps
any non-empty output other than `NO REPLY` to an alert. Qwen3-VL, DriveStream-SFT
and API VLMs use `strict_literal`; StreamForest, SimpleStream, Dispider,
VideoLLM-online and MMDuet2 use `action_level`. M4's native trigger is converted
to `ALERT` / `NO REPLY` and scored with `strict_literal`.

```bash
python scripts/evaluate.py --task l4 \
  --annotations annotations/l4_core_test.jsonl \
  --predictions results/predictions/model.jsonl \
  --parser strict_literal --fps 2 --output results/scores/model.json
```

Use `--fps 1` for native 1-FPS predictions. The Python APIs `score_l4`,
`score_l4_files` and `build_outcomes` accept the same `fps` argument; L3-gated
scoring uses the outcomes from `build_outcomes(..., fps=2)` or the matching native
rate. Prepared native `scoring_annotations.jsonl` stores scalar `t_ref` values,
so those values are not quantized again.

Reactive predictions use `{"question_id":"sample_id","prediction":"Yes"}`.
Run `scripts/evaluate.py --task l123` with the corresponding annotation and
prediction files for L1-L3 QA scores. L1/L2 accuracy, L3 event F1 and paired
L3-gated L4 metrics are defined in the paper; the Python metric APIs are in
`drivestreambench/metrics/`.

```bash
bash scripts/score_predictions.sh
python -m pytest tests/
```
