# External streaming models

These integrations run the six external baselines through their native batch
inference programs. Each model uses a separate environment and its own checkpoint.
The YAML files specify the upstream source revision, source files and inference
settings. They use JSON syntax, which is also valid YAML.

| Model | Native FPS in the paper | Output adapter | Source |
| --- | ---: | --- | --- |
| Dispider | 1 | Action level | [Dispider](https://github.com/Mark12Ding/Dispider) |
| StreamForest | 1 | Action level | [StreamForest](https://github.com/MCG-NJU/StreamForest) |
| SimpleStream | 1 | Action level | [SimpleStream](https://github.com/EvolvingLMMs-Lab/SimpleStream) |
| VideoLLM-online | 2 | Action level | [VideoLLM-online](https://github.com/showlab/videollm-online) |
| MMDuet2 | 1 | Action level | [MMDuet2](https://github.com/yellow-binary-tree/MMDuet2) |
| M4 / OmniMMI | 1 | Native intersuit trigger | [M4](https://github.com/OmniMMI/M4) |

From the repository checkout, obtain the pinned source:

```bash
bash external_baselines/bootstrap_baseline.sh dispider --install-dir ./_baselines/dispider
```

This command downloads the required source directories, applies the supplied
inference compatibility patch and verifies the revision. Python dependencies and
weights follow the installation instructions printed by the command. Run inference
with the Python interpreter from the corresponding model environment.

Prepare native inputs from the public annotations and frame directory:

```bash
python external_baselines/prepare.py \
  --annotations /path/to/annotations/l4_core_test.jsonl \
  --dataroot /path/to/frames --fps 1 --output-dir outputs/native_fps1
```

This creates `input.json`, `gold.json` and `scoring_annotations.jsonl`. The sampled
frames and reference times share the selected FPS grid. Question IDs and the
annotation's alert mode are preserved. Use `--fps 2` and a separate output directory
for VideoLLM-online.

```bash
python external_baselines/run.py dispider \
  --install-dir ./_baselines/dispider \
  --checkpoint /path/to/Dispider \
  --input outputs/native_fps1/input.json \
  --gold outputs/native_fps1/gold.json \
  --output outputs/dispider.jsonl --gpus 0

python scripts/evaluate.py --task l4 \
  --annotations outputs/native_fps1/scoring_annotations.jsonl \
  --predictions outputs/dispider.jsonl --parser action_level --fps 1 \
  --output outputs/dispider.score.json
```

Use the prepared native frame-input JSON and its matching gold JSON at the selected
FPS. Image paths must be absolute. Each input sample has `question_id`,
`question_level` and `conversation`; user content contains ordered
`{"type": "image", "image": "/absolute/path.jpg"}` entries and the watch instruction
as `{"type": "text", "text": "..."}`. The gold file is an array with matching
`question_id` values and timestamped `answer` entries. `--fps` defaults to the
model's value in the table. `--dry-run` validates source, inputs and paths and prints
the native command without loading a model.

Dispider uses the legacy [Mar2Ding/Dispider checkpoint revision
8bcbb3a1cde4465f82c4c447a34822b7e3806768](https://huggingface.co/Mar2Ding/Dispider/tree/8bcbb3a1cde4465f82c4c447a34822b7e3806768).
Obtain this snapshot as a local directory, including its `stream_compressor`
subdirectory, and pass that directory as `--checkpoint`. `--compressor` can specify
the compressor separately. Its default vision checkpoint is
`openai/clip-vit-large-patch14`; `--vision` accepts a local copy. Dispider retains
its native system template and receives the watch instruction through its native
prompt interface.

StreamForest uses the Qwen2 model loader. Preserve `qwen` in the checkpoint
directory name, for example `StreamForest-Qwen2-7B`. Its vision checkpoint defaults
to `google/siglip-so400m-patch14-384`; `--vision` overrides the checkpoint location.

SimpleStream uses `Qwen/Qwen3-VL-8B-Instruct` with the Qwen3 environment described
upstream. Its recent-window implementation is obtained from the upstream source.
MMDuet2 uses its custom `proactive_eval/qwen2_5_vl` implementation and the
dependencies in the upstream `proactive_eval/requirements.txt`.

VideoLLM-online requires its PEFT checkpoint, `--llm
/path/to/Meta-Llama-3-8B-Instruct` and `--vision /path/to/siglip-large-patch16-384`.
M4 uses its Qwen2 checkpoint, with `qwen` in the directory name, for example
`M4-LongVA-7B-Qwen2`. Its vision checkpoint defaults to
`openai/clip-vit-large-patch14-336` and can be overridden with `--vision`. M4 retains
its native system and conversation template; the watch instruction enters through
the native query interface. Score M4's converted predictions with `--parser
strict_literal`.

The output is canonical prediction JSONL. A sibling `.manifest.json` records the
source revision, FPS, checkpoints, parser and command. The `.native/` directory
retains native predictions and raw responses. Conversion requires complete sample
coverage and rejects recorded inference errors. It preserves all free-form
responses for action-level scoring, including `NO` and `No.`. M4 maps only its
native `REQUIREMENT MEET AT <t>s` trigger to `ALERT`, retaining the extracted target
time and the native observation time.

The native clock is preserved: Dispider, StreamForest and VideoLLM-online report
frame `k` at `(k + 1) / FPS`; MMDuet2 reports its cumulative received-frame time.
SimpleStream uses a four-frame replay history, receives two new frames per decision
and reports the end of that window. Dispider uses an eight-frame sliding window.
M4 observes frame `k` at `k / FPS` and reports its own trigger target time. These
native schedules are recorded separately from the common Mode-α protocol.

Source licenses and checkpoint terms remain with the respective projects. See
[NOTICE.md](NOTICE.md) and [model usage](../docs/models.md).
