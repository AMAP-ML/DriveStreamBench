# DriveStreamBench

Official implementation of **DriveStreamBench: Evaluating User-Conditioned Watch-and-Notify in Streaming Driving Video** (NeurIPS 2026).

DriveStreamBench evaluates user-conditioned watch-and-notify in streaming driving
video. Given a watch instruction, a model observes causal video frames and emits
`ALERT` when the specified event occurs, or `NO REPLY` otherwise.

[Data](https://github.com/AMAP-ML/DriveStreamBench/releases/tag/v1.0.0) · [Croissant](https://github.com/AMAP-ML/DriveStreamBench/releases/download/v1.0.0/croissant.json)

## Install

```bash
python -m pip install -e '.[test]'
bash scripts/run_smoke_test.sh
```

Python 3.9 or later is required. Scoring and the example run work on CPU.
Install `.[vlm]`, `.[api]` or `.[train]` for the corresponding model backend.

## Data

```bash
bash scripts/prepare_data.sh
# A downloaded ZIP can also be installed locally:
python scripts/download_data.py --archive /path/to/DriveStreamBench-data-v1.0.0.zip
```

The annotation release contains 26,853 records across L1-L4. Source media are
obtained separately under the providers' terms. See [data preparation](docs/data.md).

### OOD sources and preprocessing

After data installation, `external/ood/file/ood_video_links.txt` lists the OOD
source videos, and `external/ood/OOD_data_process.md` provides the preprocessing
workflow.

## Evaluate

```bash
python -m drivestreambench.inference.run_l4 \
  --annotations annotations/l4_core_test.jsonl \
  --dataroot /path/to/frames \
  --model qwen3vl:Qwen/Qwen3-VL-8B-Instruct \
  --output results/predictions/model.jsonl \
  --manifest results/model.manifest.json

python scripts/evaluate.py \
  --annotations annotations/l4_core_test.jsonl \
  --predictions results/predictions/model.jsonl \
  --output results/scores/model.json --fps 2
```

The default L4 protocol uses 2 FPS input and one decision per second.
Scoring maps each 10-FPS reference frame onto the corresponding sampled-frame
grid; pass `--fps 1` when scoring a native 1-FPS run.
Core Macro averages AlertScore@2s across S1-S5; S6 is evaluated separately.
See [evaluation](docs/evaluation.md) for output formats, parser selection and
L1-L3 scoring, and [models](docs/models.md) for adapter usage.

For the short setting, run `bash scripts/build_l4_short.sh`; it writes
`annotations/l4_short_test.jsonl` with a maximum duration of 15 seconds per sample.

The [reference SFT recipe](drivestreambench/training/README.md) provides data
conversion and training entry points. Model weights are outside this release.

## Citation

```bibtex
@inproceedings{drivestreambench2026,
  title = {DriveStreamBench: Evaluating User-Conditioned Watch-and-Notify in Streaming Driving Video},
  author = {Yi Wang and Xin Zhao and Tian Meng and Rui Dai and Chi Zhang and Cui Tang and Bingzheng Liu and Fucheng Wang and Haopeng Zhang and Ruidong Ding and Yang Li and Kaikui Liu and Xiangxiang Chu},
  booktitle = {NeurIPS},
  year = {2026}
}
```

## License

Code: [Apache 2.0](LICENSE). Benchmark annotations: CC BY 4.0. Source media and
third-party models retain their respective licenses. Dataset metadata and
source-media instructions accompany the versioned data release.
