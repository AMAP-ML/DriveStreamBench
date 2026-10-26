# Data preparation

The release contains the benchmark annotations, schema, source-video references,
privacy masks and preprocessing utilities. Download the fixed-version archive
using `bash scripts/prepare_data.sh`; the downloader verifies its SHA-256 and
installs all ten JSONL files under `annotations/`.

| Split | Train | Test |
|---|---:|---:|
| L1 Core | 4,843 | 1,024 |
| L2 Core | 5,211 | 1,073 |
| L3 Core | 5,003 | 1,949 |
| L4 Core | 5,003 | 1,949 |
| L3 OOD | — | 399 |
| L4 OOD | — | 399 |

Each record contains `question_id`, `question_level`, `video_name`,
`question_type`, `split`, `sample_type`, `alert_mode`, `source_type`,
`image_samples`, `t_ref_frame`, `question`, `answer` and
`trigger_candidate_count`. `image_samples` contains ordered paths relative to
`--dataroot`; `t_ref_frame` indexes the source frame sequence at 10 FPS.

Obtain Core camera images from [nuPlan](https://www.nuscenes.org/nuplan).
Preserve each relative image path under the selected dataroot. OOD source links,
frame extraction and privacy-mosaic instructions are installed under
`external/ood/`:

OOD preprocessing uses Python 3.10 or later, FFmpeg on `PATH`, NumPy and OpenCV.
Install the Python dependencies with `python -m pip install numpy opencv-python-headless`.

```bash
python external/ood/code/extract_resize_682x384.py \
  --input-path /path/to/source_videos --output-path /path/to/frames/bilibili \
  --annotation-jsonl annotations/l4_ood.jsonl
python external/ood/code/mask_privacy.py \
  --mask-dir external/ood/file/masks --img-root /path/to/frames \
  --output /path/to/redacted_frames
```

See `external/ood/OOD_data_process.md` for the source-video and frame-layout details.

```bash
python -m drivestreambench.data.validate annotations/l4_core_test.jsonl
bash scripts/build_l4_short.sh
```

Short positives retain a window around the reference frame; short negatives
retain their first 15 seconds. `metadata.crop` records the window and remapped
reference index. Use `DSB_SHORT_SECONDS` to select another duration.

The data archive includes `FILES.sha256` and a split index. Croissant metadata
is a separate release asset. The benchmark-created annotations and utilities are
CC BY 4.0; source-media rights remain with the original providers.
