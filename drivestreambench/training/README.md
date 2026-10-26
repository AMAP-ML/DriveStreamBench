# Reference SFT recipe

Install the training dependencies with `python -m pip install -e '.[train]'`.
Convert training annotations to conversations and launch the SFT backend:

```bash
python -m drivestreambench.training.build_sft_data \
  --annotations annotations/l4_core_train.jsonl \
  --output data_sft/l4_train.jsonl --fps 2 --frames-per-turn 2 \
  --dataroot /path/to/frames

torchrun --nproc_per_node=16 -m drivestreambench.training.train_sft \
  --train-jsonl data_sft/l4_train.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output-dir outputs/drivestream_sft --backend ms-swift \
  --batch-size 1 --grad-accum 4 --epochs 3 --lr 1e-5 \
  --adam-beta1 0.9 --adam-beta2 0.95 --weight-decay 0.1 \
  --max-length 40960 --image-max-tokens 256 \
  --deepspeed configs/training/zero3.json --frozen-vit --frozen-aligner
```

Use a held-out portion of the training data for validation when supplying
`--val-jsonl`. For the multitask recipe, convert the L1-L3 training annotations
with `drivestreambench.training.build_sft_data_l13` and combine them with the
L4 conversations. The converter marks selected `NO REPLY` messages with
`loss: false`, retaining their context while excluding their response loss.

The reference configuration uses 16 H20 GPUs, per-GPU batch size 1 and four
gradient accumulation steps, giving an effective batch size of 64. Multi-node
launches use the standard `torchrun` node and rendezvous arguments with total
world size 16. The image-token budget is passed as `IMAGE_MAX_TOKEN_NUM`.

`--backend dry-run` prints the training command. The trained checkpoint can be
loaded through `qwen3vl:<checkpoint>` for inference. Reference weights are outside
this code and annotation release.
