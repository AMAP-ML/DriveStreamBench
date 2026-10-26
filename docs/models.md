# Model adapters

Both inference entry points accept `--model <spec>`:

| Spec | Backend |
|---|---|
| `qwen3vl:<checkpoint>` | Qwen3-VL with vLLM |
| `api_vlm:<model>` | OpenAI-compatible API |
| `dummy`, `dummy_yes`, `dummy_no`, `alert_on_turn:N` | CPU examples |

The API adapter reads `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL_NAME` from
the environment and sends the visible frame history at each step. Use the
short L4 annotations for the short-version setting.
The default request extensions are `top_k=1` and `chat_template_kwargs={"enable_thinking":false}`; override them with `LLM_EXTRA_BODY_JSON` or an explicit `APIVLM(extra_body=...)` JSON object.
Set `LLM_EXTRA_BODY_JSON='{}'` for a standard request without extensions; unsupported parameters raise an error rather than silently changing the evaluation settings.

Streaming adapters support `streamforest`, `simplestream`, `dispider`,
`videollm_online`, `mmduet2` and `m4_omnimmi`. Install the upstream software and
checkpoint under its own license. The external models use their native batch
inference entry points. Each YAML in `external_baselines/` specifies its repository and environment:

```bash
bash external_baselines/bootstrap_baseline.sh dispider --install-dir ./_baselines/dispider
python external_baselines/run.py --help
```

For Qwen3-VL, APIs and custom step adapters, `BaseModelAdapter` defines
`reset_session` and `step(frames, instruction, history, video_time)`. The DSB runner handles frame sampling and causal decision
times; custom adapters return the per-turn model output for the selected parser.
