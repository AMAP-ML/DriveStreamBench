# DSB integration: https://github.com/OmniMMI/M4
# Source revision: 574d2244576190cd150cd28aaaa8dbd2ea7502e6; see external_baselines/NOTICE.md.
"""M4 native DSB inference."""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from typing import Any
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark_utils.proactive_eval import ALERT_MATCH_MODES, build_gold_aligned_entry_from_alert_times, derive_model_log_path, is_alert_reply, load_gold_templates, normalize_reply_text, parse_frame_input_sample, save_json, save_jsonl
ALERT_TIME_PATTERNS = [re.compile('REQUIREMENT MEET AT\\s*([0-9]+(?:\\.[0-9]+)?)s', re.IGNORECASE), re.compile('Current Time =\\s*([0-9]+(?:\\.[0-9]+)?)s', re.IGNORECASE)]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='M4 proactive benchmark inference')
    parser.add_argument('--model_path', type=str, default=os.environ.get('M4_CHECKPOINT_ROOT', 'checkpoints/M4'))
    parser.add_argument('--test_fname', type=str, required=True)
    parser.add_argument('--gold_fname', type=str, required=True)
    parser.add_argument('--output_fname', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--frame_interval', type=float, default=0.5)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--conv_mode', type=str, default='qwen_1_5')
    parser.add_argument('--attn_implementation', type=str, default='eager')
    parser.add_argument('--max_frames', type=int, default=512)
    parser.add_argument('--alert_match_mode', type=str, default='strict', choices=ALERT_MATCH_MODES)
    return parser.parse_args()

def build_liveinfer_argv(args: argparse.Namespace) -> list[str]:
    frame_fps = max(1, int(round(1.0 / args.frame_interval)))
    return [sys.argv[0], '--model_path', args.model_path, '--conv_mode', args.conv_mode, '--frame_fps', str(frame_fps), '--max_num_frames', str(args.max_frames), '--attn_implementation', args.attn_implementation, '--output_dir', 'outputs/benchmark']

def extract_alert_time(response: str, fallback_time: float) -> float:
    for pattern in ALERT_TIME_PATTERNS:
        match = pattern.search(response or '')
        if match:
            return round(float(match.group(1)), 2)
    return round(float(fallback_time), 2)

def extract_model_reply_text(response: str | None, current_output: str | None) -> str:
    if current_output is not None and str(current_output).strip():
        return str(current_output).strip()
    if response is None:
        return ''
    text = str(response).strip()
    if 'Assistant:' in text:
        return text.split('Assistant:', 1)[-1].strip()
    return text

def main() -> None:
    args = parse_args()
    with open(args.test_fname, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gold_templates = load_gold_templates(args.gold_fname)
    end_idx = args.end_idx if args.end_idx is not None else len(data)
    data_slice = data[args.start_idx:end_idx]
    original_argv = sys.argv[:]
    sys.argv = build_liveinfer_argv(args)
    try:
        from local_demo.inference import LiveInfer
        infer = LiveInfer()
    finally:
        sys.argv = original_argv
    results: list[dict[str, Any]] = []
    raw_logs: list[dict[str, Any]] = []
    total = len(data_slice)
    for (sample_idx, sample) in enumerate(data_slice):
        (question_id, question_level, question_text, frame_paths) = parse_frame_input_sample(sample)
        t0 = time.time()
        predicted_alert_times: list[float] = []
        raw_events: list[dict[str, Any]] = []
        sample_error: str | None = None
        try:
            infer.reset()
            infer.load_frames_from_paths(frame_paths)
            infer.input_query_stream(question_text, video_time=0.0)
            for frame_i in range(infer.num_video_frames):
                current_time = round(frame_i / infer.frame_fps, 2)
                infer.input_video_stream(current_time)
                (_query, response) = infer()
                parsed_output = extract_model_reply_text(response, getattr(infer, 'current_output', None))
                is_alert = is_alert_reply(parsed_output, match_mode=args.alert_match_mode)
                raw_events.append({'frame_index': frame_i, 'time': round(float(infer.video_time), 2), 'raw_output': response, 'parsed_output': normalize_reply_text(parsed_output), 'is_alert': is_alert})
                if is_alert:
                    predicted_alert_times.append(extract_alert_time(response or parsed_output, infer.video_time))
        except Exception as exc:
            print(f'[ERROR] id={question_id} | frames={len(frame_paths)} | {exc}')
            sample_error = str(exc)
        results.append(build_gold_aligned_entry_from_alert_times(question_id=question_id, question_level=question_level, question_text=question_text, predicted_alert_times=predicted_alert_times, gold_templates=gold_templates))
        raw_logs.append({'question_id': question_id, 'question_level': question_level, 'question_text': question_text, 'frame_count': len(frame_paths), 'alert_match_mode': args.alert_match_mode, 'predicted_alert_times': predicted_alert_times, 'raw_model_outputs': raw_events, 'error': sample_error})
        elapsed = time.time() - t0
        print(f'[Sample {sample_idx + 1}/{total}] id={question_id} | frames={len(frame_paths)} | alert_times={predicted_alert_times} | {elapsed:.1f}s')
    save_json(args.output_fname, results)
    save_jsonl(derive_model_log_path(args.output_fname, 'M4', '_raw_outputs.jsonl'), raw_logs)
    print(f'Saved {len(results)} results to {args.output_fname}')
if __name__ == '__main__':
    main()
