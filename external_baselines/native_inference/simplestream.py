# DSB integration: https://github.com/EvolvingLMMs-Lab/SimpleStream
# Source revision: be227ee8ce90a258b286a34161688c2a59eeeadb; see external_baselines/NOTICE.md.
"""SimpleStream native DSB inference."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any
import torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from lib.recent_window_eval import RecentWindowQAModel as _BaseQAModel
from benchmark_utils.proactive_eval import ALERT_MATCH_MODES, build_gold_aligned_entry_from_alert_times, derive_model_log_path, is_alert_reply, load_gold_templates, parse_frame_input_sample, save_json, save_jsonl
DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant. Your task is to answer questions based on continuously incoming video frames.\n\nYou are monitoring a driving scene over time. At each turn, decide whether the monitored event has happened now based on current frames and history.\nOutput policy:\n1) If the event is happening now, output exactly: ALERT\n2) If the event is not happening now, output exactly: NO REPLY\n3) Any other output is invalid.\nDo not use XML tags or any extra text.'

class ProactiveQAModel(_BaseQAModel):

    def __init__(self, system_prompt: str=DEFAULT_SYSTEM_PROMPT, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.system_prompt = system_prompt

    @torch.inference_mode()
    def generate_from_frames_with_system(self, frames: list[Image.Image], question: str) -> str:
        visual_device = self._get_visual_device()
        text_device = self._get_text_input_device()
        content: list[dict[str, Any]] = [{'type': 'image', 'image': f} for f in frames]
        content.append({'type': 'text', 'text': question})
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]})
        messages.append({'role': 'user', 'content': content})
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(text_device)
        attention_mask = inputs['attention_mask'].to(text_device)
        pixel_values = inputs['pixel_values'].to(visual_device, dtype=self._get_visual_dtype())
        image_grid_thw = inputs['image_grid_thw'].to(visual_device)
        self._last_num_vision_tokens = int((input_ids == self.image_token_id).sum().item())
        self._last_num_vision_frames = int(image_grid_thw.shape[0])
        return self._generate_from_model_inputs(prompt_length=int(input_ids.shape[1]), input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

def parse_frame_input(example: dict[str, Any]) -> tuple[str, str, str, str, list[str]]:
    (question_id, question_level, question_text, frame_paths) = parse_frame_input_sample(example)
    conversation = example['conversation']
    system_prompt = DEFAULT_SYSTEM_PROMPT
    for turn in conversation:
        if turn['role'] == 'system':
            system_prompt = turn['content']
    return (question_id, question_level, system_prompt, question_text, frame_paths)

def main() -> None:
    parser = argparse.ArgumentParser(description='Proactive Video QA inference with SimpleStream recent-window')
    parser.add_argument('--model_path', required=True, help='HF model path, e.g. Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--test_fname', required=True, help='Path to frame_input_format.json')
    parser.add_argument('--gold_fname', required=True, help='Path to corresponding gold.json')
    parser.add_argument('--output_fname', required=True, help='Path to output gold-aligned json')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id')
    parser.add_argument('--recent_frames_only', type=int, default=4, help='Number of replay history frames')
    parser.add_argument('--input_stride', type=int, default=2, help='Number of new frames per inference step')
    parser.add_argument('--frame_interval', type=float, default=0.5, help='Seconds between frames')
    parser.add_argument('--max_qa_tokens', type=int, default=256, help='Max new tokens for generation')
    parser.add_argument('--start_idx', type=int, default=0, help='Start sample index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=None, help='End sample index (exclusive)')
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2')
    parser.add_argument('--alert_match_mode', type=str, default='strict', choices=ALERT_MATCH_MODES)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_fname) or '.', exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f'Proactive Video QA — SimpleStream Recent Window')
    print(f"{'=' * 70}")
    print(f'Model       : {args.model_path}')
    print(f'GPU         : cuda:{args.gpu_id}')
    print(f'Replay hist : {args.recent_frames_only} frames')
    print(f'Input stride: {args.input_stride} frames')
    print(f'Frame interval: {args.frame_interval}s')
    print(f'Gold        : {args.gold_fname}')
    print(f'Input       : {args.test_fname}')
    print(f'Output      : {args.output_fname}')
    print(f"{'=' * 70}\n")
    with open(args.test_fname) as f:
        data_list = json.load(f)
    gold_templates = load_gold_templates(args.gold_fname)
    end_idx = args.end_idx if args.end_idx is not None else len(data_list)
    data_list = data_list[args.start_idx:end_idx]
    print(f'Loaded {len(data_list)} samples (range [{args.start_idx}, {end_idx}))')
    device = f'cuda:{args.gpu_id}'
    qa: ProactiveQAModel | None = None
    completed = 0
    total = len(data_list)
    results: list[dict[str, Any]] = []
    raw_logs: list[dict[str, Any]] = []
    for (sample_idx, example) in enumerate(data_list):
        (question_id, question_level, system_prompt, question_text, frame_paths) = parse_frame_input(example)
        if qa is None:
            qa = ProactiveQAModel(system_prompt=system_prompt, model_name=args.model_path, device=device, max_new_tokens=args.max_qa_tokens, attn_implementation=args.attn_implementation)
        else:
            qa.system_prompt = system_prompt
        t0 = time.time()
        n_frames = len(frame_paths)
        history_frames = args.recent_frames_only
        step = max(1, args.input_stride)
        predicted_alert_times: list[float] = []
        raw_events: list[dict[str, Any]] = []
        sample_error: str | None = None
        print(f'\n[Sample {sample_idx}/{total}] id={question_id}, frames={n_frames}')
        for start_idx in range(0, n_frames, step):
            t_frame = time.time()
            end_idx = min(start_idx + step, n_frames)
            replay_start = max(0, start_idx - history_frames)
            window_paths = frame_paths[replay_start:end_idx]
            frames: list[Image.Image] = []
            for p in window_paths:
                try:
                    frames.append(Image.open(p).convert('RGB'))
                except Exception as e:
                    raise RuntimeError(f'Cannot decode image {p}: {e}') from e
            if not frames:
                continue
            try:
                reply = qa.generate_from_frames_with_system(frames, question_text)
            except Exception as e:
                print(f'  [ERROR] frames {start_idx}:{end_idx}: {e}')
                reply = 'NO REPLY'
                sample_error = str(e)
            reply_clean = reply.strip()
            for img in frames:
                img.close()
            current_time = round(end_idx * args.frame_interval, 2)
            t_elapsed_frame = time.time() - t_frame
            tag = 'REPLY' if is_alert_reply(reply_clean, match_mode=args.alert_match_mode) else 'SILENT'
            reply_short = reply_clean[:80] + ('...' if len(reply_clean) > 80 else '')
            print(f'  Frames {start_idx:>4d}:{end_idx:<4d} | t={current_time:>7.1f}s | hist={start_idx - replay_start:>2d} | total={len(window_paths):>2d} | {tag:6s} | {t_elapsed_frame:.1f}s | {reply_short}')
            raw_events.append({'window_start': replay_start, 'window_end': end_idx, 'new_frame_start': start_idx, 'new_frame_end': end_idx, 'window_size': len(window_paths), 'time': current_time, 'raw_output': reply_clean, 'is_alert': is_alert_reply(reply_clean, match_mode=args.alert_match_mode)})
            if is_alert_reply(reply_clean, match_mode=args.alert_match_mode):
                predicted_alert_times.append(current_time)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result = build_gold_aligned_entry_from_alert_times(question_id=question_id, question_level=question_level, question_text=question_text, predicted_alert_times=predicted_alert_times, gold_templates=gold_templates)
        results.append(result)
        raw_logs.append({'question_id': question_id, 'question_level': question_level, 'system_prompt': system_prompt, 'question_text': question_text, 'frame_count': n_frames, 'alert_match_mode': args.alert_match_mode, 'predicted_alert_times': predicted_alert_times, 'raw_model_outputs': raw_events, 'error': sample_error})
        completed += 1
        t_total = time.time() - t0
        print(f'  [Done] {t_total:.1f}s total | alert_times={predicted_alert_times} | Progress: {completed}/{total}')
    save_json(args.output_fname, results)
    save_jsonl(derive_model_log_path(args.output_fname, 'SimpleStream', '_raw_outputs.jsonl'), raw_logs)
    print(f"\n{'=' * 70}")
    print(f'Inference complete. {completed} results saved to {args.output_fname}')
    print(f"{'=' * 70}")
if __name__ == '__main__':
    main()
