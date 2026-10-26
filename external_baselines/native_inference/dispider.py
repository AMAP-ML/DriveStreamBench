# DSB integration: https://github.com/Mark12Ding/Dispider
# Source revision: 8b03eda448993f491b3237f8d511ace654c0af23; see external_baselines/NOTICE.md.
"""Dispider native DSB inference."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any
import numpy as np
import torch
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
DEFAULT_ANS_TOKEN = None
DEFAULT_IMAGE_TOKEN = None
DEFAULT_IM_END_TOKEN = None
DEFAULT_IM_START_TOKEN = None
DEFAULT_TODO_TOKEN = None
IMAGE_TOKEN_INDEX = None
conv_templates = None
get_model_name_from_path = None
tokenizer_image_token = None
load_pretrained_model = None
disable_torch_init = None
_IMPORT_ERROR = None
try:
    from dispider.constants import DEFAULT_ANS_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_TODO_TOKEN, IMAGE_TOKEN_INDEX
    from dispider.conversation import conv_templates
    from dispider.mm_utils import get_model_name_from_path, tokenizer_image_token
    from dispider.model.builder import load_pretrained_model
    from dispider.utils import disable_torch_init
except Exception as exc:
    _IMPORT_ERROR = exc
from benchmark_utils.proactive_eval import ALERT_MATCH_MODES, build_gold_aligned_entry_from_alert_times, derive_model_log_path, is_alert_reply, load_gold_templates, save_json, save_jsonl

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=None):
        super().__init__()
        self.stops = stops or []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all(stop == input_ids[0][-len(stop):]).item():
                return True
        return False
DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant. Your task is to answer questions based on continuously incoming video frames.\n\nYou are monitoring a driving scene over time. At each turn, decide whether the monitored event has happened now based on current frames and history.\nOutput policy:\n1) If the event is happening now, output exactly: ALERT\n2) If the event is not happening now, output exactly: NO REPLY\n3) Any other output is invalid.\nDo not use XML tags or any extra text.'
NUM_FRAMES_PER_CLIP = 16

def parse_frame_input(example: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    question_id = example['question_id']
    question_level = example.get('question_level', 'L4')
    conversation = example['conversation']
    question_text = ''
    frame_paths: list[str] = []
    for turn in conversation:
        if turn['role'] == 'user':
            content = turn['content']
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'image':
                            frame_paths.append(item['image'])
                        elif item.get('type') == 'text':
                            question_text = item['text']
            elif isinstance(content, str) and (not question_text):
                question_text = content
    return (question_id, question_level, question_text, frame_paths)

def load_frames_as_clips(frame_paths: list[str], image_processor, num_frm: int=NUM_FRAMES_PER_CLIP, max_clip: int=4) -> tuple[list[Image.Image], np.ndarray, int]:
    total_frames = len(frame_paths)
    if total_frames == 0:
        raise ValueError('No frame paths provided')
    num_clip = max(1, min(max_clip, (total_frames + num_frm - 1) // num_frm))
    total_num_frm = num_frm * num_clip
    seg_size = float(total_frames - 1) / total_num_frm if total_num_frm > 1 else 0
    frame_idx = []
    for i in range(total_num_frm):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        idx = min((start + end) // 2, total_frames - 1)
        frame_idx.append(idx)
    clip_imgs: list[Image.Image] = []
    for idx in frame_idx:
        img = Image.open(frame_paths[idx]).convert('RGB')
        (w, h) = img.size
        if w != h:
            s = min(w, h)
            img = img.resize((s, s))
        clip_imgs.append(img)
    frm_per_clip = len(frame_idx) // num_clip
    starts = np.array([frame_idx[i * frm_per_clip] for i in range(num_clip)], dtype=float)
    ends = np.array([frame_idx[i * frm_per_clip + frm_per_clip - 1] for i in range(num_clip)], dtype=float)
    time_idx = np.concatenate([starts, ends])
    return (clip_imgs, time_idx, num_clip)

def preprocess_time(time_array, num_clip, tokenizer, frame_interval: float):
    time_reshaped = time_array.reshape(2, num_clip)
    seq = []
    for i in range(num_clip):
        start_sec = int(np.round(time_reshaped[0, i] * frame_interval))
        end_sec = int(np.round(time_reshaped[1, i] * frame_interval))
        sentence = f'This contains a clip sampled in {start_sec} to {end_sec} seconds' + DEFAULT_IMAGE_TOKEN
        sentence = tokenizer_image_token(sentence, tokenizer, return_tensors='pt')
        seq.append(sentence)
    return seq

def preprocess_question(questions, tokenizer):
    seq = []
    for q in questions:
        sentence = tokenizer_image_token(q + DEFAULT_TODO_TOKEN, tokenizer, return_tensors='pt')
        seq.append(sentence)
    return seq

def process_frames_data(frame_paths: list[str], question: str, model_config, tokenizer, processor, processor_large, time_tokenizer, frame_interval: float=0.5, max_clip: int=4):
    num_frames = NUM_FRAMES_PER_CLIP
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv = conv_templates['qwen'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    (frames, time_idx, num_clips) = load_frames_as_clips(frame_paths, processor, num_frames, max_clip)
    video = processor.preprocess(frames, return_tensors='pt')['pixel_values']
    video = video.view(num_clips, num_frames, *video.shape[1:])
    video_large = processor_large.preprocess(frames, return_tensors='pt')['pixel_values']
    video_large = video_large.view(num_clips, num_frames, *video_large.shape[1:])[:, :1].contiguous()
    seqs = preprocess_time(time_idx, num_clips, time_tokenizer, frame_interval)
    seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=time_tokenizer.pad_token_id)
    compress_mask = seqs.ne(time_tokenizer.pad_token_id)
    q_tokens = preprocess_question([question], time_tokenizer)
    q_tokens = torch.nn.utils.rnn.pad_sequence(q_tokens, batch_first=True, padding_value=time_tokenizer.pad_token_id)
    qs_mask = q_tokens.ne(time_tokenizer.pad_token_id)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    return (input_ids, video, video_large, seqs, compress_mask, q_tokens, qs_mask)

def main() -> None:
    parser = argparse.ArgumentParser(description='Proactive Video QA inference with Dispider')
    parser.add_argument('--model_path', required=True, help='Path to Dispider checkpoint')
    parser.add_argument('--test_fname', required=True, help='Path to frame_input_format.json')
    parser.add_argument('--gold_fname', required=True, help='Path to corresponding gold.json')
    parser.add_argument('--output_fname', required=True, help='Path to output gold-aligned json')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id')
    parser.add_argument('--frame_interval', type=float, default=0.5, help='Seconds between frames')
    parser.add_argument('--max_clip', type=int, default=4, help='Max number of clips')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Max new tokens for generation')
    parser.add_argument('--start_idx', type=int, default=0, help='Start sample index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=None, help='End sample index (exclusive)')
    parser.add_argument('--window_size', type=int, default=0, help='Sliding window size in frames (0 = feed all frames at once)')
    parser.add_argument('--conv_mode', type=str, default='qwen')
    parser.add_argument('--alert_match_mode', type=str, default='strict', choices=ALERT_MATCH_MODES, help='How to map raw model text to ALERT/NO REPLY')
    parser.add_argument('--timing_file', type=str, default=None, help='Path to save per-sample timing JSON')
    args = parser.parse_args()
    if _IMPORT_ERROR is not None:
        raise _IMPORT_ERROR
    os.makedirs(os.path.dirname(args.output_fname) or '.', exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f'Proactive Video QA — Dispider')
    print(f"{'=' * 70}")
    print(f'Model       : {args.model_path}')
    print(f'GPU         : cuda:{args.gpu_id}')
    print(f'Max clips   : {args.max_clip}')
    print(f'Frame interval: {args.frame_interval}s')
    print(f"Window size : {(args.window_size if args.window_size > 0 else 'all frames')}")
    print(f'Input       : {args.test_fname}')
    print(f'Gold        : {args.gold_fname}')
    print(f'Output      : {args.output_fname}')
    print(f"{'=' * 70}\n")
    disable_torch_init()
    with open(args.test_fname) as f:
        data_list = json.load(f)
    gold_templates = load_gold_templates(args.gold_fname)
    end_idx = args.end_idx if args.end_idx is not None else len(data_list)
    data_list = data_list[args.start_idx:end_idx]
    print(f'Loaded {len(data_list)} samples (range [{args.start_idx}, {end_idx}))')
    device = f'cuda:{args.gpu_id}'
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f'Loading model from {model_path} ...')
    (tokenizer, model, image_processor_tuple, context_len) = load_pretrained_model(model_path, None, model_name, device_map={'': device})
    (image_processor, time_tokenizer) = image_processor_tuple
    image_processor_large = image_processor
    if time_tokenizer.pad_token is None:
        time_tokenizer.pad_token = '<pad>'
    stop_words_ids = [torch.tensor(tokenizer('<|im_end|>').input_ids).to(device)]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    print('Model loaded.\n')
    completed = 0
    total = len(data_list)
    results: list[dict[str, Any]] = []
    raw_logs: list[dict[str, Any]] = []
    per_sample_timing: list[dict[str, Any]] = []
    for (sample_idx, example) in enumerate(data_list):
        (question_id, question_level, question_text, frame_paths) = parse_frame_input(example)
        t0 = time.time()
        n_frames = len(frame_paths)
        print(f'\n[样本 {sample_idx}/{total}] id={question_id}, frames={n_frames}')
        model_output = 'NO REPLY'
        reply_frame_indices: list[int] = []
        raw_events: list[dict[str, Any]] = []
        sample_error: str | None = None
        if args.window_size > 0:
            for frame_i in range(n_frames):
                t_frame = time.time()
                start = max(0, frame_i - args.window_size + 1)
                window_paths = frame_paths[start:frame_i + 1]
                try:
                    (input_ids, image_tensor, image_tensor_large, seqs, compress_mask, qs, qs_mask) = process_frames_data(window_paths, question_text, model.config, tokenizer, image_processor, image_processor_large, time_tokenizer, args.frame_interval, args.max_clip)
                    input_ids = input_ids.unsqueeze(0).to(device=device, non_blocking=True)
                    with torch.inference_mode():
                        output_ids = model.generate(input_ids, images=image_tensor.to(dtype=torch.float16, device=device, non_blocking=True), images_large=image_tensor_large.to(dtype=torch.float16, device=device, non_blocking=True), seqs=seqs.to(device=device, non_blocking=True), compress_mask=compress_mask.to(device=device, non_blocking=True), qs=qs.to(device=device, non_blocking=True), qs_mask=qs_mask.to(device=device, non_blocking=True), ans_token=time_tokenizer(DEFAULT_ANS_TOKEN, return_tensors='pt').input_ids.to(device=device, non_blocking=True), todo_token=time_tokenizer(DEFAULT_TODO_TOKEN, return_tensors='pt').input_ids.to(device=device, non_blocking=True), q_id=None, insert_position=0, ans_position=[], do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria, use_cache=True)
                    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                except Exception as e:
                    print(f'  [ERROR] frame {frame_i}: {e}')
                    output_text = 'NO REPLY'
                    sample_error = str(e)
                is_reply = is_alert_reply(output_text, match_mode=args.alert_match_mode)
                t_elapsed = time.time() - t_frame
                current_time = round((frame_i + 1) * args.frame_interval, 2)
                raw_events.append({'mode': 'sliding_window', 'frame_index': frame_i, 'window_start': start, 'window_end': frame_i + 1, 'window_size': len(window_paths), 'time': current_time, 'raw_output': output_text, 'is_alert': is_reply})
                if frame_i < 3 or frame_i == n_frames - 1 or is_reply:
                    tag = 'REPLY' if is_reply else 'SILENT'
                    short = output_text[:80] + ('...' if len(output_text) > 80 else '')
                    print(f'  Frame {frame_i:>4d}/{n_frames} | win={len(window_paths)} | {tag:6s} | {t_elapsed:.1f}s | {short}')
                if is_reply:
                    model_output = output_text
                    reply_frame_indices.append(frame_i)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            try:
                (input_ids, image_tensor, image_tensor_large, seqs, compress_mask, qs, qs_mask) = process_frames_data(frame_paths, question_text, model.config, tokenizer, image_processor, image_processor_large, time_tokenizer, args.frame_interval, args.max_clip)
                input_ids = input_ids.unsqueeze(0).to(device=device, non_blocking=True)
                with torch.inference_mode():
                    output_ids = model.generate(input_ids, images=image_tensor.to(dtype=torch.float16, device=device, non_blocking=True), images_large=image_tensor_large.to(dtype=torch.float16, device=device, non_blocking=True), seqs=seqs.to(device=device, non_blocking=True), compress_mask=compress_mask.to(device=device, non_blocking=True), qs=qs.to(device=device, non_blocking=True), qs_mask=qs_mask.to(device=device, non_blocking=True), ans_token=time_tokenizer(DEFAULT_ANS_TOKEN, return_tensors='pt').input_ids.to(device=device, non_blocking=True), todo_token=time_tokenizer(DEFAULT_TODO_TOKEN, return_tensors='pt').input_ids.to(device=device, non_blocking=True), q_id=None, insert_position=0, ans_position=[], do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria, use_cache=True)
                output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            except Exception as e:
                print(f'  [ERROR] full-video: {e}')
                output_text = 'NO REPLY'
                sample_error = str(e)
            is_reply = is_alert_reply(output_text, match_mode=args.alert_match_mode)
            current_time = round(n_frames * args.frame_interval, 2) if n_frames else 0.0
            raw_events.append({'mode': 'one_shot', 'frame_index': max(n_frames - 1, 0), 'window_start': 0, 'window_end': n_frames, 'window_size': n_frames, 'time': current_time, 'raw_output': output_text, 'is_alert': is_reply})
            if is_reply:
                model_output = output_text
                reply_frame_indices.append(n_frames - 1)
            short = output_text[:120] + ('...' if len(output_text) > 120 else '')
            tag = 'REPLY' if is_reply else 'SILENT'
            print(f'  [结果] {tag} | {short}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        predicted_alert_times = [(frame_idx + 1) * args.frame_interval for frame_idx in reply_frame_indices]
        result = build_gold_aligned_entry_from_alert_times(question_id=question_id, question_level=question_level, question_text=question_text, predicted_alert_times=predicted_alert_times, gold_templates=gold_templates)
        results.append(result)
        raw_logs.append({'question_id': question_id, 'question_level': question_level, 'question_text': question_text, 'frame_count': n_frames, 'alert_match_mode': args.alert_match_mode, 'predicted_alert_times': predicted_alert_times, 'last_non_silent_output': model_output, 'raw_model_outputs': raw_events, 'error': sample_error})
        completed += 1
        t_total = time.time() - t0
        per_sample_timing.append({'question_id': question_id, 'question_level': question_level, 'frame_count': n_frames, 'elapsed_seconds': round(t_total, 3), 'predicted_alert_times': predicted_alert_times})
        print(f'  [耗时] {t_total:.1f}s | 累计 {completed}/{total}')
    save_json(args.output_fname, results)
    save_jsonl(derive_model_log_path(args.output_fname, 'Dispider', '_raw_outputs.jsonl'), raw_logs)
    if args.timing_file:
        save_json(args.timing_file, per_sample_timing)
        print(f'Per-sample timing saved to {args.timing_file}')
    print(f"\n{'=' * 70}")
    print(f'Inference complete. {completed} results saved to {args.output_fname}')
    print(f"{'=' * 70}")
if __name__ == '__main__':
    main()
