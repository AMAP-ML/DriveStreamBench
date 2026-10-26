# DSB integration: https://github.com/MCG-NJU/StreamForest
# Source revision: f9d14670ff451ff5865638e17c9958e99bf2fec5; see external_baselines/NOTICE.md.
"""StreamForest native DSB inference."""
import os
import sys
import json
import time
import copy
import argparse
import numpy as np
from PIL import Image
import torch
import torch.backends.cuda
torch.backends.cuda.matmul.allow_tf32 = True
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
load_pretrained_model = None
tokenizer_image_token = None
KeywordsStoppingCriteria = None
IMAGE_TOKEN_INDEX = None
DEFAULT_IMAGE_TOKEN = None
conv_templates = None
SeparatorStyle = None
_IMPORT_ERROR = None
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
except Exception as exc:
    _IMPORT_ERROR = exc
from benchmark_utils.proactive_eval import ALERT_MATCH_MODES, build_gold_aligned_entry_from_alert_times, derive_model_log_path, is_alert_reply, load_gold_templates, parse_frame_input_sample, save_json, save_jsonl

def parse_frame_input(sample):
    (question_id, question_level, question_text, frame_paths) = parse_frame_input_sample(sample)
    conversation = sample['conversation']
    system_prompt = None
    for msg in conversation:
        if msg['role'] == 'system':
            system_prompt = msg['content']
            continue
    return (question_id, question_level, system_prompt, question_text, frame_paths)

def load_frames_as_video(frame_paths, max_frames):
    total = len(frame_paths)
    if total <= max_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, max_frames, dtype=int).tolist()
    frames = []
    for idx in indices:
        path = frame_paths[idx]
        img = Image.open(path).convert('RGB')
        frames.append(np.array(img))
    return (np.stack(frames), indices)

def build_time_msg(num_frames, total_frames, fps=2.0, time_msg_type='short_online_v2'):
    duration = (total_frames - 1) / fps
    sec = [round(i / fps, 1) for i in range(total_frames)]
    sampled_sec = np.linspace(0, len(sec) - 1, num_frames, dtype=int)
    sampled_sec = [sec[i] for i in sampled_sec]
    if time_msg_type == 'short_online':
        return f'\nThe video segment contains {num_frames} frames sampled from the past {sampled_sec[-1] - sampled_sec[0]:.1f} seconds ago up to the present moment. '
    elif time_msg_type == 'short_online_v2':
        return f'\nThe video contains {num_frames} frames sampled from the past {sampled_sec[-1] - sampled_sec[0]:.1f} seconds ago ({sampled_sec[0]:.1f}s of the entire video) up to the present moment ({sampled_sec[-1]:.1f}s of the entire video). '
    elif time_msg_type == 'short':
        return f'\nThe video lasts for {duration:.2f} seconds, and {num_frames} frames are uniformly sampled from it. '
    else:
        return ''

def decode_generated_tokens(tokenizer, output_ids):
    """The native inputs_embeds generation path returns only generated tokens."""
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def require_initialized_weights(model):
    if any(getattr(parameter, 'is_meta', False) for parameter in model.parameters()):
        raise RuntimeError('Model loading left uninitialized meta parameters; inference cannot continue.')


def run_inference(args):
    if _IMPORT_ERROR is not None:
        raise _IMPORT_ERROR
    gpu_id = int(args.gpu_ids.split(',')[0])
    device = f'cuda:{gpu_id}'
    print(f'[初始化] 加载模型: {args.model_path}')
    print(f'[初始化] 设备: {device}')
    from llava.mm_utils import get_model_name_from_path
    model_name = get_model_name_from_path(args.model_path)
    overwrite_config = {'delay_load': True}
    if os.environ.get('DSB_VISION_CHECKPOINT'):
        overwrite_config['mm_vision_tower'] = os.environ['DSB_VISION_CHECKPOINT']
    (tokenizer, model, image_processor, context_len) = load_pretrained_model(args.model_path, None, model_name, device_map=device, multimodal=True, attn_implementation='sdpa', overwrite_config=overwrite_config)
    require_initialized_weights(model)
    model.eval()
    with open(args.test_fname, 'r') as f:
        test_data = json.load(f)
    gold_templates = load_gold_templates(args.gold_fname)
    start_idx = args.start_idx if args.start_idx >= 0 else 0
    end_idx = args.end_idx if args.end_idx > 0 else len(test_data)
    end_idx = min(end_idx, len(test_data))
    test_data = test_data[start_idx:end_idx]
    total = len(test_data)
    os.makedirs(os.path.dirname(args.output_fname) or '.', exist_ok=True)
    completed = 0
    global_t0 = time.time()
    results = []
    raw_logs: list[dict[str, object]] = []
    for (i, sample) in enumerate(test_data):
        (question_id, question_level, system_prompt, question_text, frame_paths) = parse_frame_input(sample)
        t0 = time.time()
        print(f'\n[样本 {i}/{total}] id={question_id}')
        print(f'  [输入] 帧数={len(frame_paths)}, 问题={question_text[:80]}')
        predicted_alert_times: list[float] = []
        raw_events: list[dict[str, object]] = []
        sample_error: str | None = None
        try:
            n_frames = len(frame_paths)
            for frame_i in range(n_frames):
                window_paths = frame_paths[max(0, frame_i - args.max_frames + 1):frame_i + 1]
                (frames, sampled_indices) = load_frames_as_video(window_paths, min(args.max_frames, len(window_paths)))
                frames_tensor = image_processor.preprocess(list(frames), return_tensors='pt')['pixel_values'].to(device=device, dtype=model.dtype)
                time_msg = build_time_msg(len(frames), len(window_paths), fps=args.fps, time_msg_type=args.time_msg)
                question = DEFAULT_IMAGE_TOKEN + '\n' + time_msg.rstrip() + ' ' + question_text
                conv = conv_templates[args.conv_template].copy()
                if system_prompt:
                    conv.system = f'<|im_start|>system\n{system_prompt}'
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
                attention_mask = input_ids.ne(tokenizer.pad_token_id or tokenizer.eos_token_id).to(device)
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
                with torch.inference_mode():
                    output_ids = model.generate(input_ids, attention_mask=attention_mask, images=[frames_tensor], modalities=['video'], do_sample=False, temperature=0, max_new_tokens=args.max_new_tokens, use_cache=True, stopping_criteria=[stopping_criteria], pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                prediction = decode_generated_tokens(tokenizer, output_ids)
                current_time = round((frame_i + 1) * (1.0 / args.fps), 2)
                tag = 'REPLY' if is_alert_reply(prediction, match_mode=args.alert_match_mode) else 'SILENT'
                raw_events.append({'frame_index': frame_i, 'window_start': max(0, frame_i - args.max_frames + 1), 'window_end': frame_i + 1, 'window_size': len(window_paths), 'time': current_time, 'raw_output': prediction, 'is_alert': is_alert_reply(prediction, match_mode=args.alert_match_mode)})
                if frame_i < 3 or frame_i == n_frames - 1 or is_alert_reply(prediction, match_mode=args.alert_match_mode):
                    print(f'  Frame {frame_i:>4d}/{n_frames} | t={current_time:>7.1f}s | win={len(window_paths):>3d} | {tag:6s} | {prediction[:80]}')
                if is_alert_reply(prediction, match_mode=args.alert_match_mode):
                    predicted_alert_times.append(current_time)
        except Exception as e:
            print(f'  [错误] {e}')
            predicted_alert_times = []
            sample_error = str(e)
        result = build_gold_aligned_entry_from_alert_times(question_id=question_id, question_level=question_level, question_text=question_text, predicted_alert_times=predicted_alert_times, gold_templates=gold_templates)
        results.append(result)
        raw_logs.append({'question_id': question_id, 'question_level': question_level, 'system_prompt': system_prompt, 'question_text': question_text, 'frame_count': len(frame_paths), 'alert_match_mode': args.alert_match_mode, 'predicted_alert_times': predicted_alert_times, 'raw_model_outputs': raw_events, 'error': sample_error})
        completed += 1
        elapsed = time.time() - t0
        total_elapsed = time.time() - global_t0
        speed = completed / total_elapsed if total_elapsed > 0 else 0
        print(f'  [结果] alert_times={predicted_alert_times}')
        print(f'  [耗时] {elapsed:.1f}s | 累计 {completed}/{total} | 速度 {speed:.2f} 条/秒')
    save_json(args.output_fname, results)
    save_jsonl(derive_model_log_path(args.output_fname, 'StreamForest', '_raw_outputs.jsonl'), raw_logs)
    print(f'\n[完成] 共处理 {completed} 条, 结果保存到 {args.output_fname}')

def main():
    parser = argparse.ArgumentParser(description='StreamForest ProactiveVideoQA Inference')
    parser.add_argument('--model_path', type=str, required=True, help='StreamForest 模型 checkpoint 路径')
    parser.add_argument('--test_fname', type=str, required=True, help='frame_input_format.json 文件路径')
    parser.add_argument('--gold_fname', type=str, required=True, help='对应 gold.json 文件路径')
    parser.add_argument('--output_fname', type=str, default='outputs/pred.json', help='输出 gold-aligned json 文件路径')
    parser.add_argument('--gpu_ids', type=str, default='0', help="GPU ID, 如 '0' 或 '0,1,2,3'")
    parser.add_argument('--max_frames', type=int, default=64, help='每个样本最大采样帧数')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='生成最大 token 数')
    parser.add_argument('--fps', type=float, default=2.0, help='帧率 (用于时间描述)')
    parser.add_argument('--time_msg', type=str, default='short_online_v2', choices=['short', 'short_online', 'short_online_v2', 'none'], help='时间描述类型')
    parser.add_argument('--conv_template', type=str, default='qwen_2', help='对话模板')
    parser.add_argument('--alert_match_mode', type=str, default='strict', choices=ALERT_MATCH_MODES, help='How to map raw model text to ALERT/NO REPLY')
    parser.add_argument('--start_idx', type=int, default=-1, help='起始样本索引 (用于分片)')
    parser.add_argument('--end_idx', type=int, default=-1, help='结束样本索引 (用于分片)')
    args = parser.parse_args()
    run_inference(args)
if __name__ == '__main__':
    main()
