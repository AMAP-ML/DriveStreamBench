# DSB integration: https://github.com/showlab/videollm-online
# Source revision: cfbd7ce808a9ec666264b704a73c5db596705098; see external_baselines/NOTICE.md.
"""videollm-online native DSB inference."""
import argparse
import json
import os
import time as time_module
import sys
import torch
import torchvision
import transformers
from PIL import Image
from dataclasses import asdict
from tqdm import tqdm
import torchvision.transforms
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
build_model_and_tokenizer = None
fast_greedy_generate = None
LiveOnePlusTrainingArguments = None
_IMPORT_ERROR = None
try:
    from models import build_model_and_tokenizer, fast_greedy_generate
    from models.arguments_live import LiveOnePlusTrainingArguments
except Exception as exc:
    _IMPORT_ERROR = exc
from benchmark_utils.proactive_eval import ALERT_MATCH_MODES, all_alert_times_from_model_response_list, build_gold_aligned_entry_from_alert_times, derive_model_log_path, load_gold_templates, save_json, save_jsonl
logger = transformers.logging.get_logger('videollm_online_inference')
NO_REPLY = 'NO REPLY'
DEFAULT_VIDEOLLM_ROOT = os.environ.get('VIDEOLLM_ONLINE_ROOT', 'checkpoints/videollm')
DEFAULT_VIDEOLLM_CKPT = os.environ.get('VIDEOLLM_ONLINE_CHECKPOINT', DEFAULT_VIDEOLLM_ROOT)
DEFAULT_VIDEOLLM_LLM = os.environ.get('VIDEOLLM_ONLINE_LLM', os.path.join(DEFAULT_VIDEOLLM_ROOT, 'Meta-Llama-3-8B-Instruct'))
DEFAULT_VIDEOLLM_VISION = os.environ.get('VIDEOLLM_ONLINE_VISION', os.path.join(DEFAULT_VIDEOLLM_ROOT, 'siglip-large-patch16-384'))

def parse_args():
    parser = argparse.ArgumentParser(description='VideoLLM-online Streaming Inference')
    parser.add_argument('--resume_from_checkpoint', type=str, default=DEFAULT_VIDEOLLM_CKPT, help='PEFT checkpoint path or HuggingFace ID')
    parser.add_argument('--llm_pretrained', type=str, default=DEFAULT_VIDEOLLM_LLM, help='Base LLM model path')
    parser.add_argument('--vision_pretrained', type=str, default=DEFAULT_VIDEOLLM_VISION)
    parser.add_argument('--input_file', type=str, required=True, help='Path to frame_input_format.json')
    parser.add_argument('--gold_file', type=str, required=True, help='Path to corresponding gold.json')
    parser.add_argument('--output_file', type=str, default='pred.json')
    parser.add_argument('--frame_interval', type=float, default=0.5, help='Time interval between frames (seconds)')
    parser.add_argument('--frame_fps', type=int, default=2, help='Frame FPS (default 2, from model config)')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--image_root_remap', type=str, default=None, help="Remap image root, format 'old_prefix:new_prefix'")
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2', choices=['flash_attention_2', 'sdpa', 'eager'])
    parser.add_argument('--frame_token_interval_threshold', type=float, default=0.725, help='Threshold for deciding whether to reply at a frame')
    parser.add_argument('--system_prompt', type=str, default=None, help="Override system prompt. If None, uses data's system prompt.")
    parser.add_argument('--alert_match_mode', type=str, default='strict', choices=ALERT_MATCH_MODES, help='How to map raw model text to ALERT/NO REPLY')
    parser.add_argument('--timing_file', type=str, default=None, help='Path to save per-sample timing JSON')
    return parser.parse_args()

def remap_image_path(path, old_prefix, new_prefix):
    if path.startswith(old_prefix):
        return new_prefix + path[len(old_prefix):]
    return path

def extract_frames_and_question(conversation):
    system_prompt = None
    user_question = None
    frame_paths = []
    for turn in conversation:
        role = turn['role']
        content = turn.get('content')
        if role == 'system':
            system_prompt = content if isinstance(content, str) else str(content)
            continue
        if role == 'user':
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'image':
                            frame_paths.append(item['image'])
                        elif item.get('type') == 'text' and user_question is None:
                            user_question = item['text']
            elif isinstance(content, str) and content.strip():
                if user_question is None:
                    user_question = content.strip()
    return (system_prompt, user_question, frame_paths)

class LiveInferForBenchmark:

    def __init__(self, args):
        if _IMPORT_ERROR is not None:
            raise _IMPORT_ERROR
        model_args = LiveOnePlusTrainingArguments(llm_pretrained=args.llm_pretrained, vision_pretrained=args.vision_pretrained, attn_implementation=args.attn_implementation, resume_from_checkpoint=args.resume_from_checkpoint, output_dir='outputs/inference')
        (self.model, self.tokenizer) = build_model_and_tokenizer(is_training=False, set_vision_inside=True, **asdict(model_args))
        self.device = f'cuda:{args.gpu_id}'
        self.model.to(self.device)
        self.hidden_size = self.model.config.hidden_size
        self.frame_fps = args.frame_fps
        self.frame_interval = args.frame_interval
        self.frame_resolution = self.model.config.frame_resolution
        self.frame_num_tokens = self.model.config.frame_num_tokens
        self.frame_token_interval_id = self.model.config.frame_token_interval_id
        self.eos_token_id = self.model.config.eos_token_id
        self.frame_token_interval_threshold = args.frame_token_interval_threshold
        self.inplace_output_ids = torch.zeros(1, 200, device=self.device, dtype=torch.long)
        self._start_ids = self.tokenizer.apply_chat_template([{'role': 'system', 'content': 'placeholder'}], add_stream_prompt=True, return_tensors='pt').to(self.device)
        self._added_stream_prompt_ids = self.tokenizer.apply_chat_template([{}], add_stream_prompt=True, return_tensors='pt').to(self.device)
        self._added_stream_generation_ids = self.tokenizer.apply_chat_template([{}], add_stream_generation_prompt=True, return_tensors='pt').to(self.device)
        self.frame_resize = torchvision.transforms.Resize((self.frame_resolution, self.frame_resolution), antialias=True)
        self.reset()

    def reset(self):
        self.past_key_values = None
        self.last_ids = torch.tensor([[]], device=self.device, dtype=torch.long)
        self.video_time = 0.0

    def _load_and_encode_frame(self, frame_path):
        img = Image.open(frame_path).convert('RGB')
        tensor = torchvision.transforms.functional.pil_to_tensor(img)
        tensor = self.frame_resize(tensor.unsqueeze(0)).squeeze(0)
        tensor = tensor.unsqueeze(0).to(self.device)
        frame_embeds = self.model.visual_embed(tensor)
        return frame_embeds.view(1, -1, self.hidden_size)

    def _generate_response(self, query=None):
        if query is not None:
            self.last_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': query}], add_stream_query_prompt=True, add_generation_prompt=True, return_tensors='pt').to(self.device)
        else:
            self.last_ids = self._added_stream_generation_ids
        embed_layer = self.model.get_input_embeddings()
        embed_size = embed_layer.weight.shape[0]
        inputs_embeds = embed_layer(self.last_ids.clamp(max=embed_size - 1))
        (output_ids, self.past_key_values) = fast_greedy_generate(model=self.model, inputs_embeds=inputs_embeds, past_key_values=self.past_key_values, eos_token_id=self.eos_token_id, inplace_output_ids=self.inplace_output_ids)
        self.last_ids = output_ids[:, -1:]
        response_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        return response_text

    def _feed_frame(self, frame_embeds):
        if not self.past_key_values:
            self.last_ids = self._start_ids
        elif self.last_ids.item() == self.eos_token_id if self.last_ids.numel() == 1 else False:
            self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)
        embed_layer = self.model.get_input_embeddings()
        embed_size = embed_layer.weight.shape[0]
        inputs_embeds = torch.cat([embed_layer(self.last_ids.clamp(max=embed_size - 1)).view(1, -1, self.hidden_size), frame_embeds], dim=1)
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.past_key_values)
        self.past_key_values = outputs.past_key_values
        next_score = outputs.logits[:, -1:].softmax(dim=-1)
        if next_score[:, :, self.frame_token_interval_id] < self.frame_token_interval_threshold:
            next_score[:, :, self.frame_token_interval_id].zero_()
        self.last_ids = next_score.argmax(dim=-1)
        wants_to_reply = self.last_ids.item() != self.frame_token_interval_id
        return wants_to_reply

    def process_sample(self, system_prompt, user_question, frame_paths, frame_interval):
        self.reset()
        self._start_ids = self.tokenizer.apply_chat_template([{'role': 'system', 'content': system_prompt}], add_stream_prompt=True, return_tensors='pt').to(self.device)
        model_response_list = [{'role': 'user', 'content': user_question, 'time': frame_interval}]
        for (i, frame_path) in enumerate(frame_paths):
            current_time = round((i + 1) * frame_interval, 2)
            frame_embeds = self._load_and_encode_frame(frame_path)
            if i == 0:
                self._feed_frame(frame_embeds)
                response_text = self._generate_response(query=user_question)
                model_response_list.append({'role': 'assistant', 'content': response_text if response_text else NO_REPLY, 'time': current_time})
                continue
            wants_to_reply = self._feed_frame(frame_embeds)
            if wants_to_reply:
                response_text = self._generate_response(query=None)
                model_response_list.append({'role': 'assistant', 'content': response_text if response_text else NO_REPLY, 'time': current_time})
            else:
                model_response_list.append({'role': 'assistant', 'content': NO_REPLY, 'time': current_time})
        return model_response_list

def main():
    args = parse_args()
    print(f'Loading data: {args.input_file}')
    with open(args.input_file) as f:
        data = json.load(f)
    gold_templates = load_gold_templates(args.gold_file)
    end_idx = args.end_idx if args.end_idx is not None else len(data)
    data_slice = data[args.start_idx:end_idx]
    print(f"\n{'=' * 60}")
    print(f'  Model: VideoLLM-online ({args.resume_from_checkpoint})')
    print(f'  Base LLM: {args.llm_pretrained}')
    print(f'  Vision: {args.vision_pretrained}')
    print(f'  Samples: {len(data_slice)}  |  Frame interval: {args.frame_interval}s')
    print(f'  GPU: {args.gpu_id}')
    print(f'  Threshold: {args.frame_token_interval_threshold}')
    print(f"{'=' * 60}\n")
    print('Loading model...')
    infer = LiveInferForBenchmark(args)
    print('Model loaded successfully.\n')
    remap_rule = None
    if args.image_root_remap:
        remap_rule = args.image_root_remap.split(':', 1)
    results = []
    raw_logs: list[dict[str, object]] = []
    per_sample_timing: list[dict[str, object]] = []
    completed = 0
    for sample in tqdm(data_slice, desc='Samples'):
        qid = sample['question_id']
        question_level = sample.get('question_level', 'L4')
        conversation = sample['conversation']
        (system_prompt, user_question, frame_paths) = extract_frames_and_question(conversation)
        if system_prompt is None:
            system_prompt = 'You are a helpful assistant. Your task is to answer questions based on continuously incoming video frames.\n\nYou are monitoring a driving scene over time. At each turn, decide whether the monitored event has happened now based on current frames and history.\nOutput policy:\n1) If the event is happening now, output exactly: ALERT\n2) If the event is not happening now, output exactly: NO REPLY\n3) Any other output is invalid.\nDo not use XML tags or any extra text.'
        if args.system_prompt:
            system_prompt = args.system_prompt
        if remap_rule:
            frame_paths = [remap_image_path(p, remap_rule[0], remap_rule[1]) for p in frame_paths]
        t0 = time_module.time()
        response_list = infer.process_sample(system_prompt, user_question, frame_paths, args.frame_interval)
        predicted_alert_times = all_alert_times_from_model_response_list(response_list, match_mode=args.alert_match_mode)
        entry = build_gold_aligned_entry_from_alert_times(question_id=qid, question_level=question_level, question_text=user_question, predicted_alert_times=predicted_alert_times, gold_templates=gold_templates)
        results.append(entry)
        raw_logs.append({'question_id': qid, 'question_level': question_level, 'system_prompt': system_prompt, 'question_text': user_question, 'frame_count': len(frame_paths), 'alert_match_mode': args.alert_match_mode, 'predicted_alert_times': predicted_alert_times, 'raw_model_outputs': response_list})
        completed += 1
        elapsed = time_module.time() - t0
        n_frames = len(frame_paths)
        valid = sum((1 for r in response_list if r['role'] == 'assistant' and r['content'] != NO_REPLY))
        per_sample_timing.append({'question_id': qid, 'question_level': question_level, 'frame_count': n_frames, 'valid_replies': valid, 'elapsed_seconds': round(elapsed, 3), 'predicted_alert_times': predicted_alert_times})
        print(f'  [{qid}] {n_frames} frames | {valid} valid replies | alert_times={predicted_alert_times} | {elapsed:.1f}s | done {completed}')
    save_json(args.output_file, results)
    save_jsonl(derive_model_log_path(args.output_file, 'videollm_online', '_raw_outputs.jsonl'), raw_logs)
    if args.timing_file:
        save_json(args.timing_file, per_sample_timing)
        print(f'Per-sample timing saved to {args.timing_file}')
    print(f'\nFinished. {completed} results saved to {args.output_file}')
if __name__ == '__main__':
    main()
