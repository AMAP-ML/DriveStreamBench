# DSB integration: https://github.com/yellow-binary-tree/MMDuet2
# Source revision: 7bd91d5239adcb273df0977bc6c23faff950912c; see external_baselines/NOTICE.md.
"""MMDuet2 native DSB inference."""
from __future__ import annotations
import collections, csv, math, json, copy, re, os, sys
from dataclasses import asdict, dataclass, field
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import transformers
from transformers import TrainingArguments, HfArgumentParser
from transformers import AutoProcessor
from torchvision.io import read_video
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'train')))
from qwen_vl_utils import process_vision_info
from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from benchmark_utils.proactive_eval import ALERT_MATCH_MODES, build_answer_entries_from_alert_times, derive_model_log_path, is_alert_reply, normalize_reply_text, save_jsonl
logger = transformers.logging.get_logger('inference_benchmark')
logger.setLevel(logging.INFO)
SYSTEM_PROMPT = 'You are a helpful assistant. Your task is to answer questions based on continuously incoming video frames. Your responses should include information from the video since your last reply (if any). If the information in this segment of the video cannot answer the question, output "NO REPLY".\n\nYou are monitoring a driving scene over time. At each turn, decide whether the monitored event has happened now based on current frames and history.\nOutput policy:\n1) If the event is happening now, output exactly: ALERT\n2) If the event is not happening now, output exactly: NO REPLY\n3) Any other output is invalid.\nDo not use XML tags or any extra text.'

def _parse_csv_images(raw: str) -> list[str]:
    return [str(p) for p in json.loads(raw.replace('""', '"'))]

def _csv_row_to_input_entry(row: dict[str, str], fpg: int) -> dict:
    imgs = _parse_csv_images(row.get('image_samples', row.get('image_samples(list)', '[]')))
    question = row.get('question', '')
    n_turns = -(-len(imgs) // fpg)
    conversation = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for t in range(n_turns):
        start = t * fpg
        end = min((t + 1) * fpg, len(imgs))
        content = [{'type': 'image', 'image': p} for p in imgs[start:end]]
        if t == 0:
            content.append({'type': 'text', 'text': question})
        conversation.append({'role': 'user', 'content': content})
    return {'question_id': row.get('question_id', ''), 'question_level': row.get('question_level', 'L4'), 'conversation': conversation}

def load_csv_as_benchmark_data(csv_path: str, frame_interval: float=1.0) -> list[dict]:
    frames_per_turn = max(1, int(round(1.0 / frame_interval)))
    with open(csv_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return [_csv_row_to_input_entry(row, frames_per_turn) for row in rows]

def load_csv_as_gold_data(csv_path: str, frame_interval: float=1.0) -> list[dict]:
    entries = []
    with open(csv_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        question = row.get('question', '')
        sample_type = row.get('sample_type', 'positive')
        alert_mode = row.get('alert_mode', 'event_triggered')
        t_ref = None
        if sample_type == 'positive' and row.get('t_ref_frame'):
            t_ref = round(float(row['t_ref_frame']) * frame_interval, 3)
        entries.append({'question_id': row.get('question_id', ''), 'question_level': row.get('question_level', 'L4'), 'conversation': [{'role': 'user', 'content': question, 'time': 0.5}], 'answer': [{'content': ALERT_TEXT if sample_type == 'positive' else NO_REPLY_TEXT, 't_ref': t_ref, 'alert_mode': alert_mode}]})
    return entries

@dataclass
class BenchmarkTestArguments(TrainingArguments):
    llm_pretrained: str = os.environ.get('MMDUET2_CHECKPOINT', 'checkpoints/mmduet2/MMDuet2')
    attn_implementation: str = 'flash_attention_2'
    system_prompt: str = None
    is_online_model: bool = True
    input_assistant_turns: bool = False
    test_fname: str = ''
    gold_fname: str = ''
    output_fname: str = ''
    start_idx: int = 0
    end_idx: int = None
    frame_interval: float = 1.0
    gpu_ids: str = '0'
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 40
    alert_match_mode: str = 'strict'

def get_args():
    (args,) = HfArgumentParser(BenchmarkTestArguments).parse_args_into_dataclasses()
    return args

def _parse_gpu_ids(gpu_ids_str):
    return [int(x.strip()) for x in gpu_ids_str.split(',') if x.strip()]

def _load_model(args):
    gpu_list = _parse_gpu_ids(args.gpu_ids)
    if len(gpu_list) == 1:
        device = f'cuda:{gpu_list[0]}'
        logger.info(f'单卡模式: 加载模型到 {device}')
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.llm_pretrained, torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation).eval().to(device)
    else:
        logger.info(f'多卡模型并行模式: 使用 GPU {gpu_list}')
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.llm_pretrained, torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation, device_map='auto').eval()
    return model

def _get_input_device(model):
    if hasattr(model, 'hf_device_map'):
        first_device_id = min(model.hf_device_map.values()) if isinstance(list(model.hf_device_map.values())[0], int) else list(model.hf_device_map.values())[0]
        return f'cuda:{first_device_id}' if isinstance(first_device_id, int) else first_device_id
    return next(model.parameters()).device
NO_REPLY_TEXT = 'NO REPLY'
ALERT_TEXT = 'ALERT'

def extract_user_prompt(conversation):
    for turn in conversation:
        if turn['role'] == 'user' and isinstance(turn.get('content'), list):
            for item in turn['content']:
                if isinstance(item, dict) and item.get('type') == 'text':
                    return item['text']
    return ''

def count_frames_in_conversation(conversation):
    num_frames = 0
    for turn in conversation:
        if turn['role'] == 'user' and isinstance(turn.get('content'), list):
            for item in turn['content']:
                if isinstance(item, dict) and item.get('type') == 'image':
                    num_frames += 1
    return num_frames

class ProactiveInferenceClient:

    def __init__(self, args=None, model=None, processor=None) -> None:
        self.args = args
        self.model = model if model is not None else _load_model(args)
        self.input_device = _get_input_device(self.model)
        logger.info(f'模型输入设备: {self.input_device}')
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(args.llm_pretrained)
        self.system_prompt = args.system_prompt
        if self.system_prompt:
            logger.info('using system prompt:' + self.system_prompt[:100])
        self.input_assistant_turns = args.input_assistant_turns
        self.do_sample = args.do_sample
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.history = list()
        self.prev_frame_before_token_drop = None
        self.must_reply_prompt = 'I must reply.\n'
        self.prev_image_inputs = list()
        self.prev_video_inputs = list()
        self.all_keep_masks = list()
        self.frame_interval = args.frame_interval
        self.reset()

    def set_fps(self, fps=None, frame_interval=None):
        assert fps is not None or frame_interval is not None
        assert not (fps is not None and frame_interval is not None)
        if fps is not None:
            self.frame_fps = fps
            self.frame_interval = 1 / self.frame_fps
        else:
            self.frame_interval = frame_interval
            self.frame_fps = 1 / self.frame_interval

    def reset(self):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.video_time = 0
        self.frame_idx = 0
        self.video_tensor = None
        self.past_key_values = None
        self.history = list()
        self.prev_frame_before_token_drop = None
        self.prev_image_inputs = list()
        self.prev_video_inputs = list()
        self.all_keep_masks = list()
        self._cached_pixel_values = None
        self._cached_image_grid_thw = None
        self._cached_pixel_values_videos = None
        self._cached_video_grid_thw = None
        self._cached_image_count = 0
        self._cached_video_count = 0
        if hasattr(self.model, 'reset_status'):
            self.model.reset_status()

    def input_query_stream(self, conversation):
        if conversation[0]['role'] != 'system':
            if self.system_prompt:
                self.query_queue.append({'role': 'system', 'content': self.system_prompt})
            else:
                self.query_queue.append(conversation[0])
        else:
            self.query_queue.append(conversation[0])
            conversation = conversation[1:]
        for turn in conversation:
            if self.input_assistant_turns or turn['role'] == 'user':
                self.query_queue.append(turn)

    def _recursive_stat_num_frames(self, inputs):
        num_frames = 0
        if isinstance(inputs, (list, tuple)):
            for input_item in inputs:
                if isinstance(input_item, (torch.Tensor, Image.Image, np.ndarray)):
                    num_frames += 1
                elif isinstance(input_item, (list, tuple)):
                    num_frames += self._recursive_stat_num_frames(input_item)
        return num_frames

    def _encode_query(self, debug_print=False):
        newly_added_turns = list()
        while True:
            query = self.query_queue.popleft()
            self.history.append(query)
            newly_added_turns.append(query)
            if query['role'] in ['system', 'assistant'] or query.get('skip_inference', False):
                pass
            else:
                break
        text = self.processor.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        if query.get('must_reply', False):
            text += self.must_reply_prompt
        if debug_print:
            print('DEBUG text before generate:', text[:200])
        (new_image_inputs, new_video_inputs) = process_vision_info(newly_added_turns)
        if new_image_inputs is not None:
            self.prev_image_inputs.extend(new_image_inputs)
        if new_video_inputs is not None:
            self.prev_video_inputs.extend(new_video_inputs)
        num_new_images = self._recursive_stat_num_frames(new_image_inputs)
        num_new_videos = self._recursive_stat_num_frames(new_video_inputs)
        num_frames = num_new_images + num_new_videos
        self.video_time += num_frames * self.frame_interval
        self.history[-1]['time'] = self.video_time
        if new_image_inputs and len(new_image_inputs) > 0:
            new_encoded = self.processor.image_processor(images=new_image_inputs, videos=None, return_tensors='pt')
            new_pv = new_encoded['pixel_values'].to(self.input_device)
            new_thw = new_encoded['image_grid_thw'].to(self.input_device)
            if self._cached_pixel_values is not None:
                self._cached_pixel_values = torch.cat([self._cached_pixel_values, new_pv], dim=0)
                self._cached_image_grid_thw = torch.cat([self._cached_image_grid_thw, new_thw], dim=0)
            else:
                self._cached_pixel_values = new_pv
                self._cached_image_grid_thw = new_thw
            self._cached_image_count += len(new_image_inputs)
        if new_video_inputs and len(new_video_inputs) > 0:
            new_encoded = self.processor.image_processor(images=None, videos=new_video_inputs, return_tensors='pt')
            new_pv = new_encoded['pixel_values_videos'].to(self.input_device)
            new_thw = new_encoded['video_grid_thw'].to(self.input_device)
            if self._cached_pixel_values_videos is not None:
                self._cached_pixel_values_videos = torch.cat([self._cached_pixel_values_videos, new_pv], dim=0)
                self._cached_video_grid_thw = torch.cat([self._cached_video_grid_thw, new_thw], dim=0)
            else:
                self._cached_pixel_values_videos = new_pv
                self._cached_video_grid_thw = new_thw
            self._cached_video_count += len(new_video_inputs)
        merge_length = self.processor.image_processor.merge_size ** 2
        image_token = '<|image_pad|>'
        video_token = '<|video_pad|>'
        if self._cached_image_grid_thw is not None:
            img_idx = 0
            while image_token in text:
                num_pads = int(self._cached_image_grid_thw[img_idx].prod() // merge_length)
                text = text.replace(image_token, '<|placeholder|>' * num_pads, 1)
                img_idx += 1
            text = text.replace('<|placeholder|>', image_token)
        if self._cached_video_grid_thw is not None:
            vid_idx = 0
            while video_token in text:
                num_pads = int(self._cached_video_grid_thw[vid_idx].prod() // merge_length)
                text = text.replace(video_token, '<|placeholder|>' * num_pads, 1)
                vid_idx += 1
            text = text.replace('<|placeholder|>', video_token)
        text_inputs = self.processor.tokenizer([text], padding=True, return_tensors='pt')
        inputs = text_inputs
        if self._cached_pixel_values is not None:
            inputs['pixel_values'] = self._cached_pixel_values
            inputs['image_grid_thw'] = self._cached_image_grid_thw
        if self._cached_pixel_values_videos is not None:
            inputs['pixel_values_videos'] = self._cached_pixel_values_videos
            inputs['video_grid_thw'] = self._cached_video_grid_thw
        inputs = inputs.to(self.input_device)
        if self.model.model.all_keep_masks:
            assert inputs.input_ids.size(0) == 1, 'token drop in inference only support batch size 1 now'
            keep_mask = torch.ones_like(inputs.input_ids, dtype=torch.bool)
            old_keep_mask = torch.cat(self.model.model.all_keep_masks, dim=1)
            keep_mask[:, :old_keep_mask.size(1)] = old_keep_mask
            inputs['input_ids'] = inputs.input_ids[keep_mask].unsqueeze(0)
            inputs['attention_mask'] = inputs.attention_mask[keep_mask].unsqueeze(0)
        model_output = self.model.generate(**inputs, max_new_tokens=16, past_key_values=self.past_key_values, return_dict_in_generate=True, drop_method='none', drop_threshold=1.0, drop_absolute=True, do_sample=self.do_sample, temperature=self.temperature, top_k=self.top_k)
        self.past_key_values = model_output.past_key_values
        output_token_ids = model_output.sequences
        output_token_ids = output_token_ids[:, inputs.input_ids.size(1):]
        reply_text = self.processor.batch_decode(output_token_ids, skip_special_tokens=True)[0]
        if query.get('must_reply', False):
            reply_text = self.must_reply_prompt + reply_text
        self.history.append({'role': 'assistant', 'content': reply_text, 'time': self.video_time})
        if debug_print:
            print(f'  [帧推理] time={self.video_time:.2f}s, reply="{reply_text}"')

    def inference(self, debug_print=False):
        step = 0
        while self.query_queue:
            self._encode_query(debug_print=debug_print)
            step += 1
        if debug_print:
            print(f'  [推理完成] 共 {step} 轮编码, 剩余未处理帧: {len(self.query_queue)}')
        return {'conversation': copy.deepcopy(self.history), 'drop_ratio': copy.deepcopy(self.model.model.all_drop_ratios)}

def parse_model_output(conversation, alert_match_mode: str='strict'):
    alert_times: list[float] = []
    for turn in conversation:
        if turn['role'] == 'assistant':
            content = turn.get('content', '').strip()
            content = content.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
            if is_alert_reply(content, match_mode=alert_match_mode):
                alert_time = turn.get('time')
                if alert_time is not None:
                    alert_times.append(float(alert_time))
    return build_answer_entries_from_alert_times(alert_times)

def build_benchmark_output(example, conversation, alert_match_mode: str='strict'):
    question_id = example['question_id']
    question_level = example.get('question_level', 'L4')
    user_prompt = extract_user_prompt(example['conversation'])
    first_user_time = 0.5
    for turn in conversation:
        if turn['role'] == 'user' and turn.get('time') is not None:
            first_user_time = turn['time']
            break
    new_conversation = [{'role': 'user', 'content': user_prompt, 'time': first_user_time}]
    return {'question_id': question_id, 'question_level': question_level, 'conversation': new_conversation, 'answer': parse_model_output(conversation, alert_match_mode=alert_match_mode)}

def main():
    args = get_args()
    print(f'Arguments: {args}')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        args.gpu_ids = str(local_rank)
        print(f'[Rank {local_rank}] 分配 GPU: cuda:{local_rank}')
    if args.test_fname.endswith('.csv'):
        data_list = load_csv_as_benchmark_data(args.test_fname, args.frame_interval)
        logger.info(f'从 CSV 加载了 {len(data_list)} 条数据')
        if args.gold_fname and args.gold_fname.endswith('.csv'):
            gold_data = load_csv_as_gold_data(args.gold_fname, args.frame_interval)
            logger.info(f'从 CSV gold 加载了 {len(gold_data)} 条 gold 数据')
    else:
        data_list = json.load(open(args.test_fname))
    total_samples = len(data_list)
    args.end_idx = total_samples if args.end_idx is None else args.end_idx
    print(f"\n{'=' * 70}")
    print(f'Benchmark 推理开始')
    print(f'输入文件: {args.test_fname}')
    print(f'输出文件: {args.output_fname}')
    print(f'总样本数: {total_samples}')
    print(f'处理范围: [{args.start_idx}, {args.end_idx})')
    print(f'帧间隔: {args.frame_interval} 秒')
    print(f'World size: {world_size}, Local rank: {local_rank}')
    print(f"{'=' * 70}\n")
    if world_size > 1:
        rank_indices = []
        for i in range(args.start_idx, args.end_idx):
            if (i - args.start_idx) % world_size == local_rank:
                rank_indices.append(i)
        print(f'[Rank {local_rank}] 处理 {len(rank_indices)} 个样本 (索引: {rank_indices[:5]}...)')
        if len(rank_indices) == 0:
            print(f'[Rank {local_rank}] 没有分配到样本，退出')
            sys.exit(0)
        output_fname = args.output_fname.replace('.json', f'_rank{local_rank}.json')
    else:
        rank_indices = list(range(args.start_idx, args.end_idx))
        output_fname = args.output_fname
    existing_question_ids = set()
    if os.path.exists(output_fname):
        if output_fname.endswith('.json'):
            with open(output_fname) as f:
                try:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        existing_question_ids = {item['question_id'] for item in existing_data}
                except json.JSONDecodeError:
                    pass
        print(f'从 {output_fname} 加载了 {len(existing_question_ids)} 条已有结果')
    wrapper = ProactiveInferenceClient(args)
    wrapper.set_fps(frame_interval=args.frame_interval)
    results = []
    raw_logs: list[dict[str, object]] = []
    completed = 0
    skipped_oom = []
    for example_i in tqdm(rank_indices, desc=f'Rank {local_rank}'):
        example = data_list[example_i]
        question_id = example['question_id']
        if question_id in existing_question_ids:
            continue
        try:
            wrapper.reset()
            wrapper.input_query_stream(example['conversation'])
            model_outputs = wrapper.inference(debug_print=False)
        except torch.cuda.OutOfMemoryError:
            print(f'\n[Rank {local_rank}] OOM: 样本 {question_id} (index={example_i}) 序列过长，跳过')
            skipped_oom.append(question_id)
            torch.cuda.empty_cache()
            wrapper.reset()
            continue
        output_entry = build_benchmark_output(example, model_outputs['conversation'], alert_match_mode=args.alert_match_mode)
        results.append(output_entry)
        raw_logs.append({'question_id': question_id, 'question_level': example.get('question_level', 'L4'), 'question_text': extract_user_prompt(example['conversation']), 'frame_count': count_frames_in_conversation(example['conversation']), 'alert_match_mode': args.alert_match_mode, 'predicted_alert_times': [item.get('t_ref') for item in output_entry['answer'] if item.get('t_ref') is not None], 'raw_model_outputs': [{**turn, 'normalized_content': normalize_reply_text(turn.get('content')) if turn.get('role') == 'assistant' else turn.get('content'), 'is_alert': is_alert_reply(turn.get('content'), match_mode=args.alert_match_mode) if turn.get('role') == 'assistant' else None} for turn in model_outputs['conversation']], 'drop_ratio': model_outputs.get('drop_ratio')})
        completed += 1
    if skipped_oom:
        print(f'\n[Rank {local_rank}] 因 OOM 跳过了 {len(skipped_oom)} 个样本: {skipped_oom}')
    if results:
        if os.path.exists(output_fname):
            with open(output_fname) as f:
                try:
                    existing = json.load(f)
                    if isinstance(existing, list):
                        results = existing + results
                except json.JSONDecodeError:
                    pass
        with open(output_fname, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        save_jsonl(derive_model_log_path(output_fname, 'mmduet2', '_raw_outputs.jsonl'), raw_logs)
        print(f'\n[Rank {local_rank}] 完成 {completed} 条结果，写入到 {output_fname}')
    if world_size > 1:
        del wrapper
        torch.cuda.empty_cache()
        done_dir = os.path.dirname(output_fname)
        done_file = os.path.join(done_dir, f'.done_rank{local_rank}')
        with open(done_file, 'w') as f:
            f.write('done')
        if local_rank == 0:
            import time as _time
            print('\n等待所有 rank 完成...')
            for wait_rank in range(world_size):
                wait_file = os.path.join(done_dir, f'.done_rank{wait_rank}')
                waited = 0
                while not os.path.exists(wait_file):
                    _time.sleep(5)
                    waited += 5
                    if waited % 60 == 0:
                        print(f'  等待 rank {wait_rank} 已 {waited}s...')
                    if waited > 3600:
                        print(f'  超时: rank {wait_rank} 未完成，跳过')
                        break
            print('所有 rank 已完成，开始合并结果...')
            merged_data = []
            for rank_id in range(world_size):
                rank_file = args.output_fname.replace('.json', f'_rank{rank_id}.json')
                if os.path.exists(rank_file):
                    with open(rank_file) as f:
                        rank_data = json.load(f)
                    merged_data.extend(rank_data)
                    print(f'  合并 {rank_file}: {len(rank_data)} 条')
                else:
                    print(f'  警告: {rank_file} 不存在，跳过')
            with open(args.output_fname, 'w') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            print(f'合并完成: {len(merged_data)} 条结果 -> {args.output_fname}')
            for rank_id in range(world_size):
                done_f = os.path.join(done_dir, f'.done_rank{rank_id}')
                if os.path.exists(done_f):
                    os.remove(done_f)
    print(f"\n{'=' * 70}")
    print(f'[Rank {local_rank}] 推理完成')
    print(f'输出文件: {output_fname}')
    print(f'完成样本数: {completed}')
    print(f"{'=' * 70}")
if __name__ == '__main__':
    main()
