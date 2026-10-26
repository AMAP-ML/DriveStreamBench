"""CPU checks for native baseline commands, payloads and prediction conversion."""
from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from external_baselines.native import (
    BASELINES, ROOT, build_command, convert_raw_outputs, metadata, run_native,
)
from external_baselines.compatibility import configure_dispider_source, verify_dispider_source
from drivestreambench.models.streaming_baselines._base_external import normalize_to_alert_or_no_reply


def input_sample(paths: list[str] | None = None) -> dict:
    return {"question_id": "sample", "question_level": "L4", "conversation": [
        {"role": "system", "content": "Watch the instructed event."},
        {"role": "user", "content": [
            *[{"type": "image", "image": path} for path in (paths or ["a.jpg", "b.jpg"])],
            {"type": "text", "text": "Tell me when the leading vehicle starts."},
        ]},
    ]}


def load_nodes(name: str, names: tuple[str, ...], namespace: dict | None = None) -> dict:
    """Execute selected real inference definitions with CPU dependency doubles."""
    path = ROOT / "native_inference" / f"{name}.py"
    tree = ast.parse(path.read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    scope = {"__name__": "native_contract", **(namespace or {})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), scope)
    return scope


class ExternalCommandTests(unittest.TestCase):
    def test_separate_checkpoint_components_reach_the_native_loaders(self):
        for name in ("dispider", "streamforest", "m4_omnimmi"):
            command, env, _ = build_command(name, install_dir=Path("/source"), checkpoint="/weights",
                input_path=Path("input"), gold_path=Path("gold"), native_output=Path("output"),
                vision="/vision/clip-vit-large-patch14", compressor="/weights/stream_compressor")
            self.assertEqual(env["DSB_VISION_CHECKPOINT"], "/vision/clip-vit-large-patch14")
            if name == "dispider":
                self.assertEqual(env["DSB_DISPIDER_COMPRESSOR"], "/weights/stream_compressor")
            if name == "m4_omnimmi":
                self.assertEqual(env["M4_VISION_CHECKPOINT"], "/vision/clip-vit-large-patch14")

    def test_dispider_source_uses_user_networking_and_portable_vision_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "dispider/model/multimodal_encoder/clip_encoder.py"
            source.parent.mkdir(parents=True)
            source.write_text('''class CLIPVisionTower:
    def __init__(self, vision_tower):
        self.proxies = {"https": "upstream-network"}
        self.vision_tower_name = vision_tower
    def load_model(self):
        return Model.from_pretrained(self.vision_tower_name, proxies=self.proxies)
''')
            configure_dispider_source(base)
            verify_dispider_source(base)
            configured = source.read_text()
            configure_dispider_source(base)
            self.assertEqual(source.read_text(), configured)
            captured = {}
            def load(path, **kwargs): captured.update(path=path, kwargs=kwargs)
            scope = {"Model": SimpleNamespace(from_pretrained=load)}
            exec(configured, scope)
            with patch.dict("os.environ", {"DSB_VISION_CHECKPOINT": "/portable/vision"}):
                scope["CLIPVisionTower"]("original").load_model()
            self.assertEqual(captured, {"path": "/portable/vision", "kwargs": {}})

    def test_each_command_matches_its_actual_entry_flags(self):
        for name in BASELINES:
            with self.subTest(name=name):
                command, env, cwd = build_command(name, install_dir=Path("/models/source"),
                    checkpoint="/models/checkpoint", input_path=Path("/data/input.json"),
                    gold_path=Path("/data/gold.json"), native_output=Path("/outputs/native.json"),
                    fps=metadata(name)["paper_fps"], gpus="3,5", llm="/models/llm", vision="/models/vision")
                tree = ast.parse(Path(command[1]).read_text())
                flags = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                         for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
                if name == "mmduet2":
                    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BenchmarkTestArguments")
                    flags |= {"--" + node.target.id for node in cls.body if isinstance(node, ast.AnnAssign)}
                    flags.add("--output_dir")
                passed = {value for value in command[2:] if value.startswith("--")}
                self.assertFalse(passed - flags, (name, passed - flags))
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3,5")
                self.assertEqual(cwd, Path("/models/source") / metadata(name)["working_directory"])
                self.assertIn("/models/source", env["PYTHONPATH"])
                if name == "dispider":
                    self.assertEqual(command[command.index("--window_size") + 1], "8")
                    self.assertEqual(command[command.index("--frame_interval") + 1], "1.0")
                if name == "videollm_online":
                    self.assertEqual(command[command.index("--frame_interval") + 1], "0.5")

    def test_video_llm_requires_its_three_checkpoint_components(self):
        with self.assertRaisesRegex(ValueError, "--llm and --vision"):
            build_command("videollm_online", install_dir=Path("/source"), checkpoint="checkpoint",
                input_path=Path("input"), gold_path=Path("gold"), native_output=Path("output"))

    def test_complete_batch_execution_and_output_conversion_for_every_model(self):
        for name in BASELINES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                image = base / "frame.jpg"
                image.write_bytes(b"frame")
                sample = input_sample([str(image), str(image)])
                input_path, gold_path, output_path = base / "input.json", base / "gold.json", base / "prediction.jsonl"
                input_path.write_text(json.dumps([sample]))
                gold_path.write_text(json.dumps([{"question_id": "sample", "answer": []}]))
                checkpoint = base / "checkpoint"
                (checkpoint / "stream_compressor").mkdir(parents=True)
                (checkpoint / "config.json").write_text(json.dumps({"architectures": ["LongQwen2ForCausalLM"]}))
                (checkpoint / "stream_compressor/config.json").write_text("{}")

                def native_process(command, **kwargs):
                    flag = "--output_file" if name == "videollm_online" else "--output_fname"
                    target = Path(command[command.index(flag) + 1])
                    target.write_text(json.dumps([{"question_id": "sample"}]))
                    raw = "(Current Time = 2.0s) Assistant: REQUIREMENT MEET AT 0.5s" if name == "m4_omnimmi" else "No."
                    raw_path = target.parent / metadata(name)["raw_directory"] / "predictions_raw_outputs.jsonl"
                    raw_path.parent.mkdir()
                    raw_path.write_text(json.dumps({"question_id": "sample", "frame_count": 2,
                        "raw_model_outputs": [{"role": "assistant", "time": 2.0, "raw_output": raw}]}) + "\n")
                    self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")

                with patch("external_baselines.native.verify_install"), patch("external_baselines.native.subprocess.run", side_effect=native_process):
                    run_native(name, install_dir=base, checkpoint=str(checkpoint), input_path=input_path,
                        gold_path=gold_path, output_path=output_path, llm="llm", vision="vision")
                result = json.loads(output_path.read_text())
                self.assertEqual(result["turns"][0]["reply"], "ALERT" if name == "m4_omnimmi" else "No.")
                self.assertEqual(result["turns"][0]["video_time"], 0.5 if name == "m4_omnimmi" else 2.0)


class RawOutputTests(unittest.TestCase):
    def test_free_form_is_preserved_including_negative_words(self):
        inputs = [input_sample()]
        for text in ["NO", "No.", "FALSE", "0", "NO REPLY", "The car starts.", "not an ALERT", "", None]:
            with self.subTest(text=text):
                result = convert_raw_outputs("dispider", inputs, [{"question_id": "sample", "frame_count": 2,
                    "raw_model_outputs": [{"time": 1.0, "raw_output": text}]}], [{"question_id": "sample"}])
                self.assertEqual(result[0]["turns"][0]["reply"], "" if text is None else text)
                self.assertEqual(normalize_to_alert_or_no_reply(text), "" if text is None else text)

    def test_m4_only_accepts_the_native_requirement_trigger(self):
        result = convert_raw_outputs("m4_omnimmi", [input_sample()], [{"question_id": "sample", "frame_count": 2,
            "raw_model_outputs": [
                {"time": 0.0, "raw_output": "ALERT", "parsed_output": "ALERT"},
                {"time": 12.0, "raw_output": "(Current Time = 12.0s) Assistant: REQUIREMENT MEET AT 0.0s", "parsed_output": "description"},
            ]}], [{"question_id": "sample"}])
        turns = result[0]["turns"]
        self.assertEqual([turn["reply"] for turn in turns], ["NO REPLY", "ALERT"])
        self.assertEqual(turns[1]["video_time"], 0.0)
        self.assertEqual(turns[1]["extra"]["native_time"], 12.0)

    def test_conversion_rejects_missing_duplicate_errors_empty_and_bad_times(self):
        good = {"question_id": "sample", "frame_count": 2, "raw_model_outputs": [{"time": 1.0, "raw_output": "NO REPLY"}]}
        bad_logs = [[], [good, good], [{**good, "error": "inference failed"}],
                    [{**good, "frame_count": 1}], [{**good, "raw_model_outputs": []}],
                    [{**good, "raw_model_outputs": [{"time": float("nan"), "raw_output": "ALERT"}]}]]
        for rows in bad_logs:
            with self.subTest(rows=rows), self.assertRaises((ValueError, RuntimeError)):
                convert_raw_outputs("dispider", [input_sample()], rows, [{"question_id": "sample"}])


class NativePayloadTests(unittest.TestCase):
    def test_simplestream_rejects_an_undecodable_frame_instead_of_scoring_partial_input(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "input.json").write_text(json.dumps([input_sample()]))
            def cannot_decode(path): raise OSError("invalid image")
            scope = load_nodes("simplestream", ("main",), {
                "argparse": argparse, "json": json, "os": os, "sys": sys, "time": time,
                "ALERT_MATCH_MODES": ("strict", "lenient"),
                "load_gold_templates": lambda path: {},
                "parse_frame_input": lambda sample: ("sample", "L4", "system", "instruction", ["bad.jpg"]),
                "ProactiveQAModel": lambda **kwargs: SimpleNamespace(),
                "Image": SimpleNamespace(open=cannot_decode),
            })
            argv = ["simplestream", "--model_path", "checkpoint", "--test_fname", str(base / "input.json"),
                    "--gold_fname", str(base / "gold.json"), "--output_fname", str(base / "output.json")]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, "Cannot decode image"):
                scope["main"]()
            self.assertFalse((base / "output.json").exists())

    def test_streamforest_decodes_the_returned_new_tokens_without_a_second_prefix_slice(self):
        tokens = [[41, 42]]
        tokenizer = SimpleNamespace(batch_decode=lambda output, **kwargs: ["ALERT"] if output is tokens else [""])
        scope = load_nodes("streamforest", ("decode_generated_tokens",))
        self.assertEqual(scope["decode_generated_tokens"](tokenizer, tokens), "ALERT")

    def test_streamforest_refuses_uninitialized_weights(self):
        scope = load_nodes("streamforest", ("require_initialized_weights",))
        scope["require_initialized_weights"](SimpleNamespace(parameters=lambda: [SimpleNamespace(is_meta=False)]))
        with self.assertRaisesRegex(RuntimeError, "uninitialized meta"):
            scope["require_initialized_weights"](SimpleNamespace(parameters=lambda: [SimpleNamespace(is_meta=True)]))

    def test_dispider_and_streamforest_receive_the_ordered_input_frames(self):
        scope = load_nodes("dispider", ("parse_frame_input",))
        sample = input_sample()
        self.assertEqual(scope["parse_frame_input"](sample)[-1], ["a.jpg", "b.jpg"])
        shared_path = str(ROOT / "shared")
        with patch.object(sys, "path", [shared_path, *sys.path]):
            from benchmark_utils.proactive_eval import parse_frame_input_sample
        scope = load_nodes("streamforest", ("parse_frame_input",), {"parse_frame_input_sample": parse_frame_input_sample})
        parsed = scope["parse_frame_input"](sample)
        self.assertEqual(parsed[2], "Watch the instructed event.")
        self.assertEqual(parsed[-1], ["a.jpg", "b.jpg"])

    def test_simplestream_actual_generation_payload_keeps_system_images_and_question(self):
        class Tensor:
            shape = (1, 7)
            def to(self, *args, **kwargs): return self
            def __eq__(self, other): return self
            def sum(self): return self
            def item(self): return 2
        tensor = Tensor()
        received = {}
        def process(messages, **kwargs):
            received["messages"], received["options"] = messages, kwargs
            return {key: tensor for key in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]}
        torch = SimpleNamespace(inference_mode=lambda: lambda function: function)
        scope = load_nodes("simplestream", ("ProactiveQAModel",), {"torch": torch,
            "_BaseQAModel": object, "DEFAULT_SYSTEM_PROMPT": "Watch the event."})
        cls = scope["ProactiveQAModel"]
        model = cls.__new__(cls)
        model.system_prompt = "Watch the event."
        model.processor = SimpleNamespace(apply_chat_template=process)
        model.image_token_id = 1
        model._get_visual_device = lambda: "visual"
        model._get_text_input_device = lambda: "text"
        model._get_visual_dtype = lambda: "dtype"
        model._generate_from_model_inputs = lambda **kwargs: "No."
        self.assertEqual(model.generate_from_frames_with_system(["frame-a", "frame-b"], "When does it start?"), "No.")
        self.assertEqual(received["messages"], [
            {"role": "system", "content": [{"type": "text", "text": "Watch the event."}]},
            {"role": "user", "content": [{"type": "image", "image": "frame-a"},
                {"type": "image", "image": "frame-b"}, {"type": "text", "text": "When does it start?"}]},
        ])
        self.assertTrue(received["options"]["add_generation_prompt"])

    def test_video_llm_actual_loop_keeps_native_frame_clock_and_raw_reply(self):
        scope = load_nodes("videollm_online", ("LiveInferForBenchmark",), {"NO_REPLY": "NO REPLY"})
        model = scope["LiveInferForBenchmark"].__new__(scope["LiveInferForBenchmark"])
        model.reset = lambda: None
        model.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: SimpleNamespace(to=lambda device: None))
        model.device = "cpu"
        model._load_and_encode_frame = lambda frame: frame
        model._feed_frame = lambda frame: frame != "b.jpg"
        model._generate_response = lambda query=None: "acknowledgement" if query else "No."
        replies = model.process_sample("system", "instruction", ["a.jpg", "b.jpg", "c.jpg"], 0.5)[1:]
        self.assertEqual([turn["time"] for turn in replies], [0.5, 1.0, 1.5])
        self.assertEqual([turn["content"] for turn in replies], ["acknowledgement", "NO REPLY", "No."])

    def test_mmduet_actual_query_queue_keeps_system_and_excludes_gold_assistant(self):
        torch = SimpleNamespace(no_grad=lambda: lambda function: function)
        scope = load_nodes("mmduet2", ("ProactiveInferenceClient",), {"torch": torch})
        cls = scope["ProactiveInferenceClient"]
        client = cls.__new__(cls)
        client.query_queue = collections.deque()
        client.system_prompt = "fallback"
        client.input_assistant_turns = False
        conversation = input_sample()["conversation"] + [{"role": "assistant", "content": "gold answer"}]
        client.input_query_stream(conversation)
        self.assertEqual(list(client.query_queue), conversation[:2])

    def test_m4_actual_main_uses_zero_based_clock_and_keeps_trigger_text(self):
        class LiveInfer:
            instances = []
            def __init__(self):
                self.frame_fps = 1
                self.seen = []
                self.__class__.instances.append(self)
            def reset(self): self.video_time = 0.0
            def load_frames_from_paths(self, paths): self.num_video_frames = len(paths)
            def input_query_stream(self, question, video_time): self.question = question
            def input_video_stream(self, timestamp):
                self.video_time = timestamp
                self.seen.append(timestamp)
            def __call__(self):
                self.current_output = "description"
                return None, "REQUIREMENT MEET AT 0.0s" if self.video_time == 1.0 else None
        module = ModuleType("local_demo.inference")
        module.LiveInfer = LiveInfer
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "input.json").write_text(json.dumps([input_sample()]))
            (base / "gold.json").write_text(json.dumps([{"question_id": "sample", "answer": []}]))
            argv = ["m4", "--test_fname", str(base / "input.json"), "--gold_fname", str(base / "gold.json"),
                    "--output_fname", str(base / "predictions.json"), "--frame_interval", "1.0"]
            with patch.dict(sys.modules, {"local_demo.inference": module}), patch.object(sys, "argv", argv), \
                    patch.object(sys, "path", [str(ROOT / "shared"), *sys.path]), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT / "native_inference/m4_omnimmi.py"), run_name="__main__")
            raw = json.loads((base / "M4/predictions_raw_outputs.jsonl").read_text())
            self.assertEqual(LiveInfer.instances[-1].seen, [0.0, 1.0])
            self.assertEqual(raw["raw_model_outputs"][1]["raw_output"], "REQUIREMENT MEET AT 0.0s")


if __name__ == "__main__":
    unittest.main()
