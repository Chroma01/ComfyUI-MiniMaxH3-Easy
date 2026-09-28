"""Exercise context media rules without importing ComfyUI or loading models.

The functions under test are compiled directly from nodes.py. Only external
model/VAE calls and unrelated frame-sizing helpers are replaced with mocks.
Run with: python -m unittest discover -s tests -v
"""

import ast
import copy
import hashlib
import json
import math
import re
import unittest
from collections import namedtuple
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / "nodes.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
CONSTANTS = {
    "MAX_MEDIA", "MAX_IMAGES", "MAX_VIDEOS", "MAX_AUDIOS",
    "MIN_SECONDS", "MAX_SECONDS", "SEGMENT_MAX_COUNT",
    "SEGMENT_MAX_MEDIA", "SEGMENT_MAX_IMAGES", "SEGMENT_MAX_VIDEOS", "SEGMENT_MAX_AUDIOS",
    "SEGMENT_DIVIDER_PATTERN", "SEGMENT_DIVIDER_INVISIBLE", "SEGMENT_TAG_PATTERN",
    "REFERENCE_PLACEHOLDER_RE", "UNRESOLVED_REFERENCE_RE",
    "MODE_IMAGE", "MODE_REFERENCE", "MODE_DIGITAL_HUMAN",
    "CONTEXT_AUDIO_GENERATED", "CONTEXT_AUDIO_DIGITAL_HUMAN",
    "CONTEXT_CONTINUITY_GUIDE", "CONTEXT_CONTINUITY_LATENT",
    "CONTEXT_CONTINUITY_SOFT_AV", "CONTEXT_CONTINUITY_HARD_AV", "CONTEXT_CONTINUITY_MODES",
    "PROMPT_OPTIMIZER_LANGUAGE_EN", "PROMPT_OPTIMIZER_LANGUAGE_ZH",
    "PROMPT_OPTIMIZER_MARKER_VERSION", "PROMPT_OPTIMIZER_ON_RUN_TIMEOUT_SECONDS",
    "PROMPT_OPTIMIZER_MAX_OUTPUT_TOKENS", "CONTEXT_PROMPT_OPTIMIZER_MAX_OUTPUT_TOKENS",
    "CONTEXT_PROMPT_OPTIMIZER_DEFAULT_CONCURRENCY", "CONTEXT_PROMPT_OPTIMIZER_MAX_CONCURRENCY",
    "CONTEXT_PROMPT_OPTIMIZER_WHOLE", "CONTEXT_PROMPT_OPTIMIZER_PER_SEGMENT",
}
FUNCTIONS = {
    "_normalize_optimizer_language", "_optimizer_segment_duration_values",
    "_optimizer_segment_duration_plan", "_optimizer_segment_rules",
    "_optimizer_single_segment_rules", "_optimizer_digital_human_rules",
    "split_prompt_segments", "parse_segment_seconds",
    "_segment_expand_media_placeholders", "bind_segment_media",
    "_validate_reference_media", "_reference_conditioning",
    "_runtime_optimizer_resources", "_runtime_optimizer_marker",
    "_runtime_optimizer_prompt", "_optimizer_sha256", "_optimize_prompt_on_run",
    "_normalize_optimized_segments", "_normalize_optimized_segments_with_retry",
}


def load_contract():
    body = ast.parse("from __future__ import annotations").body
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & CONSTANTS:
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3EasyContextSegments":
            method = next(part for part in node.body if getattr(part, "name", "") == "_prepare_segments")
            method = copy.deepcopy(method)
            method.decorator_list = []
            body.append(method)
    namespace = {
        "re": re, "math": math, "json": json, "hashlib": hashlib, "Mapping": Mapping,
        "MiniMaxH3Context": SimpleNamespace,
        "_RuntimePromptOptimization": namedtuple("Optimization", "prompt marker", defaults=[None]),
        "h3": SimpleNamespace(FPS=24, _empty_av_latent=Mock(return_value=({"samples": "av"}, 124))),
        "_segment_context_frame_count_for_mode": Mock(return_value=5),
        "_motion_context_output_frame_length": Mock(return_value=124),
        "_frame_length": Mock(return_value=124),
        "_encode_reference_audio": Mock(return_value=("encoded-audio", 40)),
        "_resolve_reference_prompt": Mock(side_effect=lambda prompt, *_args: prompt),
        "node_helpers": SimpleNamespace(conditioning_set_values=Mock(
            side_effect=lambda conditioning, values: [
                [embedding, {**metadata, **values}] for embedding, metadata in conditioning
            ]
        )),
    }
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


def media(index, kind, value=None):
    return SimpleNamespace(input_index=index, media_type=kind, value=value)


class ContextMediaContractTests(unittest.TestCase):
    def setUp(self):
        self.code = load_contract()
        self.bundle = SimpleNamespace(
            model_for=Mock(return_value="ref-model"),
            video_vae=object(),
            audio_vae=object(),
            clip=SimpleNamespace(
                tokenize=Mock(return_value="tokens"),
                encode_from_tokens_scheduled=Mock(return_value=[["embedding", {}]]),
            ),
        )

    def prepare(self, prompt, items, source_audio=None, audio_mode="generated"):
        return self.code["_prepare_segments"](
            self.bundle, prompt, items, source_audio, audio_mode,
            640, 384, "16:9", "5,5", 5, "latent_guide", "1k",
        )[1]

    def test_chinese_whole_sequence_requires_character_references_in_each_block(self):
        for source in ("女孩走进花园，然后坐下。", "女孩走进花园。\n---\n女孩坐下。"):
            with self.subTest(source=source):
                rules = self.code["_optimizer_segment_rules"](2, "5,5", source, "zh")
                for required in (
                    "不能继承前段的引用", "每个出场分段都必须重复对应引用",
                    "人物名字、<Subject N>、代词或外观描述不能代替素材引用",
                    "每段对同一素材明确引用一次即可",
                    "不适用于当前段必需的素材引用", "真正没有使用的媒体不要强行加入",
                    "不要重编号", "严格返回 2 段提示词",
                ):
                    self.assertIn(required, rules)

    def test_chinese_per_segment_optimizer_also_requires_local_character_reference(self):
        rules = self.code["_optimizer_single_segment_rules"](1, 2, 5, "zh")
        self.assertIn("不能继承前段的引用", rules)
        self.assertIn("即使前段已经引用过也必须重复", rules)
        self.assertIn("当前段未使用的素材不要强行加入", rules)
        self.assertIn("只优化用户消息中当前这一段", rules)

    def test_english_reference_and_duration_rules_remain_available(self):
        rules = self.code["_optimizer_segment_rules"](2, "5,5", "A girl walks and sits.", "en")
        self.assertIn("repeat its reference in every block that depends on it", rules)
        self.assertIn("must never rely on an earlier block's reference being inherited", rules)
        self.assertIn("block 2: local time 0.00-5 seconds", rules)

    def test_digital_human_rule_still_overrides_reference_audio_tagging(self):
        rules = self.code["_optimizer_digital_human_rules"]()
        self.assertIn("do not add <Audio N> tags", rules)
        self.assertIn("repeat a visual reference in each block", rules)
        self.assertIn("overrides any generic instruction", rules)

    def test_audio_only_reference_passes_media_validation(self):
        self.code["_validate_reference_media"]([media(1, "audio")], "Context Segment 2")

    def test_mixed_and_visual_reference_validation_is_unchanged(self):
        for items in ([media(1, "image")], [media(1, "video")],
                      [media(1, "image"), media(2, "audio")]):
            with self.subTest(items=items):
                self.code["_validate_reference_media"](items)

    def test_invalid_and_over_budget_media_are_still_rejected(self):
        cases = [
            ([], "at least one media resource"),
            ([media(1, "unknown")], "unsupported media resource"),
            ([media(i, "audio") for i in range(4)], "media limits"),
            ([media(i, "image") for i in range(10)], "media limits"),
            ([media(i, "video") for i in range(4)], "media limits"),
            ([media(i, "image") for i in range(16)], "at most fifteen"),
        ]
        for items, message in cases:
            with self.subTest(message=message, count=len(items)):
                with self.assertRaisesRegex(ValueError, message):
                    self.code["_validate_reference_media"](items)

    def test_audio_only_conditioning_keeps_audio_payload_and_tag(self):
        audio = {"waveform": "waveform", "sample_rate": 32000}
        conditioning, latent = self.code["_reference_conditioning"](
            self.bundle, "声音来自 <Audio 1>", 640, 384, 124, "1k", [media(2, "audio", audio)],
        )
        self.code["_encode_reference_audio"].assert_called_once_with(self.bundle.audio_vae, audio)
        self.bundle.clip.tokenize.assert_called_once_with(
            "声音来自 <Audio 1>", minimax_ref_items=[{"type": "audio"}],
        )
        self.assertEqual(conditioning[0][1]["minimax_refs"], [
            {"kind": "audio", "ref_audio_t": 40, "audio_latent": "encoded-audio"},
        ])
        self.assertEqual(latent, {"samples": "av"})
        self.assertEqual(self.code["_resolve_reference_prompt"].call_args.args[1], {2: "<Audio 1>"})

    def test_malformed_audio_and_empty_reference_conditioning_are_still_rejected(self):
        for items, message in (
            ([media(1, "audio", {})], "must be AUDIO payloads"),
            ([], "at least one media resource"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.code["_reference_conditioning"](
                        self.bundle, "", 640, 384, 124, "1k", items,
                    )

    def test_second_segment_can_keep_audio_without_visual_reference(self):
        image, audio = media(1, "image"), media(2, "audio")
        context = self.prepare(
            "<Picture 1> 的女孩，声音来自 <Audio 1>\n---\n女孩坐下，声音来自 <Audio 1>",
            [image, audio],
        )
        self.assertEqual(context.segment_plan["shots"][0]["media"], [image, audio])
        self.assertEqual(context.segment_plan["shots"][1]["media"], [audio])
        self.assertEqual(context.segment_plan["model_role"], "ref2va")

    def test_text_only_segments_still_generate_without_automatically_attaching_media(self):
        context = self.prepare("女孩走进花园。\n---\n女孩坐下。", [media(1, "image")])
        self.assertTrue(all(not shot["media"] for shot in context.segment_plan["shots"]))
        self.assertEqual(context.segment_plan["model_role"], "fl2va")

    def test_digital_human_driver_remains_global_without_per_segment_audio_tags(self):
        driver = {"waveform": "driver", "sample_rate": 32000}
        context = self.prepare(
            "女孩讲话。\n---\n女孩继续讲话。", [media(1, "audio", driver)],
            driver, "digital_human",
        )
        self.assertIs(context.source_audio, driver)
        self.assertIs(context.segment_plan["source_audio"], driver)
        self.assertEqual(context.segment_plan["model_role"], "ref2va")
        self.assertTrue(all(not shot["media"] for shot in context.segment_plan["shots"]))

    def test_digital_human_keeps_visual_references_but_removes_driver_audio_tags(self):
        driver = {"waveform": "driver", "sample_rate": 32000}
        image, audio = media(1, "image"), media(2, "audio", driver)
        context = self.prepare(
            "<Picture 1> 的女孩讲话 <Audio 1>\n---\n女孩继续讲话 <Audio 1>",
            [image, audio], driver, "digital_human",
        )
        self.assertEqual(context.segment_plan["shots"][0]["media"], [image])
        self.assertEqual(context.segment_plan["shots"][1]["media"], [])
        self.assertTrue(all("<Audio" not in shot["prompt"] for shot in context.segment_plan["shots"]))
        self.assertIs(context.segment_plan["source_audio"], driver)

    def test_one_idea_runtime_request_contains_strengthened_chinese_rules(self):
        source = "女孩走进花园，然后坐下。"
        settings = {
            "optimize_on_run": True, "language": "zh", "api_url": "test://optimizer",
            "api_key": "test", "model": "test", "read_media": False,
        }
        self.code.update({
            "_read_prompt_optimizer_config": lambda: settings,
            "_configured_prompt_scene_guide": lambda _settings, scene: scene,
            "_prompt_optimizer_settings_complete": lambda *_args: True,
            "_runtime_optimizer_context_hash": lambda *_args, **_kwargs: "context-hash",
            "_optimizer_system_prompt": lambda *_args, **_kwargs: "base guide",
            "_optimizer_media_read_allowed": lambda _resources: True,
            "_optimizer_http_json": Mock(return_value="女孩走进花园。\n---\n女孩坐下。"),
        })
        old_marker = {
            "version": 9,
            "prompt_sha256": self.code["_optimizer_sha256"](source),
            "context_sha256": "context-hash",
        }
        result = self.code["_optimize_prompt_on_run"](
            source, "reference", 5, "none", [], [], old_marker, False,
            segment_spec=(2, "5,5"),
        )
        request = self.code["_optimizer_http_json"].call_args
        self.assertIn("每个出场分段都必须重复对应引用", request.args[4])
        self.assertEqual(request.args[5], source)
        self.assertEqual(len(self.code["split_prompt_segments"](result.prompt)), 2)
        self.assertEqual(result.marker["version"], self.code["PROMPT_OPTIMIZER_MARKER_VERSION"])
        self.assertNotEqual(result.marker["version"], old_marker["version"])


if __name__ == "__main__":
    unittest.main()
