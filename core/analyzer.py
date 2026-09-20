from __future__ import annotations

from dataclasses import dataclass

from core.joytag_engine import JoyTagEngine
from core.semantic_layer import SemanticLayer
from core.prompt_builder import PromptBuilder
from core.rule_engine import RuleEngine
from core.character_detector import FaceDetector


@dataclass
class AnalysisResult:
    detections: list
    semantic_tags: list
    rule_result: dict
    prompt: str
    prompt_groups: dict


class Analyzer:
    """Image -> JoyTag evidence -> optional face-crop evidence -> three-line prompt.

    JoyTag is the source of truth for the vocabulary. This class contains no
    character/franchise vocabulary and does not add or remove semantic tags.
    """

    def __init__(self, device="cuda", threshold=0.4):
        self.joytag = JoyTagEngine(device=device, threshold=threshold)
        self.semantic = SemanticLayer()
        self.rules = RuleEngine()
        self.builder = PromptBuilder()
        self.face_detector = FaceDetector()

    def analyze(self, image):
        threshold = float(self.joytag.threshold)

        # Full image is authoritative for scene-level declarations/background/actions.
        detections = self.joytag.predict(image, threshold=threshold)

        # Character crops are only an additional observation channel. They do not have
        # their own vocabulary or rules and never replace the full-image predictions.
        crop_detections = self._character_crop_evidence(image, detections, threshold)
        merged = self._merge_detections(detections, crop_detections)

        semantic_tags = self.semantic.analyze(merged)
        rule_result = self.rules.apply(semantic_tags)
        prompt = self.builder.build(rule_result)
        prompt_groups = self.builder.build_groups(rule_result)
        return AnalysisResult(merged, semantic_tags, rule_result, prompt, prompt_groups)

    def _character_crop_evidence(self, image, scene_detections, threshold):
        try:
            expected = self._expected_character_count(scene_detections)
            faces = self.face_detector.detect(image, expected_faces=expected)
            if not faces:
                print("[CharacterDetector] No readable faces; using full-image JoyTag only.", flush=True)
                return []

            all_crop = []
            for index, face in enumerate(faces, 1):
                views = self.face_detector.character_views(image, face, None, faces)
                # Identity is often easiest from the head/upper view, while body/action
                # labels can require the full character view. We let JoyTag decide.
                for view_name, view in views.items():
                    predictions = self.joytag.predict(view, threshold=threshold)
                    for item in predictions:
                        item = dict(item)
                        item["source"] = f"joytag_face_{index}:{view_name}"
                        all_crop.append(item)
            print(f"[CharacterDetector] Added {len(all_crop)} crop detections.", flush=True)
            return all_crop
        except Exception as exc:
            print(f"[CharacterDetector] Crop evidence skipped: {exc}", flush=True)
            return []

    @staticmethod
    def _expected_character_count(detections):
        # Read the model's own declaration label. No fixed maximum is imposed here.
        for item in detections:
            tag = str(item.get("tag", ""))
            import re
            m = re.fullmatch(r"(\d+)(?:girl|girls|boy|boys|futa|futas)", tag)
            if m:
                return int(m.group(1))
        return None

    @staticmethod
    def _merge_detections(scene, crops):
        # Preserve every unique model label. If a crop rediscovers a label already
        # present, keep the strongest confidence instead of duplicating the prompt.
        best = {}
        for item in list(scene) + list(crops):
            tag = str(item.get("tag", "")).strip()
            if not tag:
                continue
            score = float(item.get("confidence", 0.0))
            current = best.get(tag)
            if current is None or score > float(current.get("confidence", 0.0)):
                best[tag] = dict(item)
        result = list(best.values())
        result.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return result
