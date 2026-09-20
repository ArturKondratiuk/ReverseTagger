from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEMANTIC_FILE = PROJECT_ROOT / "config" / "semantic_tags.yaml"

PROMPT_CATEGORIES = (
    "character", "camera", "pose", "environment", "relationship",
    "identity", "appearance", "clothing", "face", "expression",
    "body", "action", "object", "interaction",
)


@dataclass
class SemanticTag:
    """A JoyTag prediction plus its prompt-routing category."""

    tag: str
    confidence: float
    source: str
    category: str
    type: str = "model_tag"
    value: str = ""
    prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticLayer:
    """Routes the complete JoyTag vocabulary into the three-line prompt.

    ``semantic_tags.yaml`` is generated from the actual JoyTag ``labels.txt`` and
    contains every label known to the installed model. It is a routing table, not
    a filter: labels are never deleted or replaced. If a future JoyTag version
    contains a label that is not in the generated table, it is preserved and sent
    to line 3 as ``action`` until the table is regenerated.
    """

    def __init__(self, semantic_file: Path = SEMANTIC_FILE):
        self.semantic_file = Path(semantic_file)
        self._routes = self._load_routes(self.semantic_file)
        print(f"[SemanticLayer] Loaded {len(self._routes)} label routes.", flush=True)

    @staticmethod
    def _load_routes(path: Path) -> dict[str, str]:
        if not path.exists():
            raise FileNotFoundError(
                f"Semantic vocabulary file not found: {path}. "
                "It must be generated from JoyTag labels.txt."
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        routes: dict[str, str] = {}
        for category in PROMPT_CATEGORIES:
            values = data.get(category, []) or []
            if not isinstance(values, list):
                raise ValueError(f"Semantic category '{category}' must be a list.")
            for tag in values:
                tag = str(tag).strip()
                if not tag:
                    continue
                if tag in routes and routes[tag] != category:
                    raise ValueError(f"Tag appears in multiple semantic categories: {tag}")
                routes[tag] = category
        return routes

    def analyze(self, detections: list[dict[str, Any]]) -> list[SemanticTag]:
        result: list[SemanticTag] = []
        for detection in detections:
            tag = str(detection.get("tag", "")).strip()
            if not tag:
                continue
            confidence = float(detection.get("confidence", 1.0))
            source = str(detection.get("source", "joytag"))
            category = self.route(tag)
            result.append(
                SemanticTag(
                    tag=tag,
                    confidence=confidence,
                    source=source,
                    category=category,
                    value=tag,
                    prompt=tag.replace("_", " ").strip(),
                )
            )
        return result

    def route(self, tag: str) -> str:
        """Return the configured category without changing the model label."""
        return self._routes.get(tag, "action")
