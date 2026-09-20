from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core.semantic_layer import SemanticTag

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = PROJECT_ROOT / "config" / "rules.yaml"


class RuleEngine:
    """Apply user-defined prompt rules without a built-in tag vocabulary."""

    def __init__(self, rules_path: Path = RULES_PATH):
        self.rules_path = Path(rules_path)
        self.rules: list[dict[str, Any]] = []
        self.load_rules()

    def load_rules(self) -> None:
        if not self.rules_path.exists():
            self.rules = []
            return
        data = yaml.safe_load(self.rules_path.read_text(encoding="utf-8")) or {}
        loaded = data.get("rules", []) if isinstance(data, dict) else []
        self.rules = loaded if isinstance(loaded, list) else []
        print(f"[Rules] Loaded {len(self.rules)} rules", flush=True)

    def apply(self, semantic_tags: list[SemanticTag]) -> dict[str, Any]:
        tags = list(semantic_tags)
        components: list[dict[str, Any]] = []
        for rule in self.rules:
            condition = rule.get("if", {}) if isinstance(rule, dict) else {}
            if self._matches(condition, tags):
                self._actions(rule.get("then", {}), tags, components)
        return {"tags": tags, "components": components}

    def _matches(self, condition: Any, tags: list[SemanticTag]) -> bool:
        if not isinstance(condition, dict):
            return False
        if not all(self._tag_exists(item, tags) for item in condition.get("all", [])):
            return False
        if condition.get("any") and not any(self._tag_exists(item, tags) for item in condition["any"]):
            return False
        if condition.get("none") and any(self._tag_exists(item, tags) for item in condition["none"]):
            return False
        return True

    @staticmethod
    def _tag_exists(condition: Any, tags: list[SemanticTag]) -> bool:
        if isinstance(condition, str):
            return any(t.tag == condition for t in tags)
        if not isinstance(condition, dict):
            return False
        for key in ("tag", "category", "type", "value"):
            if key in condition and not any(getattr(t, key, None) == condition[key] for t in tags):
                return False
        return True

    def _actions(self, actions: Any, tags: list[SemanticTag], components: list[dict[str, Any]]) -> None:
        if not isinstance(actions, dict):
            return
        for item in actions.get("add", []):
            self._add(item, tags)
        for name in actions.get("remove", []):
            tags[:] = [t for t in tags if t.tag != str(name)]
        for old, new in (actions.get("replace", {}) or {}).items():
            self._replace(str(old), new, tags)
        for item in actions.get("move", []):
            self._move(item, tags)
        for item in actions.get("component", []):
            self._component(item, components)

    @staticmethod
    def _parse_generated(item: Any) -> tuple[str, str]:
        if isinstance(item, dict):
            return str(item.get("tag", "")).strip(), str(item.get("category", "action")).strip() or "action"
        return str(item).strip(), "action"

    def _add(self, item: Any, tags: list[SemanticTag]) -> None:
        name, category = self._parse_generated(item)
        if not name or any(t.tag == name for t in tags):
            return
        tags.append(SemanticTag(
            tag=name, confidence=1.0, source="rule", category=category,
            type="generated", value=name, prompt=name,
            metadata={"generated_by_rule": True},
        ))

    def _replace(self, old: str, new: Any, tags: list[SemanticTag]) -> None:
        if not any(t.tag == old for t in tags):
            return
        tags[:] = [t for t in tags if t.tag != old]
        values = new if isinstance(new, list) else [new]
        for item in values:
            self._add(item, tags)

    @staticmethod
    def _move(item: Any, tags: list[SemanticTag]) -> None:
        if isinstance(item, str):
            name, category, position, anchor = item, None, "end", None
        elif isinstance(item, dict):
            name = str(item.get("tag", "")).strip()
            category = str(item.get("category", "")).strip() or None
            position = str(item.get("position", "end")).strip().lower()
            anchor = str(item.get("anchor", "")).strip() or None
        else:
            return
        target = next((t for t in tags if t.tag == name), None)
        if target is None:
            return
        target.metadata.setdefault("rule_position", {})
        target.metadata["rule_position"] = {
            "category": category,
            "position": position,
            "anchor": anchor,
        }

    @staticmethod
    def _component(item: Any, components: list[dict[str, Any]]) -> None:
        if isinstance(item, str) and item.strip():
            components.append({"prompt": item.strip(), "category": "action", "source": "rule"})
        elif isinstance(item, dict) and str(item.get("prompt", "")).strip():
            components.append({
                "prompt": str(item["prompt"]).strip(),
                "category": str(item.get("category", "action")),
                "source": "rule",
            })
