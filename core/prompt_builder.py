from __future__ import annotations

from collections import OrderedDict

PROMPT_ORDER = [
    "character", "camera", "pose", "environment", "relationship",
    "identity", "appearance", "clothing", "face", "expression",
    "body", "action", "object", "interaction",
]

LINE_GROUPS = (
    ("character", "camera", "pose", "environment", "relationship"),
    ("identity", "appearance", "clothing", "face", "expression"),
    ("body", "action", "object", "interaction"),
)


class PromptBuilder:
    """Build the fixed three-line prompt while preserving every supplied tag."""

    def build(self, rule_result) -> str:
        groups = self.build_groups(rule_result)
        lines = [self._join(groups, names) for names in LINE_GROUPS]
        components = rule_result.get("components", []) if isinstance(rule_result, dict) else []
        if components:
            groups = self._copy_groups(groups)
            for item in components:
                category = item.get("category", "action")
                groups.setdefault(category, []).append(str(item.get("prompt", "")).strip())
            lines = [self._join(groups, names) for names in LINE_GROUPS]
        return "\n".join(lines)

    def build_groups(self, rule_result):
        tags = rule_result.get("tags", []) if isinstance(rule_result, dict) else rule_result
        groups = OrderedDict((name, []) for name in PROMPT_ORDER)
        seen = set()
        for item in tags:
            tag = getattr(item, "tag", "").strip()
            if not tag:
                continue
            prompt = (getattr(item, "prompt", None) or tag).strip()
            key = prompt.lower()
            if key in seen:
                continue
            seen.add(key)
            category = getattr(item, "category", "action")
            placement = getattr(item, "metadata", {}).get("rule_position", {})
            target_category = placement.get("category") or category
            if target_category not in groups:
                target_category = "action"
            groups[target_category].append(prompt)
        self._apply_positions(groups, tags)
        return groups

    @staticmethod
    def _copy_groups(groups):
        return OrderedDict((k, list(v)) for k, v in groups.items())

    @staticmethod
    def _apply_positions(groups, tags) -> None:
        for item in tags:
            placement = getattr(item, "metadata", {}).get("rule_position", {})
            if not placement:
                continue
            name = (getattr(item, "prompt", None) or getattr(item, "tag", "")).strip()
            category = placement.get("category") or getattr(item, "category", "action")
            if category not in groups or name not in groups[category]:
                continue
            values = groups[category]
            values.remove(name)
            position = placement.get("position", "end")
            anchor = placement.get("anchor")
            if position == "start":
                values.insert(0, name)
            elif position in {"before", "after"} and anchor in values:
                index = values.index(anchor) + (1 if position == "after" else 0)
                values.insert(index, name)
            else:
                values.append(name)

    @staticmethod
    def _join(groups, names):
        values = []
        for name in names:
            values.extend(groups.get(name, []))
        return ", ".join(values)
