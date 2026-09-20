import inspect
import socket
import warnings
from pathlib import Path

import gradio as gr
import torch
import yaml

warnings.filterwarnings("ignore", message=r".*torch\.backends\.cuda\.sdp_kernel\(\) is deprecated.*", category=FutureWarning)

from core.analyzer import Analyzer

PROJECT_ROOT = Path(__file__).resolve().parent
RULES_PATH = PROJECT_ROOT / "config" / "rules.yaml"

THRESHOLDS = {
    "Strict / confident": 0.55,
    "Balanced": 0.40,
    "Sensitive / uncertain": 0.25,
}
CATEGORIES = [
    "character", "camera", "pose", "environment", "relationship",
    "identity", "appearance", "clothing", "face", "expression",
    "body", "action", "object", "interaction",
]

_analyzer = None


def patch_starlette_template_response():
    try:
        from starlette.templating import Jinja2Templates

        original = Jinja2Templates.TemplateResponse
        if getattr(original, "_reversetagger_compat", False):
            return
        params = list(inspect.signature(original).parameters.values())
        if len(params) < 2 or params[1].name != "request":
            return

        def compat(self, *args, **kwargs):
            if args and isinstance(args[0], str):
                name = args[0]
                context = args[1] if len(args) > 1 else kwargs.get("context")
                if isinstance(context, dict) and "request" in context:
                    request = context["request"]
                    forwarded = dict(kwargs)
                    forwarded.pop("context", None)
                    return original(self, request, name, context, *args[2:], **forwarded)
            return original(self, *args, **kwargs)

        compat._reversetagger_compat = True
        Jinja2Templates.TemplateResponse = compat
    except Exception:
        pass


patch_starlette_template_response()


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _analyzer = Analyzer(device=device, threshold=0.40)
    return _analyzer


def analyze(image, mode):
    if image is None:
        return "", "Upload an image first."
    try:
        engine = get_analyzer()
        threshold = THRESHOLDS[mode]
        engine.joytag.threshold = threshold
        result = engine.analyze(image)
        return result.prompt, f"Ready — {len(result.detections)} JoyTag labels, threshold {threshold:.2f}."
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return "", f"Error: {type(exc).__name__}: {exc}"


def load_rules():
    if not RULES_PATH.exists():
        return {"rules": []}
    try:
        data = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"rules": []}
    return data if isinstance(data, dict) and isinstance(data.get("rules"), list) else {"rules": []}


def save_rules(data):
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RULES_PATH.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if _analyzer is not None:
        _analyzer.rules.load_rules()


def rules_text():
    data = load_rules()
    if not data["rules"]:
        return "No rules saved yet."
    return "```yaml\n" + yaml.safe_dump(data, allow_unicode=True, sort_keys=False) + "```"


def rule_choices():
    choices = []
    for i, rule in enumerate(load_rules()["rules"]):
        triggers = ", ".join(rule.get("if", {}).get("all", []))
        actions = rule.get("then", {})
        action = next((name.replace("_", " ").title() for name in ("add", "replace", "remove", "move", "component") if name in actions), "Keep")
        choices.append((f"{i + 1}. {triggers} — {action}", str(i)))
    return choices


def empty_dropdown():
    return gr.Dropdown(choices=rule_choices(), value=None)


def add_rule(trigger, action, output, category, move_tag, move_position, move_anchor):
    trigger = (trigger or "").strip()
    output = (output or "").strip()
    move_tag = (move_tag or "").strip()
    move_anchor = (move_anchor or "").strip()

    if not trigger:
        return rules_text(), "Enter at least one trigger tag.", empty_dropdown()

    triggers = [x.strip() for x in trigger.split(",") if x.strip()]
    rule = {"if": {"all": triggers}, "then": {}}

    if action == "Keep + Add":
        if not output:
            return rules_text(), "Enter the additional tags.", empty_dropdown()
        rule["then"]["add"] = [{"tag": x.strip(), "category": category} for x in output.split(",") if x.strip()]
    elif action == "Replace":
        if not output:
            return rules_text(), "Enter the replacement tags.", empty_dropdown()
        rule["then"]["replace"] = {
            triggers[0]: [{"tag": x.strip(), "category": category} for x in output.split(",") if x.strip()]
        }
    elif action == "Remove":
        rule["then"]["remove"] = triggers
    elif action == "Move":
        if not move_tag:
            return rules_text(), "Enter the tag to move.", empty_dropdown()
        item = {"tag": move_tag, "category": category, "position": move_position}
        if move_position in {"before", "after"}:
            if not move_anchor:
                return rules_text(), "Enter an anchor tag.", empty_dropdown()
            item["anchor"] = move_anchor
        rule["then"]["move"] = [item]

    data = load_rules()
    data["rules"].append(rule)
    save_rules(data)
    return rules_text(), "Rule saved.", empty_dropdown()


def delete_rule(index):
    try:
        index = int(index)
    except (TypeError, ValueError):
        return rules_text(), "Select a rule.", empty_dropdown()
    data = load_rules()
    if index < 0 or index >= len(data["rules"]):
        return rules_text(), "Rule not found.", empty_dropdown()
    data["rules"].pop(index)
    save_rules(data)
    return rules_text(), "Rule deleted.", empty_dropdown()


def action_ui(action):
    needs_output = action in {"Keep + Add", "Replace"}
    needs_move = action == "Move"
    label = "Additional tags" if action == "Keep + Add" else "Replacement tags"
    return (
        gr.Textbox(label=label, placeholder="tag one, tag two", visible=needs_output),
        gr.Dropdown(label="Target section", choices=CATEGORIES, value="body", visible=needs_output or needs_move),
        gr.Textbox(label="Tag to move", placeholder="tag to move", visible=needs_move),
        gr.Dropdown(label="Position", choices=["start", "end", "before", "after"], value="end", visible=needs_move),
        gr.Textbox(label="Anchor tag", placeholder="anchor tag", visible=False),
    )


def position_ui(position):
    return gr.Textbox(visible=position in {"before", "after"})


CSS = """
#prompt textarea { min-height: 320px !important; font-family: Consolas, monospace !important; font-size: 15px !important; }
"""

_BLOCKS_KWARGS = {"title": "ReverseTagger"}
_LAUNCH_KWARGS = {"theme": gr.themes.Soft(), "css": CSS}
try:
    launch_params = inspect.signature(gr.Blocks.launch).parameters
    if "theme" not in launch_params:
        _BLOCKS_KWARGS["theme"] = _LAUNCH_KWARGS.pop("theme")
    if "css" not in launch_params:
        _BLOCKS_KWARGS["css"] = _LAUNCH_KWARGS.pop("css")
except Exception:
    pass

with gr.Blocks(**_BLOCKS_KWARGS) as demo:
    with gr.Tabs():
        with gr.Tab("Analyzer"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(label="Input image", type="pil", height=520)
                    mode = gr.Radio(list(THRESHOLDS), value="Balanced", label="Recognition mode")
                    analyze_btn = gr.Button("ANALYZE", variant="primary", size="lg")
                with gr.Column():
                    prompt = gr.Textbox(label="FINAL PROMPT (editable)", lines=18, elem_id="prompt", interactive=True)
                    status = gr.Markdown("Model loads on first analysis.")

        with gr.Tab("Rule Builder"):
            gr.Markdown("## Rule Builder")
            gr.Markdown("Rules are applied after JoyTag/semantic routing. Matching tags are kept unless an action changes them.")
            with gr.Row():
                trigger = gr.Textbox(label="When JoyTag finds", placeholder="1boy, holding", scale=2)
                action = gr.Radio(["Keep + Add", "Replace", "Remove", "Move"], value="Keep + Add", label="Action", scale=1)
            with gr.Row():
                output = gr.Textbox(label="Additional tags", placeholder="tag one, tag two", scale=2)
                category = gr.Dropdown(label="Target section", choices=CATEGORIES, value="body", scale=1)
            with gr.Row():
                move_tag = gr.Textbox(label="Tag to move", placeholder="tag to move", visible=False)
                move_position = gr.Dropdown(label="Position", choices=["start", "end", "before", "after"], value="end", visible=False)
                move_anchor = gr.Textbox(label="Anchor tag", placeholder="anchor tag", visible=False)
            with gr.Row():
                save_btn = gr.Button("SAVE RULE", variant="primary")
                refresh_btn = gr.Button("REFRESH RULES")
            rule_status = gr.Markdown()
            rules_view = gr.Markdown(rules_text())
            with gr.Row():
                delete_select = gr.Dropdown(choices=rule_choices(), label="Rule to delete", interactive=True, scale=4)
                delete_btn = gr.Button("DELETE RULE", variant="stop")

            action.change(action_ui, action, [output, category, move_tag, move_position, move_anchor])
            move_position.change(position_ui, move_position, move_anchor)
            save_btn.click(add_rule, [trigger, action, output, category, move_tag, move_position, move_anchor], [rules_view, rule_status, delete_select])
            refresh_btn.click(lambda: (rules_text(), empty_dropdown()), None, [rules_view, delete_select])
            delete_btn.click(delete_rule, delete_select, [rules_view, rule_status, delete_select])

    analyze_btn.click(analyze, [image, mode], [prompt, status])


def find_free_port(start=7860, end=7960):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass
    raise OSError(f"Cannot find an empty port in range {start}-{end}")


if __name__ == "__main__":
    port = find_free_port()
    print(f"Starting ReverseTagger on http://127.0.0.1:{port}")
    demo.launch(server_name="127.0.0.1", server_port=port, **_LAUNCH_KWARGS)
