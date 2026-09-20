# ReverseTagger

Local Windows image-to-prompt UI built around JoyTag.

## What it does

- Runs JoyTag locally.
- Preserves every JoyTag label that passes the selected confidence threshold.
- Routes the current JoyTag vocabulary into the fixed three-line prompt layout.
- Keeps the final prompt editable.
- Provides a Rule Builder for user-defined add / replace / remove / move rules.
- Uses anime face crops as additional JoyTag evidence for character-focused recognition.

## Prompt layout

The application keeps one fixed structure:

1. declarations, camera/view, pose, environment/background, relationship
2. identity, appearance, clothing, face, expression
3. body, action, objects, interactions and any other routed labels

ReverseTagger does not apply a content filter.

## Rules

The **Rule Builder** is optional and starts with an empty `config/rules.yaml`.
Rules are applied after JoyTag detection and semantic routing. They can:

- add a user-specified tag;
- replace a matching tag;
- remove a matching tag;
- move a tag to another prompt section or position.

Rules are user configuration, not a built-in vocabulary or automatic tag list.

## JoyTag vocabulary and character names

JoyTag's model output is fixed by the model checkpoint. Its downloaded `labels.txt` maps output indices to names. Editing that file alone does **not** teach JoyTag a new concept and does not add a new model output.

For example, adding `9S` to `labels.txt` would only rename an existing output index (and could make the mapping incorrect); it would not make the model detect 9S. A character absent from the JoyTag checkpoint requires a separate recognition model or another source of image evidence.

`config/semantic_tags.yaml` is routing metadata for the vocabulary used by this project. It contains each snapshot label exactly once. Unknown labels from a newer JoyTag vocabulary are still preserved by the semantic layer and fall back to the third prompt line until the routing table is updated.

## Model files

The official JoyTag files are downloaded on first run into `models/joytag/`. Model weights are not included in the repository.

The anime face detector resource is also downloaded on first use into `models/face_detector/`.

## Installation

Run:

```bat
run.bat
```

The launcher creates the Python 3.10 virtual environment, installs the pinned CUDA PyTorch build and runtime dependencies, verifies CUDA, and starts Gradio on the first free port from 7860 upward.

## Project structure

```text
ReverseTagger/
├── app.py
├── run.bat
├── check_env.py
├── requirements.txt
├── README.md
├── config/
│   ├── rules.yaml
│   ├── rules.yaml.example
│   └── semantic_tags.yaml
├── core/
│   ├── analyzer.py
│   ├── character_detector.py
│   ├── joytag_engine.py
│   ├── prompt_builder.py
│   ├── rule_engine.py
│   └── semantic_layer.py
└── joytag/
    ├── Models.py
    ├── LICENSE
    └── README.md
```

Runtime files under `models/`, Python bytecode and the local virtual environment are ignored by Git.

## Privacy

Image inference is local. The first run may contact Hugging Face to download JoyTag and the face detector resource. Once those files exist locally, inference runs locally.
