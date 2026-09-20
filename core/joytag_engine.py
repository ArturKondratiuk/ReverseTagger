from pathlib import Path
import sys
from typing import Any

import torch
from PIL import Image
import torchvision.transforms.functional as TVF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOYTAG_SOURCE = PROJECT_ROOT / "joytag"
JOYTAG_MODEL = PROJECT_ROOT / "models" / "joytag"
JOYTAG_REPO = "fancyfeast/joytag"
LABEL_FILE = "labels.txt"

sys.path.insert(0, str(JOYTAG_SOURCE))
from Models import VisionModel


class JoyTagEngine:
    """Thin JoyTag inference wrapper with first-run model provisioning."""

    def __init__(self, model_path: Path = JOYTAG_MODEL, device: str = "cuda", threshold: float = 0.4):
        self.model_path = Path(model_path)
        self.device = device
        self.threshold = threshold
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Check your PyTorch installation.")

        self._ensure_model_files()
        print(f"[JoyTag] Loading model from: {self.model_path}")
        print(f"[JoyTag] Device: {self.device}")
        self.model = VisionModel.load_model(str(self.model_path), device=self.device)
        self.model.eval()
        with open(self.model_path / LABEL_FILE, "r", encoding="utf-8") as f:
            self.top_tags = [line.strip() for line in f if line.strip()]
        print(f"[JoyTag] Loaded {len(self.top_tags)} labels")
        print(f"[JoyTag] Image size: {self.model.image_size}")
        print("[JoyTag] Ready!")

    def _ensure_model_files(self):
        self.model_path.mkdir(parents=True, exist_ok=True)
        required = {
            "config.json": "config.json",
            "model.safetensors": "model.safetensors",
            LABEL_FILE: "top_tags.txt",
        }
        missing = [name for name in required if not (self.model_path / name).exists() or (self.model_path / name).stat().st_size == 0]
        if not missing:
            return

        print("[JoyTag] Model files missing. Downloading the official model files...", flush=True)
        try:
            from huggingface_hub import hf_hub_download
            for local_name in missing:
                source_name = required[local_name]
                downloaded = hf_hub_download(
                    repo_id=JOYTAG_REPO,
                    filename=source_name,
                    local_dir=str(self.model_path),
                )
                downloaded_path = Path(downloaded)
                target = self.model_path / local_name
                if downloaded_path != target:
                    downloaded_path.replace(target)
                print(f"[JoyTag] Downloaded {local_name}", flush=True)
        except Exception as exc:
            raise RuntimeError(
                "JoyTag model setup failed. The application could not download its required model files "
                "from the official model repository. Check your internet connection and try again."
            ) from exc

        still_missing = [name for name in required if not (self.model_path / name).exists() or (self.model_path / name).stat().st_size == 0]
        if still_missing:
            raise RuntimeError(f"JoyTag model setup incomplete: missing {', '.join(still_missing)}")

    @staticmethod
    def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
        image = image.convert("RGB")
        width, height = image.size
        max_dim = max(width, height)
        pad_left = (max_dim - width) // 2
        pad_top = (max_dim - height) // 2
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(image, (pad_left, pad_top))
        padded = padded.resize((target_size, target_size), Image.BICUBIC)
        tensor = TVF.pil_to_tensor(padded).float() / 255.0
        return TVF.normalize(
            tensor,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    @torch.inference_mode()
    def predict_batch(self, images: list[Image.Image], threshold: float | None = None) -> list[list[dict[str, Any]]]:
        if not images:
            return []
        if threshold is None:
            threshold = self.threshold
        tensor = torch.stack([self.prepare_image(img, self.model.image_size) for img in images]).to(self.device)
        with torch.amp.autocast(device_type="cuda", enabled=self.device == "cuda"):
            predictions = self.model({"image": tensor})
        probabilities = predictions["tags"].sigmoid().float().cpu()
        all_results: list[list[dict[str, Any]]] = []
        for row in probabilities:
            results = []
            for index, probability in enumerate(row):
                if index >= len(self.top_tags):
                    break
                score = float(probability)
                if score >= threshold:
                    results.append({"tag": self.top_tags[index], "confidence": score, "source": "joytag"})
            results.sort(key=lambda item: item["confidence"], reverse=True)
            all_results.append(results)
        return all_results

    @torch.inference_mode()
    def predict(self, image: Image.Image, threshold: float | None = None):
        return self.predict_batch([image], threshold=threshold)[0]

    @torch.inference_mode()
    def predict_character(self, image: Image.Image, threshold: float | None = None):
        return self.predict(image, threshold=threshold)
